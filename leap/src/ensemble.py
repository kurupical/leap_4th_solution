import sys

sys.path.append(".")
from leap.src.const import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    ALL_ZERO_TARGET_COLS,
    LATITUDES,
    LONGITUDES,
    MAP_2D,
)
from functools import partial

from torch.nn.modules.utils import _single
import torch
from torch import nn

import pandas as pd
import polars as pl

from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime as dt
from logging import Logger, StreamHandler, Formatter, FileHandler
from sklearn.model_selection import GroupKFold
import logging
import dataclasses
import tqdm
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from transformers import get_linear_schedule_with_warmup
from timm.models.layers import trunc_normal_, DropPath
import transformers

import numba
from timm.utils.model_ema import ModelEmaV3
import timm
from timm.models.layers import to_2tuple
from torch import Tensor

try:
    import wandb
except Exception as e:
    print(e)
import shutil
import torch.nn.functional as F
import pickle
from torch.optim.lr_scheduler import StepLR, LambdaLR, CyclicLR
from typing import Tuple
import random
import glob
import gc
import copy
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import haversine_distances
from torchvision.models.video.resnet import r2plus1d_18, R2Plus1D_18_Weights

torch.backends.cudnn.benchmark = True
import torch._dynamo

from sklearn.model_selection import KFold

torch._dynamo.config.suppress_errors = True

DEVICE = "cuda"


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def get_logger(output_dir=None, logging_level=logging.INFO):
    formatter = Formatter("%(asctime)s|%(levelname)s| %(message)s")
    logger = Logger(name="log")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging_level)
    logger.addHandler(handler)
    if output_dir is not None:
        now = dt.now().strftime("%Y%m%d%H%M%S")
        file_handler = FileHandler(f"{output_dir}/{now}.txt")
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


@dataclasses.dataclass
class Config:
    exp_name: str
    lr: float = 1e-2
    min_lr: float = 0
    warmup_ratio: float = 0.1
    batch_size: int = 1
    num_workers: int = 0
    weight_decay: float = 0.075
    epochs: int = 6
    debug: bool = False
    debug_fe: bool = False
    train_files: int = 2
    model_name: str = "1dcnn"
    ema_decays: List[float] = dataclasses.field(default_factory=lambda: [])
    experiment_no_ema_model: bool = True

    optimizer: Any = torch.optim.AdamW
    scheduler: Any = transformers.get_polynomial_decay_schedule_with_warmup
    num_cycles: int = 3
    power_polynomical_decay: int = 2
    gamma_step_lr: float = 0.5
    fold: int = 0

    # ModelNN config
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [512, 256, 128])

    # 1D-CNN config
    hidden_dims_1dcnn: List[int] = dataclasses.field(
        default_factory=lambda: [128, 256, 512]
    )
    kernel_sizes_1dcnn: List[int] = dataclasses.field(default_factory=lambda: [3, 3, 3])

    # 2D-CNN config
    hidden_dims_2dcnn: List[int] = dataclasses.field(
        default_factory=lambda: [128, 256, 512]
    )
    kernel_sizes_2dcnn: List[int] = dataclasses.field(default_factory=lambda: [3, 3, 3])

    # Transformer
    hidden_dims_transformer: int = 128
    n_layers_transformer: int = 2
    n_heads_transformer: int = 8
    stem_cnn_params_transformer: List[Dict] = dataclasses.field(
        default_factory=lambda: [
            {"kernel_size": 3},
            {"kernel_size": 3},
        ]
    )
    head_cnn_params_transformer: List[Dict] = dataclasses.field(
        default_factory=lambda: [
            {"kernel_size": 3},
            {"kernel_size": 3},
            {"kernel_size": 3},
        ]
    )

    # loss
    loss: Any = nn.SmoothL1Loss
    beta_smoothl1: float = 0.01

    # fp16
    ds_dtype: str = "float32"


class LEAPDataset1D(Dataset):
    def __init__(
        self,
        X: np.array,
        y: np.array,
        config: Config,
        test: bool = False,
    ):
        self.X = X
        self.y = y
        self.test = test
        self.config = config

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        n_sequential_cols = feature.shape[1] // 60
        feature = feature.astype(np.float32)

        if self.y is None:
            label = np.zeros(feature.shape)
        else:
            label = self.y[index]

        return {
            "feature": feature,
            "label": label,
        }


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Model1DCNN(nn.Module):
    def __init__(self, hidden_dims, kernel_sizes):
        super(Model1DCNN, self).__init__()

        conv = []
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            conv.append(
                nn.LazyConv1d(
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                )
            )
            conv.append(nn.BatchNorm1d(hidden_dim))
            conv.append(nn.GELU())
        conv.append(
            nn.LazyConv1d(out_channels=1, kernel_size=2, stride=1, padding="same")
        )
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        # x.shape = (batch_size, n_models, seq_len=368)
        x = self.conv(
            x["feature"]
        )  # (batch_size, n_models, seq_len=368) -> (batch_size, hidden_size, seq_len=368)
        x = x.mean(
            dim=1
        )  # (batch_size, hidden_size, seq_len=368) -> (batch_size, seq_len=368)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        hidden_dims: int,
        n_layers: int,
        n_heads: int,
        head_mode: str,
        num_layers_head: int = 1,
        hidden_dims_independent: int = 4,
        dropout: int = 0,
        model_name: str = "transformer_ln",
        stem_mode: str = "nn",
        stem_cnn_params: List[Dict] = None,
        head_cnn_params: List[Dict] = None,
    ):
        super(TransformerModel, self).__init__()

        self.hidden_dims = hidden_dims

        self.fc_stem = nn.Sequential(
            nn.LazyLinear(hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.GELU(),
            nn.LazyLinear(hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.GELU(),
        )

        self.position_encoder = nn.Embedding(368, hidden_dims)
        self.model_name = model_name
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dims,
            nhead=n_heads,
            dim_feedforward=hidden_dims * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        fc_head = []
        for head_cnn_param in head_cnn_params:
            fc_head.append(
                nn.LazyConv1d(
                    stride=1,
                    padding="same",
                    **head_cnn_param,
                )
            )
            fc_head.append(nn.BatchNorm1d(head_cnn_param["out_channels"]))
            fc_head.append(nn.GELU())
        fc_head.append(nn.LazyConv1d(32, kernel_size=2, stride=1, padding="same"))
        self.fc_head = nn.Sequential(*fc_head)

    def forward(self, x):
        # x.shape = (bs, n_features, 368)
        x = x["feature"].permute(0, 2, 1)  # (bs, 368, n_features)
        x = self.fc_stem(x)  # (bs, 60, n_features) -> (bs, 60, hidden_dims)
        pe = self.position_encoder(
            torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        )  # (60) -> (bs, 60, hidden_dims)
        x = x + pe
        x = self.transformer(x)  # (bs, 60, hidden_dims) -> (bs, 60, hidden_dims)

        x = self.fc_head(x.permute(0, 2, 1))
        x = x.mean(dim=1)
        return x


def train_fn(
    dataloader,
    model,
    optimizer,
    device,
    scheduler,
    epoch,
    criterion,
    fp16,
    ema_models,
):
    model.train()

    loss_meter = AverageMeter()
    data_length = len(dataloader)
    tk0 = tqdm.tqdm(enumerate(dataloader), total=data_length)

    for bi, data in tk0:
        batch_size = len(data)

        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=fp16):
            pred = model(data)
            loss = criterion(pred, data["label"])
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.detach().item(), batch_size)
        wandb.log(
            {
                "epoch": epoch,
                "loss": loss_meter.avg,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        tk0.set_postfix(
            Loss=loss_meter.avg, Epoch=epoch, LR=optimizer.param_groups[0]["lr"]
        )
        for ema_model in ema_models:
            ema_model.update(model)
        scheduler.step()

    return loss_meter.avg


def eval_fn(data_loader, model, device, criterion, is_test=False):
    model.eval()
    loss_meter = AverageMeter()

    data_length = len(data_loader)
    tk0 = tqdm.tqdm(enumerate(data_loader), total=data_length)

    preds = []
    labels = []

    with torch.no_grad():
        for bi, data in tk0:
            batch_size = len(data)
            for k, v in data.items():
                data[k] = v.to(device)
            pred = model(data)
            if not is_test:
                loss = criterion(pred, data["label"])
                loss_meter.update(loss.detach().item(), batch_size)
                labels.append(data["label"].detach().cpu().numpy().reshape(-1, 368))
            tk0.set_postfix(Loss=loss_meter.avg)

            preds.append(pred.detach().cpu().numpy().reshape(-1, 368))

    preds = np.concatenate(preds, axis=0)
    if not is_test:
        labels = np.concatenate(labels, axis=0)

    return loss_meter.avg, preds, labels


class LabelPipeline:
    def __init__(self):
        self.sc = StandardScaler()

    def fit(self, X: np.array):
        self.sc.fit(X)

    def transform(self, X: np.array):
        X = self.sc.transform(X).astype(np.float32)
        return X

    def inverse_transform(self, X: np.array):
        df = pl.DataFrame(X, schema={c: pl.Float64 for c in TARGET_COLUMNS})
        df[TARGET_COLUMNS] = self.sc.inverse_transform(
            df[TARGET_COLUMNS].to_numpy()
        ).astype(np.float32)
        # ALL_ZERO_TARGET_COLS は 0 に戻す
        for col in ALL_ZERO_TARGET_COLS:
            df = df.with_columns(df[col].clip(0, 0))
        for col in [
            "ptend_q0002_12",
            "ptend_q0002_13",
            "ptend_q0002_14",
            "ptend_q0002_15",
            "ptend_q0002_16",
            "ptend_q0002_17",
            "ptend_q0002_18",
            "ptend_q0002_19",
            "ptend_q0002_20",
            "ptend_q0002_21",
            "ptend_q0002_22",
            "ptend_q0002_23",
            "ptend_q0002_24",
            "ptend_q0002_25",
            "ptend_q0002_26",
            "ptend_q0002_27",
        ]:
            df = df.with_columns(df[col].clip(0, 0))
        return df.to_numpy()


def read_inference(file):
    if ".npy" in file:
        ary = np.load(file)
        if debug:
            ary = ary[::10]
        ary = ary / weight.reshape(1, -1)
        ary[np.isnan(ary)] = 0
    elif ".parquet" in file:
        ary = pd.read_parquet(file)[TARGET_COLUMNS]
        if debug:
            ary = ary[::10]
    assert ary.shape == (len(label), 368)


def main(config: Config, output_dir: str = None):
    try:

        seed_everything()
        if output_dir is None:
            output_dir = f"output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.exp_name}"
        os.makedirs(output_dir, exist_ok=True)
        logger = get_logger(output_dir)
        logger.info("start!")
        logger.info(f"config: {config}")
        config.output_dir = output_dir
        shutil.copy(__file__, output_dir)
        device = DEVICE if torch.cuda.is_available() else "cpu"

        with open(f"{output_dir}/cfg.pickle", "wb") as f:
            pickle.dump(config, f)

        # -------------------------------------------------------
        # load data and preprocessing
        # -------------------------------------------------------
        n_data_for_eval = 636640 if not config.debug else 384 * 4
        label = pd.read_parquet(
            "input/leap-atmospheric-physics-ai-climsim/valid.parquet",
        )[TARGET_COLUMNS]
        label = label[-n_data_for_eval:]

        weight = (
            pd.read_parquet(
                "input/leap-atmospheric-physics-ai-climsim/sample_submission.parquet",
            )
            .values[0, 1:]
            .astype(np.float32)
        )

        label_pipe = LabelPipeline()
        label_pipe.fit(label)
        label = label_pipe.transform(label)

        file_dir = "input/submission"
        if os.path.exists(f"{file_dir}/X.npy") and os.path.exists(
            f"{file_dir}/X_test.npy"
        ):
            X = np.load(f"{file_dir}/X.npy")
            X_test = np.load(f"{file_dir}/X_test.npy")
        else:
            # -------------------
            # kurupical
            # -------------------
            kurupical_folders = [
                "input/submission/kurupical/20240703230157_exp042_70m_transformer_512x4_lr0.001_beta1",
                "input/submission/kurupical/20240706022848_exp042_70m_cnn64_smoothl1beta1_lr2.5e-3_beta0.01_wd0.05",
                "input/submission/kurupical/20240705215850_exp042_70m_transformer_768x4_lr0.001_beta1",
                "input/submission/kurupical/20240708233043_exp042_70m_cnn96_smoothl1beta1_lr2e-3_beta0.01_wd0.05",
                "input/submission/kurupical/20240709224049_exp042_70m_cnn128_smoothl1beta1_lr2e-3_beta0.01_wd0.05",
                "input/submission/kurupical/20240713043714_exp042_70m_cnn160_smoothl1beta1_lr2e-3_beta0.01_wd0.05_ddp",
                "input/submission/kurupical/20240714093820_exp042_70m_cnn160_smoothl1beta1_lr2e-3_beta0.01_wd0.05_ddp",
            ]

            X = []
            X_test = []
            for folder in tqdm.tqdm(kurupical_folders):
                pred_val = (
                    pd.read_parquet(f"{folder}/pred_valid.parquet")[TARGET_COLUMNS]
                    .values[-n_data_for_eval:]
                    .astype(np.float64)
                )
                pred_test = pd.read_parquet(f"{folder}/submission.parquet")[
                    TARGET_COLUMNS
                ].values.astype(np.float64)

                pred_val = label_pipe.transform(pred_val)
                pred_test = label_pipe.transform(pred_test)

                X.append(pred_val)
                X_test.append(pred_test)

            # -------------------
            # takoi
            # -------------------

            takoi_files = [
                "input/submission/takoi/exp124_val_preds.npy",
                "input/submission/takoi/exp130_val_preds.npy",
                "input/submission/takoi/exp131_val_preds.npy",
                "input/submission/takoi/exp133_val_preds.npy",
                "input/submission/takoi/exp134_val_preds.npy",
                "input/submission/takoi/exp135_val_preds.npy",
                "input/submission/takoi/exp136_val_preds.npy",
                "input/submission/takoi/exp138_val_preds.npy",
                "input/submission/takoi/exp139_val_preds.npy",
                "input/submission/takoi/exp141_val_preds.npy",
                "input/submission/takoi/exp159_val_preds.npy",
                "input/submission/takoi/exp162_val_preds.npy",
            ]

            for file in tqdm.tqdm(takoi_files):
                pred_val = np.load(file)[-n_data_for_eval:]
                pred_val = pred_val / weight.reshape(1, -1)
                pred_val[np.isnan(pred_val)] = 0

                pred_test = pd.read_parquet(
                    file.replace("exp", "ex").replace("_val_preds.npy", "_pp.parquet")
                )[TARGET_COLUMNS].values

                print(
                    file,
                    pred_val.shape,
                    pred_test.shape,
                    pred_val.dtype,
                    pred_test.dtype,
                )

                pred_val = label_pipe.transform(pred_val)
                pred_test = label_pipe.transform(pred_test)

                X.append(pred_val)
                X_test.append(pred_test)

            # -------------------
            # kami
            # -------------------
            kami_files = [
                "input/submission/kami/kami_experiments_201_unet_multi_all_384_n2_valid_pred.parquet",
                "input/submission/kami/kami_experiments_201_unet_multi_all_512_n3_valid_pred.parquet",
                "input/submission/kami/kami_experiments_201_unet_multi_all_n3_restart2_valid_pred.parquet",
                "input/submission/kami/kami_experiments_201_unet_multi_all_valid_pred.parquet",
                "input/submission/kami/kami_experiments_204_diff_last_all_lr_valid_pred.parquet",
                "input/submission/kami/kami_experiments_217_fix_transformer_leak_all_cos_head64_valid_pred.parquet",
                "input/submission/kami/kami_experiments_217_fix_transformer_leak_all_cos_head64_n4_valid_pred.parquet",
                "input/submission/kami/kami_experiments_222_wo_transformer_all_valid_pred.parquet",
                "input/submission/kami/kami_experiments_222_wo_transformer_all_004_valid_pred.parquet",
                "input/submission/kami/kami_experiments_225_smoothl1_loss_all_005_valid_pred.parquet",
                "input/submission/kami/kami_experiments_225_smoothl1_loss_all_beta_valid_pred.parquet",
            ]

            for file in tqdm.tqdm(kami_files):
                pred_val = pd.read_parquet(file)[TARGET_COLUMNS].values[
                    -n_data_for_eval:
                ]
                pred_val = pred_val / weight.reshape(1, -1)
                pred_val[np.isnan(pred_val)] = 0

                pred_test = pd.read_parquet(
                    file.replace("_valid_pred.parquet", "_submission.parquet")
                )[TARGET_COLUMNS].values

                print(
                    file,
                    pred_val.shape,
                    pred_test.shape,
                    pred_val.dtype,
                    pred_test.dtype,
                )
                pred_val = label_pipe.transform(pred_val)
                pred_test = label_pipe.transform(pred_test)

                X.append(pred_val)
                X_test.append(pred_test)
            X = np.stack(X, axis=1)
            X_test = np.stack(X_test, axis=1)

            np.save(f"{file_dir}/X.npy", X)
            np.save(f"{file_dir}/X_test.npy", X_test)

        if config.debug:
            config.epochs = 3
            X = X[: 384 * 4]
            X_test = X_test[: 384 * 4]

        kfold = KFold(10, shuffle=True, random_state=19900222)

        for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
            if config.fold == fold:
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = label[train_idx], label[valid_idx]
                break
        logger.info(
            f"X_train: {X_train.shape}, X_valid: {X_valid.shape}, y_train: {y_train.shape}, y_valid: {y_valid.shape}, X_test: {X_test.shape}"
        )
        model = Model1DCNN(
            hidden_dims=config.hidden_dims_1dcnn,
            kernel_sizes=config.kernel_sizes_1dcnn,
        )
        ds = LEAPDataset1D

        train_dataset = ds(
            X=X_train,
            y=y_train,
            config=config,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        logger.info(f"train_dataset: {len(train_dataset)}")

        val_dataset = ds(
            X=X_valid,
            y=y_valid,
            config=config,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(f"val_dataset: {len(val_dataset)}")

        test_dataset = ds(
            X=X_test,
            y=None,
            config=config,
            test=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(f"test_dataset: {len(test_dataset)}")

        # Initialize LazyModule
        model = model.to(device)
        logger.info("lazy module initialize")
        data = {
            "feature": torch.stack(
                [
                    torch.FloatTensor(train_dataset[0]["feature"]),
                    torch.FloatTensor(train_dataset[1]["feature"]),
                ],
                dim=0,
            ).to(device)
        }
        model(data)
        logger.info("lazy module initialize end")

        optimizer = config.optimizer(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        scheduler = config.scheduler(
            optimizer,
            num_warmup_steps=len(train_loader) * config.warmup_ratio,
            num_training_steps=config.epochs * len(train_loader),
            lr_end=0,
            power=config.power_polynomical_decay,
        )

        mode = "disabled"
        # mode = "online"
        wandb.init(project="leap_ensemble", name=config.exp_name, mode=mode)

        for k, v in config.__dict__.items():
            wandb.config.update({k: v})

        best_r2 = -np.inf
        not_improved_cnt = 0
        criterion = config.loss(beta=config.beta_smoothl1)
        fp16 = False

        ema_models = []
        # if not config.debug:
        #     model = torch.compile(model, mode="max-autotune")
        for ema_decay in config.ema_decays:
            ema_models.append(ModelEmaV3(model, decay=ema_decay))
        # if not config.debug:
        #     for ema_model in ema_models:
        #         ema_model = torch.compile(ema_model, mode="max-autotune")

        logger.info(f"training start: ema_decays = {config.ema_decays}")
        for epoch in range(config.epochs):
            logger.info(f"epoch: {epoch}")
            train_loss = train_fn(
                train_loader,
                model,
                optimizer,
                device,
                scheduler,
                epoch,
                criterion,
                fp16,
                ema_models,
            )
            if config.experiment_no_ema_model:
                models = [model] + ema_models
                decays = [1.0] + config.ema_decays
            else:
                models = ema_models
                decays = config.ema_decays
            for i, m in enumerate(models):
                logger.info(f"evaluate ema model(decay={decays[i]})")
                val_loss, preds, labels = eval_fn(
                    val_loader,
                    m,
                    device,
                    criterion,
                )
                decay = decays[i]

                # preds = label_pipe.scaler.inverse_transform(preds)
                preds = label_pipe.inverse_transform(preds)
                labels = label_pipe.inverse_transform(labels)
                r2 = r2_score(labels, preds)
                logger.info(f"r2(decay{decay}): {r2}")
                for i, col in enumerate(TARGET_COLUMNS):
                    score = r2_score(labels[:, i], preds[:, i])
                    logger.info(f"{col}: {score}")

                if best_r2 < r2:
                    logger.info(f"model improved: {best_r2} -> {r2}")
                    best_r2 = r2
                    if m is model:
                        torch.save(m.state_dict(), f"{output_dir}/best.pth")
                    else:
                        torch.save(m.module.state_dict(), f"{output_dir}/best.pth")
                    not_improved_cnt = 0
                else:
                    not_improved_cnt += 1 / len(decays)
                    logger.info(f"model not improved: {not_improved_cnt}")
                wandb.log(
                    {
                        "epoch": epoch,
                        f"val_loss/decay{decay}": val_loss,
                        f"r2/decay{decay}": r2,
                    }
                )
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "best_r2": best_r2,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            if not_improved_cnt > 10:
                break
            # if not config.debug and best_r2 < 0.5:
            #     logger.info("r2 < 0.5, break")
            #     break
            # if not config.debug and train_loss > 0.3 and config.loss == nn.SmoothL1Loss:
            #     logger.info("train_loss > 0.3, break")
            #     break
            gc.collect()
            torch.cuda.empty_cache()

        model.load_state_dict(torch.load(f"{output_dir}/best.pth"))
        val_loss, preds, labels = eval_fn(val_loader, model, device, criterion)

        preds = label_pipe.inverse_transform(preds)
        df_pred = pl.DataFrame(preds, schema=TARGET_COLUMNS)
        df_valid = pl.read_parquet(
            "input/leap-atmospheric-physics-ai-climsim/valid.parquet"
        )[-n_data_for_eval:][valid_idx]
        for col in ALL_ZERO_TARGET_COLS:
            df_valid = df_valid.with_columns(df_valid[col].clip(0, 0))
        df_pred = df_pred.with_columns(df_valid["sample_id"])
        r2 = r2_score(df_valid[TARGET_COLUMNS], df_pred[TARGET_COLUMNS].to_numpy())
        logger.info(f"r2_val: {r2}")
        df_pred.write_parquet(f"{output_dir}/pred_valid.parquet")

        test_loss, preds, labels = eval_fn(
            test_loader, model, device, criterion, is_test=True
        )

        preds = label_pipe.inverse_transform(preds)
        df_pred = pl.DataFrame(preds, schema=TARGET_COLUMNS)
        df_sub = pl.read_parquet(
            "input/leap-atmospheric-physics-ai-climsim/sample_submission.parquet"
        ).head(len(df_pred))
        df_pred = df_pred.with_columns(df_sub["sample_id"])

        df_sub[TARGET_COLUMNS] *= df_pred[TARGET_COLUMNS]

        df_sub[["sample_id"] + TARGET_COLUMNS].write_parquet(
            f"{output_dir}/submission.parquet"
        )
        wandb.finish()

    except Exception as e:
        print(e)
        raise e
        wandb.finish()


def main_10folds(config):
    output_dir_base = f"output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.exp_name}"
    exp_name_base = config.exp_name

    for fold in range(10):
        config.fold = fold
        output_dir = f"{output_dir_base}/fold{fold}"
        config.exp_name = f"{exp_name_base}_fold{fold}"
        main(config, output_dir)
