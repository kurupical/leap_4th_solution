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

# import pandas as pd
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

cache_path = "input/leap-atmospheric-physics-ai-climsim/cache"
exp_name = "exp042"

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
    model_name: str = "1dconvnext"

    optimizer: Any = torch.optim.AdamW
    scheduler: Any = transformers.get_polynomial_decay_schedule_with_warmup
    power_polynomical_decay: int = 2

    # ema
    experiment_no_ema_model: bool = True
    ema_decays: List[float] = dataclasses.field(default_factory=lambda: [])

    # 1d-convnext
    depths_1dconvnext: Tuple[int] = (3, 3, 27, 3)
    kernel_sizes_1dconvnext: Tuple[int] = (4, 2, 2, 2)
    dims_1dconvnext: Tuple[int] = (96, 192, 384, 768)
    head_1dconvnext: nn.Module = "Head1Dv2"
    final_layer_name_1dconvnext: nn.Module = nn.LazyConv1d
    final_layer_params_1dconvnext: List[Dict] = dataclasses.field(
        default_factory=lambda: [
            {"dim": 256, "kernel_size": 3, "drop_path": 0},
            {"dim": 256, "kernel_size": 3, "drop_path": 0},
            {"dim": 256, "kernel_size": 3, "drop_path": 0},
        ]
    )
    final_layer_2d_params_1dconvnext: List[Dict] = dataclasses.field(
        default_factory=lambda: [
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": "same"},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": "same"},
            {"out_channels": 1, "kernel_size": 1, "stride": 1, "padding": "same"},
        ]
    )
    pointwise_1dconvnext: nn.Module = "linear"
    block_type_1dconvnext: str = "pointwise"
    kernel_size_60x60_1dconvnext: int = 1
    block_kernel_size_1dconvnext: int = 11

    sequence_model: nn.Module = nn.GRU
    apply_seq_before_skip: bool = True

    drop_path_rate_1dconvnext: float = 0
    dropout_head_1dconvnext: float = 0.0

    # Transformer
    hidden_dims_transformer: int = 256
    n_layers_transformer: int = 4
    n_heads_transformer: int = 32
    head_mode_transformer: int = "nn"
    num_layers_head_transformer: int = 1
    model_name_transformer: str = "transformer_ln"
    dropout_transformer: float = 0
    stem_mode_transformer: str = "nn"
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
    hidden_dims_independent_transformer: int = 4

    # Scaler config
    clip_range: float = 100.0

    # loss
    loss: Any = nn.SmoothL1Loss
    beta_smoothl1: float = 0.01

    # fp16
    ds_dtype: str = "float32"


class LEAPDataset1D(Dataset):
    def __init__(
        self,
        files: np.array,
        config: Config,
        test: bool = False,
    ):
        self.files = files
        self.test = test
        self.config = config

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        feature, label, sample_id = load_features(
            fname=self.files[index],
            debug=self.config.debug,
            is_test=self.test,
        )
        if label is None:
            label = np.zeros(feature.shape)
        if sample_id is None:
            sample_id = np.zeros(feature.shape[0])
        n_sequential_cols = feature.shape[1] // 60
        feature = feature.astype(np.float32)
        feature = np.concatenate(
            [
                feature[:, : 60 * n_sequential_cols].reshape(-1, n_sequential_cols, 60),
                np.repeat(feature[:, 60 * n_sequential_cols :], 60, axis=1).reshape(
                    -1, 16, 60
                ),
            ],
            axis=1,
        )

        return {
            "feature": feature,
            "label": label,
            "sample_id": sample_id,
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


class Head1D(nn.Module):
    def __init__(self, final_layer, final_layer_params, dropout_head=0.0):
        super(Head1D, self).__init__()
        self.final_layer = final_layer(**final_layer_params)
        self.dout = nn.Dropout(dropout_head)
        self.fc = nn.Linear(8 * 60, 8)

    def forward(self, x):
        x = self.final_layer(x)  # (hidden_size, 60) -> (14, 60)
        x = self.dout(x)
        x_out = torch.cat(
            [
                x[:, :6, :].reshape(
                    -1, 6 * 60
                ),  # shape = (bs, 360) ptend_t, ptend_q0001, ptend_q0002, ptend_q0003, ptend_u, ptend_v
                self.fc(x[:, 6:, :].reshape(-1, 8 * 60)),  # shape = (bs, 8)
            ],
            dim=1,
        )  # shape = (bs, 368)
        return x_out


class PointWiseLinear(nn.Module):
    def __init__(self, dim):
        super(PointWiseLinear, self).__init__()
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pwconv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.act(x)
        x = self.pwconv2(x.permute(0, 2, 1))
        return x


# https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py
class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        pointwise=PointWiseLinear,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding="same", stride=1, groups=dim
        )  # depthwise conv
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.pwconv = pointwise(dim=dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)
        x = input + self.drop_path(x)
        return x


class DoubleBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        pointwise=PointWiseLinear,
        kernel_size_60x60=1,
    ):
        super().__init__()
        self.convkx1 = nn.Conv1d(
            dim,
            dim // 2,
            kernel_size=kernel_size,
            padding="same",
            stride=1,
            groups=dim // 2,
        )  # depthwise conv
        self.act = nn.GELU()

        self.conv1xk = nn.Conv1d(60, 60, kernel_size=4, padding=1, stride=2, groups=60)
        self.normkx1 = nn.BatchNorm1d(dim // 2)
        self.norm1xk = nn.BatchNorm1d(60)
        self.pwconv = pointwise(dim=dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x1 = self.convkx1(x)  # shape = (bs, dim, 60) -> (bs, dim//2, 60)
        x1 = self.normkx1(x1)
        x2 = x.permute(0, 2, 1)
        x2 = self.conv1xk(x2)  # shape = (bs, 60, dim) -> (bs, 60, dim//2)
        x2 = self.norm1xk(x2)  # shape = (bs, 60, dim) -> (bs, 60, dim//2)
        x2 = x2.permute(0, 2, 1)

        x = torch.cat([x1, x2], dim=1)  # shape = (bs, dim, 60)

        x = self.pwconv(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # shape = (bs, dim, 60)
        x = input + self.drop_path(x)
        return x


class Permute(nn.Module):
    def __init__(self, permute_dims=(0, 2, 1)):
        super(Permute, self).__init__()
        self.permute_dims = permute_dims

    def forward(self, x):
        return x.permute(self.permute_dims)


class ConvNeXt1D(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        depths=[3, 3, 9, 3],
        kernel_sizes=[4, 2, 2, 2],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        final_layer=nn.LazyConv1d,
        final_layer_params={
            "out_channels": 14,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        },
        pointwise="linear",
        block_type: str = "pointwise",
        block_kernel_size: int = 7,
        dropout_head: float = 0.0,
    ):
        super().__init__()

        pointwise = PointWiseLinear
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.LazyConv1d(
                dims[0],
                kernel_size=kernel_sizes[0],
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_sizes[i + 1],
                    stride=1,
                    padding="same",
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            if block_type == "pointwise":
                blocks = [
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        pointwise=pointwise,
                        kernel_size=block_kernel_size,
                    )
                    for j in range(depths[i])
                ]
            elif block_type == "doubleblock":
                blocks = [
                    DoubleBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        pointwise=pointwise,
                        kernel_size=block_kernel_size,
                    )
                    for j in range(depths[i])
                ]
            else:
                raise ValueError(f"block_type: {block_type} is not supported.")
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.final_layer = Head1D(final_layer, final_layer_params, dropout_head)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = x["feature"]  # shape = (bs, 384, n_features)

        x_features = self.forward_features(x)
        x_pred = self.final_layer(x_features)
        return x_pred


class TransformerModel(nn.Module):
    def __init__(
        self,
        hidden_dims: int,
        n_layers: int,
        n_heads: int,
        dropout: int = 0,
        head_cnn_params: List[Dict] = None,
    ):
        super(TransformerModel, self).__init__()

        self.hidden_dims = hidden_dims

        # stem
        self.fc_stem = nn.Sequential(
            nn.LazyLinear(hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.GELU(),
            nn.LazyLinear(hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.GELU(),
        )
        self.position_encoder = nn.Embedding(60, hidden_dims)

        # head
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dims,
            nhead=n_heads,
            dim_feedforward=hidden_dims * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # tail
        self.fc_head_list_sequential = []
        for _ in range(6):
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
            fc_head.append(nn.LazyConv1d(1, kernel_size=3, stride=1, padding="same"))
            fc_head = nn.Sequential(*fc_head)
            self.fc_head_list_sequential.append(fc_head)
        self.fc_head_list_sequential = nn.ModuleList(self.fc_head_list_sequential)
        self.fc_head_scalar = nn.Linear(hidden_dims, 8)

    def forward(self, x):
        # x.shape = (bs, n_features, 60)
        x = x["feature"].permute(0, 2, 1)  # (bs, 60, n_features)
        x = self.fc_stem(x)  # (bs, 60, n_features) -> (bs, 60, hidden_dims)
        pe = self.position_encoder(
            torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        )  # (60) -> (bs, 60, hidden_dims)
        x = x + pe
        x = self.transformer(x)  # (bs, 60, hidden_dims) -> (bs, 60, hidden_dims)

        x_out = []
        for i in range(6):
            x_out_ = self.fc_head_list_sequential[i](
                x.permute(0, 2, 1)
            )  # (bs, hidden_dims, 60) -> (bs, 1, 60)
            x_out_ = x_out_.squeeze(1)  # (bs, 1, 60) -> (bs, 60)
            x_out.append(x_out_)

        x_out.append(self.fc_head_scalar(x.mean(dim=1)))  # (bs, hidden_dims) -> (bs, 8)
        x_out = torch.cat(x_out, dim=1)  # (bs, 60*6+8)
        return x_out


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
            if k != "sample_id":
                data[k] = v.to(device)

        data["feature"] = torch.cat(
            [d for d in data["feature"]]
        )  # shape = (bs*384, n_features)
        data["label"] = torch.cat([d for d in data["label"]])  # shape = (bs*384, 368)
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
                if k != "sample_id":
                    data[k] = v.to(device)
            data["feature"] = torch.cat(
                [d for d in data["feature"]]
            )  # shape = (bs*384, n_features)
            data["label"] = torch.cat(
                [d for d in data["label"]]
            )  # shape = (bs*384, 368)
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


class FeatureEngineeringPipeline:
    def __init__(self):
        pass

    def _feature_engineering(self, df: pl.DataFrame):
        self.feature_cols = []

        for group_col in [
            "state_q0002",
            "state_q0003",
        ]:
            for i in range(60):
                col = f"{group_col}_{i}"
                new_series = (df[col] / -1200).alias(col)
                df = df.with_columns(new_series)

        for group_col in [
            "state_t",
            "state_q0001",
            "state_q0002",
            "state_q0003",
            "state_u",
            "state_v",
            "pbuf_ozone",
            "pbuf_CH4",
            "pbuf_N2O",
        ]:

            cols = [f"{group_col}_{i}" for i in range(60)]

            for i in range(60):
                col = f"{group_col}_{i}"
                diff_mean_col = f"{group_col}_diff_mean_{i}"
                new_series = (df[col] - df[cols].mean(axis=1)).alias(diff_mean_col)
                df = df.with_columns(new_series)
                self.feature_cols.append(diff_mean_col)

            for i in range(60):
                col_name = f"{group_col}_{i}_diff"
                if i == 59:
                    new_series = (
                        df.get_column(f"{group_col}_59")
                        - df.get_column(f"{group_col}_58")
                    ).alias(col_name)
                else:
                    new_series = (
                        df.get_column(f"{group_col}_{i+1}")
                        - df.get_column(f"{group_col}_{i}")
                    ).alias(col_name)
                df = df.with_columns(new_series)
                self.feature_cols.append(col_name)

            for i in range(60):
                col_name = f"{group_col}_{i}_diff2"
                if i == 59:
                    new_series = (
                        df.get_column(f"{group_col}_59")
                        - df.get_column(f"{group_col}_57")
                    ).alias(col_name)
                elif i == 0:
                    new_series = (
                        df.get_column(f"{group_col}_2")
                        - df.get_column(f"{group_col}_0")
                    ).alias(col_name)
                else:
                    new_series = (
                        df.get_column(f"{group_col}_{i+1}")
                        - df.get_column(f"{group_col}_{i-1}")
                    ).alias(col_name)
                df = df.with_columns(new_series)
                self.feature_cols.append(col_name)

        for i in range(60):
            col_name = f"state_q_mean_{i}"
            new_series = (
                df[[f"state_q0002_{i}", f"state_q0003_{i}"]]
                .mean(axis=1)
                .alias(col_name)
            )
            df = df.with_columns(new_series)
            self.feature_cols.append(col_name)

        for q in ["0002", "0003"]:
            for i in range(60):
                col_name = f"state_q{q}_{i}_diff_mean"
                new_series = (
                    df.get_column(f"state_q{q}_{i}")
                    - df.get_column(f"state_q_mean_{i}")
                ).alias(col_name)
                df = df.with_columns(new_series)
                self.feature_cols.append(col_name)

        for i in range(60):
            col_name = f"state_uv_mean_{i}"
            new_series = (
                df[[f"state_u_{i}", f"state_v_{i}"]].mean(axis=1).alias(col_name)
            )
            df = df.with_columns(new_series)
            self.feature_cols.append(col_name)
        for uv in ["u", "v"]:
            for i in range(60):
                col_name = f"state_{uv}_{i}_diff_mean"
                new_series = (
                    df.get_column(f"state_{uv}_{i}")
                    - df.get_column(f"state_uv_mean_{i}")
                ).alias(col_name)
                df = df.with_columns(new_series)
                self.feature_cols.append(col_name)

        for i in range(60):
            col_name = f"pbuf_mean_{i}"
            new_series = (
                df[[f"pbuf_ozone_{i}", f"pbuf_CH4_{i}", f"pbuf_N2O_{i}"]]
                .mean(axis=1)
                .alias(col_name)
            )
            df = df.with_columns(new_series)
            self.feature_cols.append(col_name)

        for pbuf in ["ozone", "CH4", "N2O"]:
            for i in range(60):
                col_name = f"pbuf_{pbuf}_{i}_diff_mean"
                new_series = (
                    df.get_column(f"pbuf_{pbuf}_{i}") - df.get_column(f"pbuf_mean_{i}")
                ).alias(col_name)
                df = df.with_columns(new_series)
                self.feature_cols.append(col_name)

        self.feature_cols = (
            FEATURE_COLUMNS[: 60 * 9] + self.feature_cols + FEATURE_COLUMNS[60 * 9 :]
        )
        return df

    def fit(self, df: pl.DataFrame):
        pass

    def transform(self, df: pl.DataFrame):
        df = self._feature_engineering(df)
        return df


class ScalerPipeline:
    def __init__(self, feature_cols, clip_range):
        self.scaler = StandardScaler()
        self.feature_cols = feature_cols
        self.clip_range = clip_range

    def fit(self, df: pl.DataFrame):
        self.scaler.fit(df[self.feature_cols].to_numpy())

    def transform(self, df: pl.DataFrame):
        df[self.feature_cols] = np.clip(
            self.scaler.transform(df[self.feature_cols].to_numpy()).astype(np.float32),
            -self.clip_range,
            self.clip_range,
        )
        return df


class LabelPipeline:
    def __init__(self):
        self.sc = StandardScaler()

    def fit(self, df: pl.DataFrame):
        self.sc.fit(df[TARGET_COLUMNS].to_numpy())

    def transform(self, df: pl.DataFrame):
        df[TARGET_COLUMNS] = self.sc.transform(df[TARGET_COLUMNS].to_numpy())
        return df

    def inverse_transform(self, df: pl.DataFrame):
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
        return df


def create_features_1d(X):
    n_sequential_cols = len(X) // 60
    X = np.concatenate(
        [
            X[: 60 * n_sequential_cols].reshape(n_sequential_cols, 60),
            np.repeat(X[60 * n_sequential_cols :], 60, axis=0).reshape(16, 60),
        ],
        axis=0,
    )
    return X


def save_features(df, output_fname, feature_cols, is_test=False, is_shuffle=False):

    features = df[feature_cols].to_numpy().astype(np.float32)
    chunk = 96
    iter_num = len(df) // chunk
    if len(df) % chunk != 0:
        iter_num += 1

    if not is_test:
        labels = df[TARGET_COLUMNS].to_numpy().astype(np.float32)

    if is_shuffle:
        idx = np.random.permutation(len(df))
        features = features[idx]
        if not is_test:
            labels = labels[idx]
    sample_ids = [f"E3SM-MMF_{str(i).zfill(10)}" for i in np.arange(len(df))]

    for i in range(iter_num):
        sample_id = sample_ids[i * chunk]
        feature = features[i * chunk : (i + 1) * chunk]

        output_dir = f"{output_fname}/{sample_id}"
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/feature.npy", feature)
        if not is_test:
            label = labels[i * chunk : (i + 1) * chunk]
            np.save(f"{output_dir}/label.npy", label)
    # df.write_parquet(f"{output_fname}.parquet")


def load_features(fname, debug, is_test=False):
    features = np.load(f"{fname}/feature.npy")
    if not is_test:
        labels = np.load(f"{fname}/label.npy")
    else:
        labels = None
    if os.path.isfile(f"{fname}/sample_id.npy"):
        sample_ids = np.load(f"{fname}/sample_id.npy", allow_pickle=True).tolist()
    else:
        sample_ids = None
    # if debug:
    #     # npy ファイル削除
    #     os.remove(f"{fname}_features.npy")
    #     if not is_test:
    #         os.remove(f"{fname}_labels.npy")

    return features, labels, sample_ids


def feature_engineering(config: Config, logger: Logger):

    # train_0.parquet: random 5% training data
    df_train = pl.concat(
        [
            pl.read_parquet(
                f"input/leap-atmospheric-physics-ai-climsim/random/train_0.parquet",
            )
            for i in tqdm.tqdm(range(1), desc="read train data...")
        ]
    )

    df_valid = pl.read_parquet(
        "input/leap-atmospheric-physics-ai-climsim/valid.parquet"
    )

    df_test = pl.read_parquet("input/leap-atmospheric-physics-ai-climsim/test.parquet")

    if config.debug:
        df_train = df_train.head(384 * 4)
        df_valid = df_valid.head(384 * 4)
        df_test = df_test.head(384 * 4)
        config.epochs = 3
        config.train_files = 1

    cache_path_valid = f"{cache_path}/{exp_name}/valid"
    cache_path_test = f"{cache_path}/{exp_name}/test"
    cache_path_fe_pipe = f"{cache_path}/{exp_name}/fe_pipe.pickle"
    cache_path_label_pipe = f"{cache_path}/{exp_name}/label_pipe.pickle"
    cache_path_sc_pipe = f"{cache_path}/{exp_name}/sc_pipe.pickle"

    if config.debug:
        cache_path_valid += "_debug"
        cache_path_test += "_debug"
        cache_path_fe_pipe += "_debug"
        cache_path_label_pipe += "_debug"
        cache_path_sc_pipe += "_debug"

    if (
        os.path.exists(cache_path_fe_pipe)
        and os.path.exists(cache_path_label_pipe)
        and os.path.exists(cache_path_sc_pipe)
        and not config.debug_fe
    ):
        logger.info("load valid/test/pipeline cache")
        with open(cache_path_fe_pipe, "rb") as f:
            fe_pipe = pickle.load(f)

        with open(cache_path_label_pipe, "rb") as f:
            label_pipe = pickle.load(f)

        with open(cache_path_sc_pipe, "rb") as f:
            sc_pipe = pickle.load(f)

    else:
        logger.info("feature engineering")
        fe_pipe = FeatureEngineeringPipeline()
        logger.info("label preprocessing")
        df_train = fe_pipe.transform(df_train)
        df_valid = fe_pipe.transform(df_valid)
        df_test = fe_pipe.transform(df_test)

        sc_pipe = ScalerPipeline(
            feature_cols=fe_pipe.feature_cols, clip_range=config.clip_range
        )
        sc_pipe.fit(
            df=pl.concat(
                [
                    df_train[fe_pipe.feature_cols],
                    df_valid[fe_pipe.feature_cols].head(50000),
                    df_test[fe_pipe.feature_cols],
                ],
            )
        )
        df_train = sc_pipe.transform(df_train)
        df_valid = sc_pipe.transform(df_valid)
        df_test = sc_pipe.transform(df_test)

        label_pipe = LabelPipeline()
        label_pipe.fit(df=df_train)
        df_train = label_pipe.transform(df_train)
        df_valid = label_pipe.transform(df_valid)

        logger.info("save cache")
        # df_train.write_parquet(cache_path_train)
        save_features(df_valid, cache_path_valid, fe_pipe.feature_cols)
        save_features(df_test, cache_path_test, fe_pipe.feature_cols, is_test=True)

        with open(cache_path_fe_pipe, "wb") as f:
            pickle.dump(fe_pipe, f)

        with open(cache_path_label_pipe, "wb") as f:
            pickle.dump(label_pipe, f)

        with open(cache_path_sc_pipe, "wb") as f:
            pickle.dump(sc_pipe, f)

    X_train, y_train = [], []

    files = glob.glob("input/climsim_low-res/npy/*_input.npy")
    files = [
        f.replace("_input.npy", "")
        for f in files
        if "0007-08" not in f
        and "0007-09" not in f
        and "0007-10" not in f
        and "0007-11" not in f
        and "0007-12" not in f
        and "0008-01" not in f
    ]
    fold = KFold(80, shuffle=True, random_state=19900222)

    for fold, (_, valid_idx) in enumerate(fold.split(files)):
        if fold >= config.train_files:
            continue
        cache_path_train = f"{cache_path}/{exp_name}/train_{fold}"
        if config.debug:
            cache_path_train += "_debug"

        files_train = np.array(files)[valid_idx]
        if not os.path.exists(f"{cache_path_train}/finished") or (config.debug_fe):
            if config.debug:
                files_train = files_train[:32]
            logger.info(f"train_{fold} processing... n_records={len(files_train)*384}")

            npy_files = [
                [f"{os.path.basename(f)}_input.npy".replace("_input.npy", "")] * 384
                for f in files_train
            ]
            npy_files = np.array(npy_files).reshape(-1, 1)
            input_ary = np.concatenate([np.load(f"{f}_input.npy") for f in files_train])
            output_ary = np.concatenate(
                [np.load(f"{f}_output.npy") for f in files_train]
            )

            # input_ary は余計な項目入ってるから削除
            ary = np.concatenate([input_ary, output_ary], axis=1)

            feature_columns = (
                [f"state_t_{i}" for i in range(60)]
                + [f"state_q0001_{i}" for i in range(60)]
                + [f"state_q0002_{i}" for i in range(60)]
                + [f"state_q0003_{i}" for i in range(60)]
                + [f"state_u_{i}" for i in range(60)]
                + [f"state_v_{i}" for i in range(60)]
                + [
                    "state_ps",
                    "pbuf_SOLIN",
                    "pbuf_LHFLX",
                    "pbuf_SHFLX",
                    "pbuf_TAUX",
                    "pbuf_TAUY",
                    "pbuf_COSZRS",
                    "cam_in_ALDIF",
                    "cam_in_ALDIR",
                    "cam_in_ASDIF",
                    "cam_in_ASDIR",
                    "cam_in_LWUP",
                    "cam_in_ICEFRAC",
                    "cam_in_LANDFRAC",
                    "cam_in_OCNFRAC",
                    "cam_in_SNOWHICE",
                    "cam_in_SNOWHLAND",
                ]
                + [f"pbuf_ozone_{i}" for i in range(60)]
                + [f"pbuf_CH4_{i}" for i in range(60)]
                + [f"pbuf_N2O_{i}" for i in range(60)]
                + TARGET_COLUMNS
            )
            df_train = pl.DataFrame(ary, schema=feature_columns)
            new_series = pl.Series(npy_files.reshape(-1)).alias("sample_id")
            df_train = df_train.with_columns(new_series)
            # sample_id を 先頭に
            df_train = df_train[
                [
                    "sample_id",
                    *feature_columns,
                ]
            ]
            df_train = fe_pipe.transform(df_train)
            df_train = label_pipe.transform(df_train)
            df_train = sc_pipe.transform(df_train)
            save_features(
                df_train, cache_path_train, fe_pipe.feature_cols, is_shuffle=True
            )

            # finished ファイルを作成
            with open(f"{cache_path_train}/finished", "w") as f:
                f.write("finished")

            del df_train
            gc.collect()


def finished_feature_engineering(train_files):
    for i in range(train_files):
        cache_path_train = f"{cache_path}/{exp_name}/{i}"
        if not os.path.exists(f"{cache_path_train}/finished"):
            return False
    return True


def main(config: Config, inference_mode: bool = False, output_dir: str = None):
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

        if not inference_mode:
            with open(f"{output_dir}/cfg.pickle", "wb") as f:
                pickle.dump(config, f)

        if finished_feature_engineering(config.train_files) and not config.debug:
            logger.info("already finished feature engineering.")
        else:
            feature_engineering(config=config, logger=logger)

        cache_path_label_pipe = f"{cache_path}/{exp_name}/label_pipe.pickle"
        if config.debug:
            cache_path_label_pipe += "_debug"

        with open(cache_path_label_pipe, "rb") as f:
            label_pipe = pickle.load(f)

        if config.model_name == "1dconvnext":
            model = ConvNeXt1D(
                depths=config.depths_1dconvnext,
                kernel_sizes=config.kernel_sizes_1dconvnext,
                dims=config.dims_1dconvnext,
                final_layer=config.final_layer_name_1dconvnext,
                final_layer_params=config.final_layer_params_1dconvnext,
                pointwise=config.pointwise_1dconvnext,
                block_type=config.block_type_1dconvnext,
                block_kernel_size=config.block_kernel_size_1dconvnext,
                dropout_head=config.dropout_head_1dconvnext,
            )
            ds = LEAPDataset1D

        elif config.model_name == "1dtransformer":
            model = TransformerModel(
                hidden_dims=config.hidden_dims_transformer,
                n_layers=config.n_layers_transformer,
                n_heads=config.n_heads_transformer,
                dropout=config.dropout_transformer,
                head_cnn_params=config.head_cnn_params_transformer,
            )
            ds = LEAPDataset1D
        else:
            raise ValueError(f"invalid model_name: {config.model_name}")

        cache_pathes_train = [
            f"{cache_path}/{exp_name}/train_{i}" for i in range(config.train_files)
        ]
        cache_path_valid = f"{cache_path}/{exp_name}/valid"
        cache_path_test = f"{cache_path}/{exp_name}/test"

        if config.debug:
            cache_pathes_train = [
                f"{cache_path}/{exp_name}/train_{i}_debug"
                for i in range(config.train_files)
            ]
            cache_path_valid += "_debug"
            cache_path_test += "_debug"

        train_files = []
        for path in cache_pathes_train:
            train_files.extend(glob.glob(f"{path}/E3SM-MMF*"))
        train_dataset = ds(
            files=train_files,
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
            files=glob.glob(f"{cache_path_valid}/*"),
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
            files=glob.glob(f"{cache_path_test}/*"),
            config=config,
            test=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(f"test_dataset: {len(test_dataset)}")

        # Initialize lazyModule
        model = model.to(device)
        logger.info("lazy module initialize")
        data = {"feature": torch.FloatTensor(train_dataset[0]["feature"]).to(device)}
        model(data)
        logger.info("lazy module initialize end")

        optimizer = config.optimizer(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        if config.scheduler == transformers.get_polynomial_decay_schedule_with_warmup:
            scheduler = config.scheduler(
                optimizer,
                num_warmup_steps=len(train_loader) * config.warmup_ratio,
                num_training_steps=config.epochs * len(train_loader),
                lr_end=0,
                power=config.power_polynomical_decay,
            )
        else:
            scheduler = config.scheduler(
                optimizer,
                num_warmup_steps=len(train_loader) * config.warmup_ratio,
                num_training_steps=config.epochs * len(train_loader),
            )

        mode = "disabled"
        # mode = "online"
        wandb.init(project="leap", name=config.exp_name, mode=mode)

        for k, v in config.__dict__.items():
            wandb.config.update({k: v})

        best_r2 = -np.inf
        not_improved_cnt = 0
        if config.loss == nn.SmoothL1Loss:
            criterion = config.loss(beta=config.beta_smoothl1)
        else:
            criterion = config.loss()
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
            if inference_mode:
                logger.info("break; inference mode = True")
                break
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
                preds = label_pipe.inverse_transform(
                    pl.DataFrame(preds, schema={c: pl.Float64 for c in TARGET_COLUMNS})
                ).to_numpy()
                labels = label_pipe.inverse_transform(
                    pl.DataFrame(labels, schema={c: pl.Float64 for c in TARGET_COLUMNS})
                ).to_numpy()
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

            if not_improved_cnt > 5:
                break

            gc.collect()
            torch.cuda.empty_cache()

        model.load_state_dict(torch.load(f"{output_dir}/best.pth"))
        if not inference_mode:
            val_loss, preds, labels = eval_fn(val_loader, model, device, criterion)

            df_pred = label_pipe.inverse_transform(
                pl.DataFrame(preds, schema=TARGET_COLUMNS)
            )
            df_valid = pl.read_parquet(
                "input/leap-atmospheric-physics-ai-climsim/valid.parquet"
            ).head(len(df_pred))
            for col in ALL_ZERO_TARGET_COLS:
                df_valid = df_valid.with_columns(df_valid[col].clip(0, 0))
            df_pred = df_pred.with_columns(df_valid["sample_id"])
            labels = label_pipe.inverse_transform(
                pl.DataFrame(labels, schema=TARGET_COLUMNS)
            ).to_numpy()
            r2 = r2_score(labels, df_pred[TARGET_COLUMNS].to_numpy())
            logger.info(f"r2_val: {r2}")
            df_pred.write_parquet(f"{output_dir}/pred_valid.parquet")

        test_loss, preds, labels = eval_fn(
            test_loader, model, device, criterion, is_test=True
        )

        df_pred = label_pipe.inverse_transform(
            pl.DataFrame(preds, schema=TARGET_COLUMNS)
        )
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
        wandb.finish()
        raise e


if __name__ == "__main__":

    hidden_dims = 512
    n_layers = 4
    lr = 0.75e-3
    config = Config(
        exp_name=f"exp042_70m_transformer_{hidden_dims}x{n_layers}_lr{lr}_beta1",
        model_name="1dtransformer",
        batch_size=1,
        train_files=40,
        ema_decays=[0.9999, 0.9995, 0.999, 0.995],
        epochs=7,
        n_heads_transformer=32,
        scheduler=transformers.get_polynomial_decay_schedule_with_warmup,
        power_polynomical_decay=1,
        weight_decay=0.01,
        lr=lr,
        beta_smoothl1=1,
        n_layers_transformer=n_layers,
        hidden_dims_transformer=hidden_dims,
        head_mode_transformer="1dcnn",
        head_cnn_params_transformer=[
            {"out_channels": 512, "kernel_size": 3},
            {"out_channels": 512, "kernel_size": 3},
            {"out_channels": 512, "kernel_size": 3},
        ],
        debug=False,
    )
    main(config)
