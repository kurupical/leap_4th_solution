import sys

sys.path.append("./")
from leap.src.model import (
    main,
    Config,
    FeatureEngineeringPipeline,
    LabelPipeline,
    ScalerPipeline,
)
import transformers

# ---------------------------------------------------------------
# 1. convnext 64x3-128x3-256x27-512x3 cv: 0.7881 / lb: 0.78577
# ---------------------------------------------------------------
config = Config(
    exp_name=f"exp042_70m_cnn64_smoothl1beta1_lr2.5e-3_beta0.01_wd0.05",
    dims_1dconvnext=(64, 128, 256, 512),
    depths_1dconvnext=(3, 3, 27, 3),
    model_name="1dconvnext",
    batch_size=4,
    train_files=80,
    ema_decays=[0.995, 0.999],
    block_kernel_size_1dconvnext=15,
    epochs=7,
    lr=2.5e-3,
    weight_decay=0.05,
    beta_smoothl1=0.01,
    head_1dconvnext="Head1D",
    final_layer_params_1dconvnext={
        "out_channels": 14,
        "kernel_size": 1,
        "stride": 1,
        "padding": "same",
    },
)
main(config)

# ---------------------------------------------------------------
# 2. convnext 96x3-192x3-384x27-768x3 cv: 0.7887 / lb: 0.78697
# ---------------------------------------------------------------
config = Config(
    exp_name=f"exp042_70m_cnn96_smoothl1beta1_lr2e-3_beta0.01_wd0.05",
    dims_1dconvnext=(96, 192, 384, 768),
    depths_1dconvnext=(3, 3, 27, 3),
    model_name="1dconvnext",
    batch_size=4,
    train_files=80,
    ema_decays=[0.995, 0.999],
    block_kernel_size_1dconvnext=15,
    epochs=7,
    lr=2e-3,
    weight_decay=0.05,
    beta_smoothl1=0.01,
    head_1dconvnext="Head1D",
    block_type_1dconvnext="doubleblock",
    final_layer_params_1dconvnext={
        "out_channels": 14,
        "kernel_size": 1,
        "stride": 1,
        "padding": "same",
    },
)
main(config)

# ---------------------------------------------------------------
# 3. convnext 128x3-256x3-512x27-1024x3 cv: 0.7869 / lb: (no submission)
# ---------------------------------------------------------------
config = Config(
    exp_name=f"exp042_70m_cnn128_smoothl1beta1_lr2e-3_beta0.01_wd0.05",
    dims_1dconvnext=(128, 256, 512, 768),
    depths_1dconvnext=(3, 3, 27, 3),
    model_name="1dconvnext",
    batch_size=4,
    train_files=65,
    ema_decays=[0.995, 0.999, 0.9999],
    block_kernel_size_1dconvnext=15,
    epochs=7,
    lr=2e-3,
    weight_decay=0.05,
    beta_smoothl1=0.01,
    head_1dconvnext="Head1D",
    block_type_1dconvnext="doubleblock",
    final_layer_params_1dconvnext={
        "out_channels": 14,
        "kernel_size": 1,
        "stride": 1,
        "padding": "same",
    },
)
main(config)

# ---------------------------------------------------------------
# 4. convnext 144x3-288x3-576x27-1152x3 cv: 0.7855 / lb: (no submission)
# ---------------------------------------------------------------
# note: train 8 x RTX4090 with DDP

config = Config(
    exp_name=f"exp042_70m_cnn144_smoothl1beta1_lr1.5e-3_beta0.01_wd0.05_ddp",
    dims_1dconvnext=(128 + 16, 256 + 32, 512 + 64, 1024 + 128),
    # dims_1dconvnext=(32, 64, 128, 256),
    depths_1dconvnext=(3, 3, 27, 3),
    model_name="1dconvnext",
    batch_size=3,
    train_files=80,
    ema_decays=[0.995, 0.999],
    block_kernel_size_1dconvnext=15,
    epochs=7,
    lr=1.5e-3,
    weight_decay=0.05,
    beta_smoothl1=0.01,
    head_1dconvnext="Head1D",
    final_layer_params_1dconvnext={
        "out_channels": 14,
        "kernel_size": 1,
        "stride": 1,
        "padding": "same",
    },
)
main(config)

# ---------------------------------------------------------------
# 5. transformer 512x4 cv: 0.7848 / lb: 0.78341
# ---------------------------------------------------------------
hidden_dims = 512
n_layers = 4
lr = 1e-3
config = Config(
    exp_name=f"exp042_70m_transformer_{hidden_dims}x{n_layers}_lr{lr}_beta1",
    model_name="1dtransformer",
    batch_size=4,
    train_files=80,
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
        {"out_channels": 256, "kernel_size": 3},
        {"out_channels": 256, "kernel_size": 3},
        {"out_channels": 256, "kernel_size": 3},
    ],
)
main(config)

# ---------------------------------------------------------------
# 6. transformer 768x4 cv: 0.7843 / lb: (no submission)
# ---------------------------------------------------------------
hidden_dims = 768
n_layers = 4
lr = 1e-3
config = Config(
    exp_name=f"exp042_70m_transformer_{hidden_dims}x{n_layers}_lr{lr}_beta1",
    model_name="1dtransformer",
    batch_size=4,
    train_files=80,
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
        {"out_channels": 256, "kernel_size": 3},
        {"out_channels": 256, "kernel_size": 3},
        {"out_channels": 256, "kernel_size": 3},
    ],
)
main(config)

# ---------------------------------------------------------------
# 7. convnext 144x3-288x3-576x27-1152x3 cv: 0.78769 / lb: (no submission)
# ---------------------------------------------------------------
# note: train 8 x RTX4090 with DDP

config = Config(
    exp_name=f"exp042_70m_cnn144_smoothl1beta1_lr2e-3_beta0.01_wd0.075_ddp",
    dims_1dconvnext=(128 + 16, 256 + 32, 512 + 64, 1024 + 128),
    depths_1dconvnext=(3, 3, 27, 3),
    model_name="1dconvnext",
    batch_size=3,
    train_files=80,
    ema_decays=[0.999],
    block_kernel_size_1dconvnext=15,
    epochs=4,
    lr=2e-3,
    weight_decay=0.075,
    beta_smoothl1=0.01,
    head_1dconvnext="Head1D",
    final_layer_params_1dconvnext={
        "out_channels": 14,
        "kernel_size": 1,
        "stride": 1,
        "padding": "same",
    },
)
main(config)
