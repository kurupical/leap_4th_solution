import sys

sys.path.append("./")

from leap.src.ensemble import (
    main,
    Config,
    LabelPipeline,
    main_10folds,
)
import transformers


for lr in [1e-3]:
    bs = 256
    hidden_dims = [256, 256, 256]
    config = Config(
        exp_name=f"exp302_1dcnn_hidden{hidden_dims}_lr{lr}_bs{bs}_beta1_newdata_40epochs_ks3",
        model_name="1dcnn",
        batch_size=bs,
        epochs=20,
        scheduler=transformers.get_polynomial_decay_schedule_with_warmup,
        kernel_sizes_1dcnn=[3, 3, 3],
        hidden_dims_1dcnn=hidden_dims,
        power_polynomical_decay=1,
        weight_decay=0,
        lr=lr,
        beta_smoothl1=1,
    )
    main_10folds(config)
