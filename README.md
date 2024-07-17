
# How to reproduce

## Hardware
- local
  - GeForce RTX4090 x1
  - 32core / 256GB RAM
- vast.ai server #1
  - GeForce RTX4090 x2
  - 32core / 128GB RAM
- vast.ai server #2
  - GeForce RTX4090 x8
  - 32core / 774GB RAM

## Environments
- local
  - windows10
  - python 3.12.0
- vast.ai #1 & #2
  - pytorch/pytorch_2.2.0-cuda12.1-cudnn8-devel

## Create environments
1. run ``pip install -r requirements.txt``
2. run ''pip install pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121``. (Install pytorch 2.3.1)

## Data Download / Prepare data
1. download LEAP data from kaggle and put folder on ``input/``
2. run ``python leap/src/download_data.py``
3. run ``leap/src/split.ipynb``

## FeatureEngineering / Training / Inference
1. run ``python leap/src/run.py``
2. run ``python leap/src/pp.ipynb`` for each submissions (please change the path manually).

## Stacking
1. Correct all submission into one folders (``.\input\submissions\{user_name}``) like below.
```
├─kami
│      kami_experiments_201_unet_multi_all_384_n2_submission.parquet
│      kami_experiments_201_unet_multi_all_384_n2_valid_pred.parquet
│      kami_experiments_201_unet_multi_all_512_n3_submission.parquet
│      kami_experiments_201_unet_multi_all_512_n3_valid_pred.parquet
|      ...
│      
├─kurupical
│  ├─20240703230157_exp042_70m_transformer_512x4_lr0.001_beta1
│  │      pred_valid.parquet
│  │      submission.parquet
│  │      
│  ├─20240705215850_exp042_70m_transformer_768x4_lr0.001_beta1
│  │      pred_valid.parquet
│  │      submission.parquet
│  │      
|  ...
│          
└─takoi
        ex123_pp.parquet
        ex124_pp.parquet
        ...
        exp123_val_preds.npy
        exp124_val_preds.npy
        ...

```

2. ``python src/run_emsemble.py``
3. ``python src/pp_ensemble.ipynb`` (please change the path manually)
  - 20240715_ensemble_per_target_500.parquet is one of final submission (To avoid irregular inference, if r2_score between nelder-mead and cnn is worse, replace nelder-mead inference)


# Other
- I prepare a set of the model weight and submission and training code here: https://www.kaggle.com/datasets/kurupical/leap-raw-model-and-code
