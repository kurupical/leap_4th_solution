# 1. Kurupical part

## 1-1. Summary

| # | model_name | cv | public | private | training time | note |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1 | convnext 64x3-128x3-256x27-512x3 | 0.7881 | 0.78577 | 0.78147 |60h with 1xRTX4090 | |
| 2 | convnext 96x3-192x3-384x27-768x3 | 0.7887 | 0.78697 | 0.78281 | 120h with 1xRTX4090 | |
| 3 | convnext 128x3-256x3-512x27-1024x3 | 0.78693 | 0.78480 | 0.78150 | 132h with 1xRTX4090 | use 81.25% data |
| 4 | convnext 144x3-288x3-576x27-1152x3 | 0.78547 | 0.78317 | 0.77903 | 28h with 8xRTX4090 |  |
| 5 | transformer 512x4 | 0.7848 | 0.78341 | 0.77910 | 54h with 1xRTX4090 |  |
| 6 | transformer 768x4 | 0.7843 | 0.78350 | 0.77966 | 72h with 1xRTX4090 |  |
| 7 | convnext 144x3-288x3-576x27-1152x3 | 0.78769 | 0.78622 | 0.78137 | 16h with 8xRTX4090 | training 4epochs |

## 1-2. Feature Engineering / Preprocessing
I use all row-les datasets.

### 1-2-1. Feature Engineering

- diff
  - x[i] - x[i-1]
  - x[i] - x[i-2]
- mean and diff_mean in similar features
  - q0002, q0003
  - state_u, state_v
  - pbuf_*

### 1-2-2. Preprocessing
- Standard Scaler for feature / label
  - Calculate feature mean/std with both train/test datasets.
- Extremely large values can make model training unstable, so features are standardized and then clipped to the range of -100 to 100.
- To convert to the shape of (batch_size, n_feature, 60), scalar values (e.g., state_ps) are transformed into time series data by repeating the same value 60 times.

## 1-3. Training Methods
### 1-3-1. Models
#### 1-3-1-1. ConvNeXt

A ConvNext with inputs of shape (batch_size, n_features, 60) and outputs of shape (batch_size, 368). The pseudocode is shown below.

```python
class Head1D(nn.Module):
    def __init__(self):
        super(Head1D, self).__init__()
        self.final_layer = nn.LazyConv1d(out_channels=14, kernel_size=1, stride=1, padding="same")
        self.fc = nn.Linear(8 * 60, 8)

    def forward(self, x):
        x = self.final_layer(x)  # (hidden_size, 60) -> (14, 60)
        x_out = torch.cat(
            [
                x[:, :6, :].reshape(
                    -1, 6 * 60
                ),  # shape = (bs, 360) ptend_t, ptend_q0001, ptend_q0002, ptend_q0003, ptend_u, ptend_v
                self.fc(x[:, 6:, :].reshape(-1, 8 * 60)),  # shape = (bs, 8)
            ],
            dim=1,
        )  # shape = (bs, 360 + 8)
        return x_out

class ConvNeXt(nn.Module):
    def __init__(self):
        self.head = Head1D()
        ...

    def forward_feature(x):
        # convnext 
        ...
        
    def forward(self, x):
        x = x["feature"]  # shape = (bs, 384, n_features)
           
        # convnext part
        x_features = self.forward_feature(x)  # shape = (bs, n_features, 60) -> (bs, hidden_dims, 60)
        
        x_pred = self.head(x_features)  # shape = (bs, hidden_dims, 60) -> (bs, 368)
        return x_pred
    
model = ConvNeXt()
x = torch.randn(4, 25, 60)
assert model(x).shape == (4, 368)

```

- Based on https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py  ,I've rewritten it for 1D inputs with the following considerations:
  - Adjusted to ensure the shape remains unchanged between input and output by setting stride=1 and padding="same".
  - Replaced LayerNorm with BatchNorm since LayerNorm was not effective.

#### 1-3-1-2. Transformer

A Transformer with input of shape  ``(batch_size, 60, n_features)`` and outputs of shape ``(batch_size, 368)``.  The pseudocode is shown below.

```python
class TransformerModel(nn.Module):
    def __init__(
        self,
        hidden_dims: int,
        n_layers: int,
        n_heads: int,
        head_mode: str,
        dropout: int = 0,
    ):
        super(TransformerModel, self).__init__()
        
        N_SEQUENTIAL_COLUMNS = 6
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
        
        # transformer
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dims,
            nhead=n_heads,
            dim_feedforward=hidden_dims * 4,
            dropout=0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)        
        
        # head
        ## 1. cnn (for sequential (60x6))
        self.fc_head_list_sequential = []
        head_cnn_params = [{"out_channels": 256, "kernel_size": 3}] * 3
        for _ in range(N_SEQUENTIAL_COLUMNS):
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
            fc_head.append(
                nn.LazyConv1d(1, kernel_size=3, stride=1, padding="same")
            )
            fc_head = nn.Sequential(*fc_head)
            self.fc_head_list_sequential.append(fc_head)
        self.fc_head_list_sequential = nn.ModuleList(self.fc_head_list_sequential)
        
        ## 2. linear (for scalar (8))
        self.fc_head_scalar = nn.Linear(hidden_dims, 8)
        
    def forward(self, x):

        # stem
        x = self.fc_stem(x)  # (bs, 60, n_features) -> (bs, 60, hidden_dims)
        pe = self.position_encoder(
            torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        )  # (60) -> (bs, 60, hidden_dims)
        x = x + pe
        
        # transformer
        x = self.transformer(x)  # (bs, 60, hidden_dims) -> (bs, 60, hidden_dims)
        
        # head
        ## 1. cnn
        x_out = []
        for i in range(6):
            x_out_ = self.fc_head_list_sequential[i](
                x.permute(0, 2, 1)
            )  # (bs, hidden_dims, 60) -> (bs, 1, 60)
            x_out_ = x_out_.squeeze(1)  # (bs, 1, 60) -> (bs, 60)
            x_out.append(x_out_)        
        ## 2. linear
        x_out_ = self.fc_head_scalar(x.mean(dim=1))  # (bs, hidden_dims) -> (bs, 8)
        x_out.append(x_out_)
        x_out = torch.cat(x_out, dim=1)  # (bs, 60*6+8)
        return x_out        
```


### 1-3-2. Hyper Parameter
| param_name | #1 | #2 | #3 | #4 | #5 | #6 | #7 | 
| --- | --- | --- | --- | --- | --- | --- | --- |
| epochs | 7 | 7 | 7 | 7 | 7 | 7 | 4 |
| lr | 2.5e-3 | 2e-3 | 2e-3 | 1.5e-3 | 1e-3 | 1e-3 | 2e-3 |
| batch_size | 384 | 384 | 384 | 288 | 384 | 384 | 288 |
| weight_decay | 0.05 | 0.05 | 0.05 | 0.05 | 0.01 | 0.01 | 0.075 |

- other
    - optimizer: AdamW
        - cnn: AdamW(weight_decay=0.05 or 0.075)
        - transformer: AdamW(weight_decay=0.01)
    - loss: 
        - cnn: SmoothL1Loss(beta=0.01)
        - transformer: SmoothL1Loss(beta=1)
    - scheduler
        - cnn: transformers.get_polynomial_decay_schedule_with_warmup(alpha=2, warmup_ratio=0.1)
        - transformer: transformers.get_polynomial_decay_schedule_with_warmup(alpha=1, warmup_ratio=0.1), equals to get_linear_scheduler_with_warmup
    - The batch size of 384 is a remnant of using lat/lon leak.
    - Transformers tend to diverge quickly with a high learning rate, so keep it as low as possible. CNNs were not as sensitive.
    - Regarding the beta of SmoothL1Loss, smaller models tended to perform better with something close to Huber Loss, while larger models showed better performance with something closer to L1 Loss.

## 1-4. Interesting Findings

This was the competition where we needed to rely on neural networks the most among all the competitions we've participated in so far.

- Common
    - Proper learning rate scheduling was crucial, and we spent a significant amount of time tuning it.
    - We tested with a small amount of data (n=1.8m, n=9m) before using the full dataset (n=70m). There were cases where what worked with a small amount of data didn't work with the full dataset.
    - The same issue occurred between small models (ConvNeXt 32x3-64x3-128x27-256x3) and large models (ConvNeXt 96x3-192x3-384x27-768x3).
- ConvNeXt
    - Polynomial decay scheduler and high weight decay were effective.
    - Observing the train loss on W&B, it seemed that training progressed significantly when the learning rate was low. Therefore, we adopted a polynomial decay scheduler to stay at a low learning rate for a longer period. We found that training progressed too much and caused overfitting, so we controlled it with weight decay. (cv: +0.004)
    - It seemed that the larger the model, the higher the accuracy, given proper hyperparameter settings.
        - Larger models converged faster, so we reduced the number of epochs.
        - We didn't have time to test this thoroughly towards the end.
- Transformer
    - We tried various architectures, and attaching a CNN-head or adding positional encoding proved effective.
    - We tested large models of 512x4 and above with n=9m, but found no significant improvement over 512x4 or even a decrease in accuracy. It is unclear whether this was due to poor hyperparameter tuning or if 512x4 was simply sufficient for this dataset.


# 4. Ensemble/Stacking
## 4-1. Summary
Ensemble 30 models.

| # | method | public | private |
| --- | --- | --- | --- |
| 1 | nelder-mead | 0.79100 | 0.78713 |
| 2 | 1d-cnn stacking | 0.79193 | 0.78774 |

## 4-2. Nelder-mead
(takoi's part)

## 4-3. 1D-CNN Stacking

Create simple 1d-cnn model with inputs of  ``(batch_size, n_models, n_labels(=368))`` and outputs of ``(batch_size, n_labels)`` 

```python

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
            x
        )  # (batch_size, n_models, seq_len=368) -> (batch_size, hidden_size, seq_len=368)
        x = x.mean(
            dim=1
        )  # (batch_size, hidden_size, seq_len=368) -> (batch_size, seq_len=368)
        return x
```

Hyperparameter is below: 
- lr: 1e-3
- batch_size: 256
- hidden_size: ``[ ** TODO **]``
- kernel_size: ``[3, 3, 3]``
- epochs: 20
- optimizer: ``AdamW(weight_decay=0)``
- scheduler: ``linear scheduler with warmup``
- loss: ``SmoothL1Loss(beta=1)``

