<div align="center">
    <h3><a href="https://openreview.net/forum?id=pwKkNSuuEs">Abstracted Shapes as Tokens - A Generalizable and Interpretable Model for Time-series Classification</a></h3>
    <h4>NeurIPS 2024</h4>
    <h4>Yunshi Wen, Tengfei Ma, Tsui-Wei Weng, Lam M. Nguyen, Anak Agung Julius</h4>
</div>

VQShape is a pre-trained model for time-series analysis. The model provides shape-level features as interpretable representations for time-series data. The model also contains a universal codebook of shapes that generalizes to different domains and datasets. The current checkpoints are pre-trained on the UEA multivariate time-series classification datasets [[Bagnall et al., 2018]](https://timeseriesclassification.com/). 


## Usage

### Environment Setup:

Python version: Python 3.11.

We use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to manage training, checkpointing, and potential future distributed training.

Install the dependencies with conda:
```bash
conda env create -f environment.yml
```

Install the dependencies with pip:
```bash
conda create -n vqshape python=3.11
conda activate vqshape
pip install -r requirements.txt
```

### Data Preparation
Because of the memory configuration/limitation of our computation resources, we currently implement a lazy-loading mechanism for pre-training. To prepare the pre-training data, we first read the UEA multivariate time series and save each univariate time series into a csv file. Refer to [this notebook](notebooks/data_preparation.ipynb) for more details. The [dataset](data_provider/timeseries_loader.py) can be replaced to fit specific computational resources.

### Pre-training
Specify the codebook size and the embedding dimension. For example, for a codebook size of 64 and an embedding dimension of 512, run:
```
bash ./scripts/pretrain.sh 64 512
```
Other hyperparameters and configurations can be specified in the [bash script](scripts/pretrain.sh).

### Load the pre-trained checkpoint

The pre-trained checkpoint can be loaded efficiently using the PyTorch Lightning module. Here is an example of loading the checkpoint to a CUDA device:
```python 
from vqshape.pretrain import LitVQShape

checkpoint_path = "checkpoints/uea_dim512_codebook64/VQShape.ckpt"
lit_model = LitVQShape.load_from_checkpoint(checkpoint_path, 'cuda')
model = lit_model.model
```

### Use the pre-trained model

#### 1. Tokenization (extract shapes from time-series)

```python
import torch
import torch.nn.functional as F
from einops import rearrange
from vqshape.pretrain import LitVQShape

# load the pre-trained model
checkpoint_path = "checkpoints/uea_dim512_codebook64/VQShape.ckpt"
lit_model = LitVQShape.load_from_checkpoint(checkpoint_path, 'cuda')
model = lit_model.model

x = torch.randn(16, 5, 1000)  # 16 multivariate time-series, each with 5 channels and 1000 timesteps
x = F.interpolate(x, 512, mode='linear')  # first interpolate to 512 timesteps
x = rearrange(x, 'b c t -> (b c) t')  # transform to univariate time-series

representations, output_dict = model(x, mode='tokenize') # tokenize with VQShape

token_representations = representations['token']
histogram_representations = representations['histogram']
```

#### 2. Classification
Specify the codebook size and the embedding dimension. For benchmarking of the UEA datasets, run:
```
bash ./scripts/mtsc.sh 64 512
```

#### 3. Visualize the Codebook
See [this notebook](notebooks/visualization.ipynb) for examples of visualizing the codebook and code distribution.

## Pre-trained Checkpoints
We provide the pre-trained checkpoints on the UEA time-series classification datasets, which produce the results in the [paper](https://openreview.net/forum?id=pwKkNSuuEs).

The checkpoints are available at the [release page](https://github.com/YunshiWen/VQShape/releases/tag/v0.1.0-cls). To use the checkpoints:
1. Download the `.zip` file.
2. Unzip the file into `./checkpoints`.

Information of the pre-trained checkpoints:
| Embedding Dimension | Codebook Size | Num. Parameters | Checkpoint | Mean Accuracy (Token) | Mean Accuracy (Hist.) |
|:------------------:|:--------------:|:----------------:|:------------:|:---------------:|:----------------:|
| 256 | 512 | 9.5M | [download](https://github.com/YunshiWen/VQShape/releases/download/v0.1.0-cls/uea_dim256_codebook512.zip) | 0.731 | 0.711 |
| 512 | 512 | 37.1M | [download](https://github.com/YunshiWen/VQShape/releases/download/v0.1.0-cls/uea_dim512_codebook512.zip) | 0.723 | 0.709 |
| 512 | 128 | 37.1M | [download](https://github.com/YunshiWen/VQShape/releases/download/v0.1.0-cls/uea_dim512_codebook128.zip) | 0.720 | 0.711 |
| 512 | 64 | 37.1M | [download](https://github.com/YunshiWen/VQShape/releases/download/v0.1.0-cls/uea_dim512_codebook64.zip) | 0.719 | 0.712 |
| 512 | 32 | 37.1M | [download](https://github.com/YunshiWen/VQShape/releases/download/v0.1.0-cls/uea_dim512_codebook32.zip) | 0.706 | 0.696 |

> [!NOTE]
> The checkpoint with embedding dimension 256 and codebook size 512 is **not** included in the paper, but produces the best performance. However, the checkpoint with embedding dimension 256 and codebook size 64 results in worse performance. We believe the scaling between embedding dimension and codebook size need further investigation.

## Citation
```
@inproceedings{
    wen2024abstracted,
    title={Abstracted Shapes as Tokens - A Generalizable and Interpretable Model for Time-series Classification},
    author={Yunshi Wen and Tengfei Ma and Tsui-Wei Weng and Lam M. Nguyen and Anak Agung Julius},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024}
}
```

## Acknowledgement

We thank the research community for the great work on time-series analysis, the open-source codebase, and the datasets, including but not limited to:
- A part of the code is adapted from [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
- The UEA and UCR teams for collecting and sharing the [time-series classification datasets](https://timeseriesclassification.com/).
