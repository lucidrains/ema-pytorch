## EMA - Pytorch

A simple way to keep track of an Exponential Moving Average (EMA) version of your pytorch model

## Install

```bash
$ pip install ema-pytorch
```

## Usage

```python
import torch
from ema_pytorch import EMA

# your neural network as a pytorch module

net = torch.nn.Linear(512, 512)

# wrap your neural network, specify the decay (beta)

ema = EMA(
    net,
    beta = 0.9999,              # exponential moving average factor
    update_after_step = 100,    # only after this number of .update() calls will it start updating
    update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
)

# mutate your network, with SGD or otherwise

with torch.no_grad():
    net.weight.copy_(torch.randn_like(net.weight))
    net.bias.copy_(torch.randn_like(net.bias))

# you will call the update function on your moving average wrapper

ema.update()

# then, later on, you can invoke the EMA model the same way as your network

data = torch.randn(1, 512)

output     = net(data)
ema_output = ema(data)

# if you want to save your ema model, it is recommended you save the entire wrapper
# as it contains the number of steps taken (there is a warmup logic in there, recommended by @crowsonkb, validated for a number of projects now)
# however, if you wish to access the copy of your model with EMA, then it will live at ema.ema_model
```

## Todo

- [ ] address the issue of annealing EMA to 1 near the end of training for BYOL https://github.com/lucidrains/byol-pytorch/issues/82

## Citations

```bibtex
@article{Karras2023AnalyzingAI,
    title   = {Analyzing and Improving the Training Dynamics of Diffusion Models},
    author  = {Tero Karras and Miika Aittala and Jaakko Lehtinen and Janne Hellsten and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2312.02696},
    url     = {https://api.semanticscholar.org/CorpusID:265659032}
}
```
