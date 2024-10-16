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

In order to use the post-hoc synthesized EMA, proposed by Karras et al. in <a href="https://arxiv.org/abs/2312.02696">a recent paper</a>, follow the example below

```python
import torch
from ema_pytorch import PostHocEMA

# your neural network as a pytorch module

net = torch.nn.Linear(512, 512)

# wrap your neural network, specify the sigma_rels or gammas

emas = PostHocEMA(
    net,
    sigma_rels = (0.05, 0.28),           # a tuple with the hyperparameter for the multiple EMAs. you need at least 2 here to synthesize a new one
    update_every = 10,                  # how often to actually update, to save on compute (updates every 10th .update() call)
    checkpoint_every_num_steps = 10,
    checkpoint_folder = './post-hoc-ema-checkpoints'  # the folder of saved checkpoints for each sigma_rel (gamma) across timesteps with the hparam above, used to synthesizing a new EMA model after training
)

net.train()

for _ in range(1000):
    # mutate your network, with SGD or otherwise

    with torch.no_grad():
        net.weight.copy_(torch.randn_like(net.weight))
        net.bias.copy_(torch.randn_like(net.bias))

    # you will call the update function on your moving average wrapper

    emas.update()

# now that you have a few checkpoints
# you can synthesize an EMA model with a different sigma_rel (say 0.15)

synthesized_ema = emas.synthesize_ema_model(sigma_rel = 0.15)

# output with synthesized EMA

data = torch.randn(1, 512)

synthesized_ema_output = synthesized_ema(data)

```

For testing out the claims of a free lunch from the <a href="https://arxiv.org/abs/2402.09240">`Switch EMA`</a> paper, just set `update_model_with_ema_every` as so

```python

ema = EMA(
    net,
    ...,
    update_model_with_ema_every = 10000 # say 10k steps is 1 epoch
)

# or you can do it manually at the end of each epoch

ema.update_model_with_ema()

```

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

```bibtex
@article{Lee2024SlowAS,
    title   = {Slow and Steady Wins the Race: Maintaining Plasticity with Hare and Tortoise Networks},
    author  = {Hojoon Lee and Hyeonseo Cho and Hyunseung Kim and Donghu Kim and Dugki Min and Jaegul Choo and Clare Lyle},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.02596},
    url     = {https://api.semanticscholar.org/CorpusID:270258586}
}
```

```bibtex
@article{Li2024SwitchEA,
    title   = {Switch EMA: A Free Lunch for Better Flatness and Sharpness},
    author  = {Siyuan Li and Zicheng Liu and Juanxi Tian and Ge Wang and Zedong Wang and Weiyang Jin and Di Wu and Cheng Tan and Tao Lin and Yang Liu and Baigui Sun and Stan Z. Li},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.09240},
    url     = {https://api.semanticscholar.org/CorpusID:267657558}
}
```
