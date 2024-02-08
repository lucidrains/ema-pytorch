from copy import deepcopy
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Module

from beartype import beartype
from beartype.typing import Set, Optional

def exists(val):
    return val is not None

def get_module_device(m: Module):
    return next(m.parameters()).device

def inplace_copy(tgt: Tensor, src: Tensor, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.copy_(src)

def inplace_lerp(tgt: Tensor, src: Tensor, weight, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.lerp_(src, weight)

# algorithm 2 in https://arxiv.org/abs/2312.02696

def sigma_rel_to_gamma(sigma_rel):
    t = sigma_rel ** -2
    return Tensor([1, 7, 16 - t, 12 - t]).numpy().real.max()

class KarrasEMA(Module):
    """
    exponential moving average module that uses hyperparameters from the paper https://arxiv.org/abs/2312.02696
    can either use gamma or sigma_rel from paper
    """

    @beartype
    def __init__(
        self,
        model: Module,
        sigma_rel: Optional[float] = None,
        gamma: Optional[float] = None,
        ema_model: Optional[Module] = None,           # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        update_every: int = 100,
        frozen: bool = False,
        param_or_buffer_names_no_ema: Set[str] = set(),
        ignore_names: Set[str] = set(),
        ignore_startswith_names: Set[str] = set(),
        include_online_model = True,                  # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
        allow_different_devices = False               # if the EMA model is on a different device (say CPU), automatically move the tensor
    ):
        super().__init__()

        assert exists(sigma_rel) ^ exists(gamma), 'either sigma_rel or gamma is given. gamma is derived from sigma_rel as in the paper, then beta is dervied from gamma'

        if exists(sigma_rel):
            gamma = sigma_rel_to_gamma(sigma_rel)

        self.gamma = gamma
        self.frozen = frozen

        # whether to include the online model within the module tree, so that state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model] # hack

        # ema model

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = deepcopy(model)
            except Exception as e:
                print(f'Error: While trying to deepcopy model: {e}')
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        self.ema_model.requires_grad_(False)

        # parameter and buffer names

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if param.dtype in [torch.float, torch.float16]}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if buffer.dtype in [torch.float, torch.float16]}

        # tensor update functions

        self.inplace_copy = partial(inplace_copy, auto_move_device = allow_different_devices)
        self.inplace_lerp = partial(inplace_lerp, auto_move_device = allow_different_devices)

        # updating hyperparameters

        self.update_every = update_every

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        # whether to manage if EMA model is kept on a different device

        self.allow_different_devices = allow_different_devices

        # init and step states

        self.register_buffer('initted', torch.tensor(False))
        self.register_buffer('step', torch.tensor(0))

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]
    
    @property
    def beta(self):
        return (1 - 1 / (self.step + 1)) ** (1 + self.gamma)

    def eval(self):
        return self.ema_model.eval()
    
    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(ma_params.data, current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(ma_buffers.data, current_buffers.data)

    def copy_params_from_ema_to_model(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(current_buffers.data, ma_buffers.data)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))

        self.update_moving_average(self.ema_model, self.model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        if self.frozen:
            return

        copy, lerp = self.inplace_copy, self.inplace_lerp
        current_decay = self.beta

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy(ma_params.data, current_params.data)
                continue

            lerp(ma_params.data, current_params.data, 1. - current_decay)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy(ma_buffer.data, current_buffer.data)
                continue

            lerp(ma_buffer.data, current_buffer.data, 1. - current_decay)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
