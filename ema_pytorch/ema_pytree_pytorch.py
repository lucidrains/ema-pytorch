from __future__ import annotations
from typing import Callable, Any

from copy import deepcopy
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.utils import _pytree as pytree_pkg

def exists(val):
    return val is not None

def divisible_by(num, den):
    return (num % den) == 0

def maybe_coerce_dtype(t, dtype):
    if t.dtype == dtype:
        return t

    return t.to(dtype)

def inplace_copy(tgt: Tensor, src: Tensor, *, coerce_dtype = False):
    if coerce_dtype:
        src = maybe_coerce_dtype(src, tgt.dtype)

    tgt.copy_(src)

def inplace_lerp(tgt: Tensor, src: Tensor, weight, *, coerce_dtype = False):
    if coerce_dtype:
        src = maybe_coerce_dtype(src, tgt.dtype)

    tgt.lerp_(src, weight)

class EMAPytree(Module):
    def __init__(
        self,
        pytree: Any,
        ema_pytree: Any | Callable[[], Any] | None = None,
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 2 / 3,
        min_value = 0.0,
        coerce_dtype = False,
    ):
        super().__init__()
        self.beta = beta
        self.is_frozen = beta == 1.

        self.online_pytree = [pytree] # hack to avoid being registered as submodule if it happens to be one

        # handle callable returning ema module

        if not exists(ema_pytree) and callable(ema_pytree):
            ema_pytree = ema_pytree()

        self.ema_pytree = ema_pytree

        if not exists(self.ema_pytree):
            self.ema_pytree = deepcopy(pytree)

        # detach all tensors in ema model

        ema_tensors, _ = pytree_pkg.tree_flatten(self.ema_pytree)
        for p in ema_tensors:
            if isinstance(p, Tensor):
                p.detach_()

        # tensor update functions

        self.inplace_copy = partial(inplace_copy, coerce_dtype = coerce_dtype)
        self.inplace_lerp = partial(inplace_lerp, coerce_dtype = coerce_dtype)

        # updating hyperparameters

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        self.coerce_dtype = coerce_dtype

        # init and step states

        self.register_buffer('initted', torch.tensor(False))
        self.register_buffer('step', torch.tensor(0))

    @property
    def pytree(self):
        return self.online_pytree[0]

    def copy_params_from_pytree_to_ema(self):
        copy = self.inplace_copy

        ema_tensors, _ = pytree_pkg.tree_flatten(self.ema_pytree)
        online_tensors, _ = pytree_pkg.tree_flatten(self.pytree)

        for ma_tensor, online_tensor in zip(ema_tensors, online_tensors):
            if isinstance(ma_tensor, Tensor) and isinstance(online_tensor, Tensor):
                copy(ma_tensor.data, online_tensor.data)

    def get_current_decay(self):
        epoch = (self.step - self.update_after_step - 1).clamp(min = 0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch.item() <= 0:
            return 0.

        return value.clamp(min = self.min_value, max = self.beta).item()

    def update(self):
        step = self.step.item()
        self.step += 1

        if not self.initted.item():
            self.copy_params_from_pytree_to_ema()
            self.initted.data.copy_(torch.tensor(True))
            return

        should_update = divisible_by(step, self.update_every)

        if should_update and step <= self.update_after_step:
            self.copy_params_from_pytree_to_ema()
            return

        if should_update:
            self.update_moving_average(self.ema_pytree, self.pytree)

    @torch.no_grad()
    def update_moving_average(self, ma_pytree, current_pytree, current_decay = None):
        if self.is_frozen:
            return

        if not exists(current_decay):
            current_decay = self.get_current_decay()

        ema_tensors, _ = pytree_pkg.tree_flatten(ma_pytree)
        online_tensors, _ = pytree_pkg.tree_flatten(current_pytree)

        for ma_tensor, online_tensor in zip(ema_tensors, online_tensors):
            if isinstance(ma_tensor, Tensor) and isinstance(online_tensor, Tensor):
                self.inplace_lerp(ma_tensor.data, online_tensor.data, 1. - current_decay)

    def __call__(self, *args, **kwargs):
        if callable(self.ema_pytree):
            return self.ema_pytree(*args, **kwargs)
        return self.ema_pytree

if __name__ == '__main__':
    online_tree = {
        'w': torch.randn(2, 2),
        'b': torch.randn(2)
    }

    ema_tree = EMAPytree(
        online_tree,
        beta = 0.5,
        min_value = 0.5,
        update_after_step = 0,
        update_every = 1
    )

    ema_tree.update() # init copy

    with torch.no_grad():
        online_tree['w'].fill_(1.0)
        online_tree['b'].fill_(1.0)

    old_ema_w = ema_tree.ema_pytree['w'].clone()
    ema_tree.update() # lerp

    expected_w = 0.5 * old_ema_w + 0.5 * online_tree['w']
    assert torch.allclose(ema_tree.ema_pytree['w'], expected_w)
    
    print('success')
