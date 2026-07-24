from __future__ import annotations
import inspect
from typing import Callable, Any

import torch
from torch import nn
from torch.nn import Module

from ema_pytorch.ema_pytorch import EMA, exists

# helpers

def identity(ema_args, ema_kwargs):
    return ema_args, ema_kwargs

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

def normalize_ema_args_kwargs(
    ema_args: Any | None,
    ema_kwargs: dict | None
) -> tuple[tuple | None, dict | None]:
    if not exists(ema_args):
        return None, ema_kwargs

    if isinstance(ema_args, dict):
        ema_kwargs = {**ema_args, **default(ema_kwargs, {})}
        return (), ema_kwargs

    if isinstance(ema_args, tuple):
        return ema_args, ema_kwargs

    return (ema_args,), ema_kwargs

def get_submodule(model: Module, path: str) -> Module:
    if not path or path == '.':
        return model

    if hasattr(model, 'get_submodule'):
        return model.get_submodule(path)

    curr_module = model
    for part in path.split('.'):
        if part.isdigit() and isinstance(curr_module, (nn.Sequential, nn.ModuleList)):
            curr_module = curr_module[int(part)]
            continue

        curr_module = getattr(curr_module, part)

    return curr_module

# parse module kwarg specification configs into structured objects

def parse_ema_module_kwargs(
    model: Module,
    ema_module_kwargs: dict | list | tuple | set | None,
    default_kwarg: str = 'ema_output'
) -> list[EMAModuleKwargSpec]:
    if not exists(ema_module_kwargs):
        return []

    # if simple list of module paths, receiver and target EMA paths are identical

    if isinstance(ema_module_kwargs, (list, tuple, set)):
        return [EMAModuleKwargSpec(receiver_path = path, ema_path = path, kwarg_name = default_kwarg) for path in ema_module_kwargs]

    assert isinstance(ema_module_kwargs, dict)
    module_kwarg_specs = []

    # parse dict mappings (supporting strings, tuples, or config dicts)

    for receiver_path, path_or_config in ema_module_kwargs.items():

        # string value: either target module path or custom kwarg name

        if isinstance(path_or_config, str):
            try:
                get_submodule(model, path_or_config)
                is_submodule = True
            except Exception:
                is_submodule = False

            ema_path = path_or_config if is_submodule else receiver_path
            kwarg_name = default_kwarg if is_submodule else path_or_config

            module_kwarg_specs.append(EMAModuleKwargSpec(receiver_path = receiver_path, ema_path = ema_path, kwarg_name = kwarg_name))
            continue

        # tuple / list specification: (ema_path, kwarg_name, optional_transform)

        if isinstance(path_or_config, (tuple, list)):
            ema_path, kwarg_name, transform, *_ = (*path_or_config, None, None, None)
            ema_path = default(ema_path, receiver_path)
            kwarg_name = default(kwarg_name, default_kwarg)

            module_kwarg_specs.append(EMAModuleKwargSpec(receiver_path = receiver_path, ema_path = ema_path, kwarg_name = kwarg_name, transform = transform))
            continue

        # dict specification: {'ema_module_path': ..., 'ema_kwarg': ..., 'transform': ...}

        if isinstance(path_or_config, dict):
            ema_path = path_or_config.get('ema_module_path', path_or_config.get('ema_path', receiver_path))
            kwarg_name = path_or_config.get('ema_kwarg', path_or_config.get('kwarg_name', default_kwarg))
            transform = path_or_config.get('transform', None)

            module_kwarg_specs.append(EMAModuleKwargSpec(receiver_path = receiver_path, ema_path = ema_path, kwarg_name = kwarg_name, transform = transform))
            continue

    return module_kwarg_specs

# spec dataclass

class EMAModuleKwargSpec:
    def __init__(
        self,
        receiver_path: str,
        ema_path: str,
        kwarg_name: str = 'ema_output',
        transform: Callable | None = None
    ):
        self.receiver_path = receiver_path
        self.ema_path = ema_path
        self.kwarg_name = kwarg_name
        self.transform = transform

# main EMA module wrapper class

class EMAModuleWrapper(Module):
    def __init__(
        self,
        model: Module,
        ema: EMA | None = None,
        ema_module_kwargs: dict | list | tuple | set | None = None,
        default_ema_kwarg: str = 'ema_output',
        forward_online_during_train: bool = True,
        **ema_kwargs
    ):
        super().__init__()

        self.ema = default(ema, lambda: EMA(model, **ema_kwargs))

        self.ema_module_kwargs = ema_module_kwargs
        self.default_ema_kwarg = default_ema_kwarg
        self.forward_online_during_train = forward_online_during_train

        self._hook_handles = []
        self._captured_ema_outputs = {}

        self.register_ema_module_hooks()

    @property
    def model(self):
        return self.ema.model

    @property
    def ema_model(self):
        return self.ema.ema_model

    # delegate EMA update calls to underlying EMA instance

    def update(self):
        return self.ema.update()

    def update_model_with_ema(self, decay = None):
        return self.ema.update_model_with_ema(decay = decay)

    def eval(self):
        return self.ema.eval()

    def forward_eval(self, *args, **kwargs):
        return self.ema.forward_eval(*args, **kwargs)

    # hook management

    def remove_ema_module_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def register_ema_module_hooks(self):
        self.remove_ema_module_hooks()

        if not exists(self.ema_module_kwargs) or not exists(self.ema_model):
            return

        specs = parse_ema_module_kwargs(self.model, self.ema_module_kwargs, self.default_ema_kwarg)

        for spec in specs:
            ema_submodule = get_submodule(self.ema_model, spec.ema_path)
            online_submodule = get_submodule(self.model, spec.receiver_path)

            # register forward hook on target EMA submodule to capture output

            def create_ema_forward_hook(path, transform):
                def hook(module, args, output):
                    out = output
                    if exists(transform):
                        out = transform(out)
                    self._captured_ema_outputs[path] = out
                return hook

            ema_hook_handle = ema_submodule.register_forward_hook(create_ema_forward_hook(spec.ema_path, spec.transform))
            self._hook_handles.append(ema_hook_handle)

            # assert receiver submodule forward method accepts the kwarg parameter

            sig = inspect.signature(online_submodule.forward)
            accepts_kwarg = spec.kwarg_name in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

            assert accepts_kwarg, f"submodule '{spec.receiver_path}' forward method must accept keyword argument '{spec.kwarg_name}'"

            # register forward pre-hook on online submodule to inject harvested EMA output into kwargs

            def create_online_forward_pre_hook(ema_path, kwarg_name):
                def pre_hook(module, args, kwargs):
                    if ema_path in self._captured_ema_outputs:
                        kwargs[kwarg_name] = self._captured_ema_outputs[ema_path]
                    return args, kwargs
                return pre_hook

            online_pre_hook_handle = online_submodule.register_forward_pre_hook(create_online_forward_pre_hook(spec.ema_path, spec.kwarg_name), with_kwargs=True)
            self._hook_handles.append(online_pre_hook_handle)

    # forward passes

    def forward_online(
        self,
        *args,
        only_online: bool = False,
        only_ema: bool = False,
        ema_args: Any = None,
        ema_kwargs: dict | None = None,
        auto_normalize_ema_args: bool = True,
        **kwargs
    ):
        assert not (only_online and only_ema), 'cannot set both only_online and only_ema to True'

        if only_online or only_ema:
            assert not exists(ema_args) and not exists(ema_kwargs), 'cannot pass ema_args or ema_kwargs when only_online or only_ema is True'

        if only_ema:
            return self.ema_model(*args, **kwargs)

        if only_online:
            self._captured_ema_outputs.clear()
            return self.model(*args, **kwargs)

        assert exists(self.ema_model), "EMA model is not initialized yet"
        self._captured_ema_outputs.clear()

        # harvest target representations by running EMA model under no_grad

        with torch.no_grad():
            training_state = self.ema_model.training
            self.ema_model.eval()

            normalize_fn = normalize_ema_args_kwargs if auto_normalize_ema_args else identity

            ema_args, ema_kwargs = normalize_fn(ema_args, ema_kwargs)

            ema_forward_args = default(ema_args, args)
            ema_forward_kwargs = default(ema_kwargs, kwargs)

            self.ema_model(*ema_forward_args, **ema_forward_kwargs)
            self.ema_model.train(training_state)

        # forward online model (pre-hooks automatically inject captured EMA outputs)

        out = self.model(*args, **kwargs)

        self._captured_ema_outputs.clear()
        return out

    def forward(
        self,
        *args,
        forward_online: bool | None = None,
        only_online: bool = False,
        only_ema: bool = False,
        ema_args: Any = None,
        ema_kwargs: dict | None = None,
        auto_normalize_ema_args: bool = True,
        **kwargs
    ):
        if (
            only_online or
            only_ema or
            default(forward_online, exists(self.ema_module_kwargs) and self.model.training and self.forward_online_during_train)
        ):
            return self.forward_online(
                *args,
                only_online = only_online,
                only_ema = only_ema,
                ema_args = ema_args,
                ema_kwargs = ema_kwargs,
                auto_normalize_ema_args = auto_normalize_ema_args,
                **kwargs
            )

        return self.ema_model(*args, **kwargs)
