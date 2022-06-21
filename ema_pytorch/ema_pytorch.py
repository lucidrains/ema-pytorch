import copy
import torch
from torch import nn

def exists(val):
    return val is not None

def clamp(value, min_value = None, max_value = None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value

class EMA(nn.Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    def __init__(
        self,
        model,
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 2 / 3,
        min_value = 0.0,
        param_or_buffer_names_no_ema = set(),
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model

        try:
            self.ema_model = copy.deepcopy(model)
        except:
            print('Your model was not copyable. Please make sure you are not using any LazyLinear')
            exit()

        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def copy_params_from_model_to_ema(self):
        for ma_param, current_param in zip(list(self.ema_model.parameters()), list(self.online_model.parameters())):
            ma_param.data.copy_(current_param.data)

        for ma_buffer, current_buffer in zip(list(self.ema_model.buffers()), list(self.online_model.buffers())):
            ma_buffer.data.copy_(current_buffer.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value = 0)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch <= 0:
            return 0.

        return clamp(value, min_value = self.min_value, max_value = self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(list(current_model.named_parameters()), list(ma_model.named_parameters())):
            if name in self.param_or_buffer_names_no_ema:
                ma_param.data.copy_(current_param.data)
                continue

            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for (name, current_buffer), (_, ma_buffer) in zip(list(current_model.named_buffers()), list(ma_model.named_buffers())):
            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
