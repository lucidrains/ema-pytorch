from __future__ import annotations

import pytest

import torch
from torch import nn, Tensor, tensor, is_tensor
import torch.nn.functional as F

from ema_pytorch import EMAModuleWrapper

def exists(val):
    return val is not None

def test_readme_module_target_injection():

    class Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Linear(dim, dim)
            self.proj = nn.Linear(dim, dim)
            self.register_buffer('zero', tensor(0.), persistent = False)

        def forward(self, x, ema_output = None):
            h = F.relu(self.net(x))

            if not exists(ema_output):
                return h, self.zero

            if isinstance(ema_output, tuple):
                ema_output, _ = ema_output

            pred = self.proj(h)
            loss = 1. - F.cosine_similarity(pred, ema_output, dim = -1).mean()

            return h, loss

    class NestedBranch(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.block1 = Block(dim)
            self.block2 = Block(dim)

        def forward(self, x):
            x, loss1 = self.block1(x)
            x, loss2 = self.block2(x)
            return x, loss1 + loss2

    class DoubleNestedModel(nn.Module):
        def __init__(self, dim = 512):
            super().__init__()
            self.branch_a = NestedBranch(dim)
            self.branch_b = NestedBranch(dim)

        def forward(self, x):
            h_a, loss_a = self.branch_a(x)
            h_b, loss_b = self.branch_b(h_a)
            return h_b, loss_a + loss_b

    model = DoubleNestedModel(512)
    model.train()

    ema = EMAModuleWrapper(
        model,
        beta = 0.99,
        update_after_step = 0,
        update_every = 1,
        ema_module_kwargs = {
            'branch_a.block1': 'branch_b.block1',
            'branch_a.block2': 'branch_b.block2'
        }
    )
    ema.update()

    x = torch.randn(2, 512)

    out, loss = ema(x)

    assert is_tensor(loss)
    loss.backward()
    ema.update()

def test_ema_module_wrapper_scenarios():

    class SubBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x, ema_output = None, custom_kwarg = None):
            h = self.linear(x)
            return h, ema_output, custom_kwarg

    class NestedTreeModel(nn.Module):
        def __init__(self, dim = 16):
            super().__init__()
            self.branch_a = nn.ModuleDict({'block': SubBlock(dim)})
            self.branch_b = nn.ModuleDict({'block': SubBlock(dim)})

        def forward(self, x):
            out_a, ema_a, custom_a = self.branch_a['block'](x)
            out_b, ema_b, custom_b = self.branch_b['block'](out_a)
            return out_b, ema_a, custom_b

    model = NestedTreeModel(16)
    model.train()

    ema = EMAModuleWrapper(
        model,
        beta = 0.9,
        update_after_step = 0,
        update_every = 1,
        ema_module_kwargs = {
            'branch_a.block': 'branch_b.block',
            'branch_b.block': ('branch_a.block', 'custom_kwarg')
        }
    )
    ema.update()

    student_input = torch.randn(2, 16)
    teacher_input = torch.randn(2, 16)

    out_b, ema_a, custom_b = ema.forward_online(student_input, ema_args = (teacher_input,))

    assert exists(ema_a)
    assert exists(custom_b)

def test_multi_view_ssl():

    class Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Linear(dim, dim)
            self.proj = nn.Linear(dim, dim)
            self.register_buffer('zero', tensor(0.), persistent = False)

        def forward(self, x, ema_output = None):
            h = F.relu(self.net(x))

            loss = self.zero

            if exists(ema_output):
                if isinstance(ema_output, tuple):
                    ema_output, _ = ema_output

                pred = self.proj(h)
                loss = 1. - F.cosine_similarity(pred, ema_output, dim = -1).mean()

            return h, loss

    class MultiViewModule(nn.Module):
        def __init__(self, dim = 8):
            super().__init__()
            self.block_a = Block(dim)
            self.block_b = Block(dim)

        def forward(self, x):
            h_a, loss_a = self.block_a(x)
            h_b, loss_b = self.block_b(h_a)
            return h_b, loss_a + loss_b

    model = MultiViewModule(8)
    model.train()

    ema = EMAModuleWrapper(
        model,
        beta = 0.9,
        update_after_step = 0,
        update_every = 1,
        ema_module_kwargs = {
            'block_a': 'block_b'
        }
    )

    ema.update()

    student_input = torch.randn(2, 8)
    teacher_input = torch.randn(2, 8)

    out, loss = ema(student_input, ema_args = (teacher_input,))

    assert is_tensor(loss)
    loss.backward()
    ema.update()

def test_self_flow_example():

    class Layer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Linear(dim, dim)
            self.projector = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
            self.register_buffer('zero', tensor(0.), persistent = False)

        def forward(self, x, teacher_repr = None):
            h = F.relu(self.net(x))

            if not exists(teacher_repr):
                return h, self.zero

            if isinstance(teacher_repr, tuple):
                teacher_repr, _ = teacher_repr

            student_pred = self.projector(h)
            repr_loss = 1. - F.cosine_similarity(student_pred, teacher_repr.detach(), dim = -1).mean()
            return h, repr_loss

    class SelfFlowModel(nn.Module):
        def __init__(self, dim = 64):
            super().__init__()
            self.layer1 = Layer(dim)
            self.layer2 = Layer(dim)
            self.layer3 = Layer(dim)

        def forward(self, x):
            x, loss1 = self.layer1(x)
            x, loss2 = self.layer2(x)
            x, loss3 = self.layer3(x)
            return x, loss1 + loss2 + loss3

    model = SelfFlowModel(64)
    model.train()

    self_flow_ema = EMAModuleWrapper(
        model,
        beta = 0.99,
        update_after_step = 0,
        update_every = 1,
        ema_module_kwargs = {
            'layer3': ('layer1', 'teacher_repr'),
            'layer1': ('layer3', 'teacher_repr')
        }
    )
    self_flow_ema.update()

    student_input = torch.randn(2, 64)
    teacher_input = torch.randn(2, 64)

    out, repr_loss = self_flow_ema(student_input, ema_args = (teacher_input,))

    assert is_tensor(repr_loss)
    repr_loss.backward()
    self_flow_ema.update()

def test_next_latent_prediction():

    # contrived next latent prediction (Teoh et al.)
    # online predictor at position t predicts EMA encoder latent at position t+1

    class Encoder(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Linear(dim, dim)

        def forward(self, x):
            return F.relu(self.net(x))

    class Predictor(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Linear(dim, dim)
            self.register_buffer('zero', tensor(0.), persistent = False)

        def forward(self, x, target_latent = None):
            pred = self.net(x)

            if target_latent is None:
                return pred, self.zero

            if isinstance(target_latent, tuple):
                target_latent = target_latent[0]

            # online pred[:, :-1] predicts shifted EMA target[:, 1:]
            loss = F.mse_loss(pred[:, :-1], target_latent[:, 1:])
            return pred, loss

    class NextLatModel(nn.Module):
        def __init__(self, dim = 32):
            super().__init__()
            self.encoder = Encoder(dim)
            self.predictor = Predictor(dim)

        def forward(self, x):
            z = self.encoder(x)
            pred, loss = self.predictor(z)
            return pred, loss

    model = NextLatModel(32)
    model.train()

    # EMA encoder output (shifted by 1) -> online predictor as target

    ema = EMAModuleWrapper(
        model,
        beta = 0.99,
        update_after_step = 0,
        update_every = 1,
        ema_module_kwargs = {
            'predictor': ('encoder', 'target_latent')
        }
    )
    ema.update()

    seq = torch.randn(2, 8, 32)  # (batch, seq_len, dim)
    out, loss = ema(seq)

    assert is_tensor(loss)
    assert loss.item() > 0.
    loss.backward()
    ema.update()

def test_invalid_kwarg_raises_assertion_at_init():

    class BlockNoKwarg(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)

        def forward(self, x):
            return self.fc(x)

    class Model(nn.Module):
        def __init__(self, dim = 8):
            super().__init__()
            self.sub = BlockNoKwarg(dim)

        def forward(self, x):
            return self.sub(x)

    model = Model()

    with pytest.raises(AssertionError):
        EMAModuleWrapper(
            model,
            ema_module_kwargs = {'sub': 'sub'}
        )
