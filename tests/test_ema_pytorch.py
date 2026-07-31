from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ema_pytorch import EMA, EMAPytree, EMATensor, PostHocEMA

def exists(val):
    return val is not None

def test_readme_basic_ema():
    net = nn.Linear(512, 512)

    ema = EMA(
        net,
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
    )

    with torch.no_grad():
        net.weight.copy_(torch.randn_like(net.weight))
        net.bias.copy_(torch.randn_like(net.bias))

    ema.update()

    data = torch.randn(1, 512)
    output = net(data)
    ema_output = ema(data)

    assert output.shape == (1, 512)
    assert ema_output.shape == (1, 512)

def test_readme_post_hoc_ema():
    net = nn.Linear(512, 512)

    emas = PostHocEMA(
        net,
        sigma_rels = (0.05, 0.28),
        update_every = 1,
        checkpoint_every_num_steps = 1,
        checkpoint_folder = './post-hoc-ema-checkpoints-test'
    )

    net.train()

    for _ in range(5):
        with torch.no_grad():
            net.weight.copy_(torch.randn_like(net.weight))
            net.bias.copy_(torch.randn_like(net.bias))

        emas.update()

    synthesized_ema = emas.synthesize_ema_model(sigma_rel = 0.15)
    data = torch.randn(1, 512)
    synthesized_ema_output = synthesized_ema(data)

    assert synthesized_ema_output.shape == (1, 512)

def test_readme_switch_ema():
    net = nn.Linear(512, 512)

    ema = EMA(
        net,
        beta = 0.99,
        update_after_step = 0,
        update_every = 1,
        update_model_with_ema_every = 5
    )

    with torch.no_grad():
        net.weight.add_(torch.ones_like(net.weight))

    for _ in range(5):
        ema.update()

def test_ema_tensor_pytree():

    online_tree = {
        'w': torch.randn(10, 10),
        'b': torch.randn(10)
    }

    ema = EMA(
        online_tree,
        beta = 0.5,
        min_value = 0.5,
        update_after_step = 0,
        update_every = 1
    )

    ema.update()

    with torch.no_grad():
        online_tree['w'].add_(torch.ones(10, 10))
        online_tree['b'].add_(torch.ones(10))

    old_ema_w = ema.ema_model['w'].clone()
    ema.update()

    expected_w = 0.5 * old_ema_w + 0.5 * online_tree['w']
    assert torch.allclose(ema.ema_model['w'], expected_w)

def test_ema_tensor_detach_and_device():
    tensor1 = torch.randn(10, 10, requires_grad = True)
    tensor2 = torch.randn(10, 10, requires_grad = True)

    ema = EMATensor(
        (tensor1, tensor2),
        beta = 0.5,
        update_after_step = 0,
        update_every = 1,
        allow_different_devices = True
    )

    ema_tensor1, ema_tensor2 = ema.ema_model
    assert not ema_tensor1.requires_grad
    assert not ema_tensor2.requires_grad

    loss = (tensor1 ** 2).sum() + (tensor2 ** 2).sum()
    loss.backward()

    ema.update()

    ema_tensor1, ema_tensor2 = ema.ema_model
    assert not ema_tensor1.requires_grad
    assert not ema_tensor2.requires_grad
