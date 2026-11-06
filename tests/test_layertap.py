import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from manifold_guard.layers import LayerTap, locate_layer, num_layers, role_masks, token_mask, unwrap
from manifold_guard.upstream import activation_steering

ActivationSteerer = activation_steering().ActivationSteerer

N_LAYERS = 4
ARCHS = {"llama": (LlamaConfig, LlamaForCausalLM), "qwen2": (Qwen2Config, Qwen2ForCausalLM)}


@pytest.fixture(params=sorted(ARCHS))
def model(request):
    """随机初始化的小模型，在 CPU 上几秒内跑完。"""
    cfg_cls, model_cls = ARCHS[request.param]
    torch.manual_seed(0)
    cfg = cfg_cls(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=N_LAYERS,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return model_cls(cfg).eval()


@pytest.fixture
def ids():
    return torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(0))


def _forward(model, ids, **kw):
    with torch.no_grad():
        return model(input_ids=ids, **kw)


def _hook_count(model):
    return sum(len(locate_layer(model, l)._forward_hooks) for l in range(num_layers(model)))


def test_locate_layer_matches_steerer(model):
    d = model.config.hidden_size
    for l in [*range(N_LAYERS), -1]:
        steerer = ActivationSteerer(model, torch.zeros(d), layer_idx=l)
        assert locate_layer(model, l) is steerer._locate_layer()
        assert locate_layer(model, l) is model.model.layers[l]


def test_matches_hidden_states_except_last(model, ids):
    with LayerTap(model) as tap:
        out = _forward(model, ids, output_hidden_states=True)
    hs = out.hidden_states
    assert len(hs) == N_LAYERS + 1 and sorted(tap.outputs) == list(range(N_LAYERS))
    for l in range(N_LAYERS - 1):
        assert torch.equal(tap.outputs[l], hs[l + 1]), f"第 {l} 层与 hidden_states[{l + 1}] 不一致"
    # HF 的最后一项已经过 final norm；hook 取到的是 norm 之前的值
    last = tap.outputs[N_LAYERS - 1]
    assert not torch.allclose(last, hs[N_LAYERS])
    torch.testing.assert_close(model.model.norm(last), hs[N_LAYERS])


def test_unwrap_tuple_output():
    """transformers 5 的解码层直接返回张量，旧版本返回 (hidden_states, ...) 元组，两种都要取对。"""
    h = torch.randn(2, 3, 4)
    assert unwrap(h) is h
    assert unwrap((h, None)) is h
    assert unwrap([h, "cache"]) is h


def test_tap_does_not_change_output(model, ids):
    ref = _forward(model, ids).logits
    with LayerTap(model):
        logits = _forward(model, ids).logits
    assert torch.equal(logits, ref)


def test_hooks_removed_on_exit(model, ids):
    before = _hook_count(model)
    with LayerTap(model):
        assert _hook_count(model) == before + N_LAYERS
    assert _hook_count(model) == before
    with pytest.raises(RuntimeError):
        with LayerTap(model):
            raise RuntimeError
    assert _hook_count(model) == before


def test_hook_order_with_steerer(model, ids):
    """先进入 steerer 再进入 LayerTap，看到的是 steering 之后的值；反过来则是之前的值。"""
    layer = 1
    v = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(1))
    with LayerTap(model, [layer]) as clean:
        _forward(model, ids)
    clean = clean.outputs[layer]

    with ActivationSteerer(model, v, coeff=2.0, layer_idx=layer, positions="all"), \
            LayerTap(model, [layer]) as after:
        _forward(model, ids)
    assert torch.equal(after.outputs[layer], clean + 2.0 * v)

    with LayerTap(model, [layer]) as before, \
            ActivationSteerer(model, v, coeff=2.0, layer_idx=layer, positions="all"):
        _forward(model, ids)
    assert torch.equal(before.outputs[layer], clean)


def test_positions_last(model, ids):
    with LayerTap(model, [0, 2]) as full:
        _forward(model, ids)
    with LayerTap(model, [0, 2], positions="last") as last:
        _forward(model, ids)
    for l in (0, 2):
        assert last.outputs[l].shape == (2, 1, model.config.hidden_size)
        assert torch.equal(last.outputs[l], full.outputs[l][:, -1:])


def test_detach(model, ids):
    with LayerTap(model, [1]) as tap:
        model(input_ids=ids)
    assert not tap.outputs[1].requires_grad

    with LayerTap(model, [1], detach=False) as tap:
        model(input_ids=ids)
    tap.outputs[1].pow(2).sum().backward()
    grad = model.model.layers[0].mlp.down_proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_layer_indices(model):
    assert LayerTap(model, [-1, 0, N_LAYERS - 1]).layers == [0, N_LAYERS - 1]
    with pytest.raises(IndexError):
        LayerTap(model, [N_LAYERS])
    with pytest.raises(ValueError):
        LayerTap(model, positions="response")


# ---------- token 掩码 ----------

def _bool(rows):
    return torch.tensor(rows, dtype=torch.bool)


def test_token_mask_left_and_right_padding():
    left = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    right = torch.tensor([[1, 1, 1, 0, 0]])
    assert torch.equal(token_mask(left), _bool([[0, 0, 0, 1, 1], [0, 1, 1, 1, 1]]))
    assert torch.equal(token_mask(right), _bool([[0, 1, 1, 0, 0]]))
    assert torch.equal(token_mask(left, drop_first=False), left.bool())


def test_role_masks():
    am = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    m = role_masks(am, prompt_lens=[2, 3])
    assert torch.equal(m["prompt"], _bool([[0, 0, 0, 1, 0], [0, 1, 1, 0, 0]]))
    assert torch.equal(m["response"], _bool([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]))
    assert torch.equal(m["prompt"] | m["response"], token_mask(am))
    assert not (m["prompt"] & m["response"]).any()
