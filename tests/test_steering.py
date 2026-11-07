from contextlib import nullcontext

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from manifold_guard.layers import LayerTap
from manifold_guard.steering import absolute_coeff, act_steerer, greedy_config, parse_choice

N_LAYERS, D = 4, 32


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=D, intermediate_size=64, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    m = LlamaForCausalLM(cfg).eval()
    m.generation_config.repetition_penalty = 1.1
    return m


@pytest.fixture
def vector():
    return torch.randn(N_LAYERS + 1, D, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("text,expected", [
    ("Some reasoning.\nFinal Answer: C", "C"),
    ("final answer: (b)", "B"),
    ("**Final Answer:** D.", "D"),
    ("**Final Answer**: A", "A"),
    ("Final Answer: B ... wait, Final Answer: C", "C"),  # 取最后一个
    ("Final Answer: Apple", None),                     # 字母后面不能紧跟字母
    ("The answer is C", None),
])
def test_parse_choice(text, expected):
    assert parse_choice(text) == expected


def test_absolute_coeff(vector):
    c = absolute_coeff(0.5, vector, 2, median_norm=10.0)
    assert c * vector[2].norm().item() == pytest.approx(0.5 * 10.0)


def test_single_layer_matches_select_steerer_convention(model, vector):
    """layer=2 时在 model.layers[1] 上加 c·vector[2]。"""
    ids = torch.randint(0, 128, (2, 6), generator=torch.Generator().manual_seed(0))
    with torch.no_grad(), LayerTap(model, [1]) as clean:
        model(input_ids=ids)
    with torch.no_grad(), act_steerer(model, vector, 3.0, 2, positions="all"), LayerTap(model, [1]) as tap:
        model(input_ids=ids)
    torch.testing.assert_close(tap.outputs[1], clean.outputs[1] + 3.0 * vector[2])


def test_all_layers_variant(model, vector):
    """layer=0：第 l-1 层加 c·(vector[l] − vector[l-1])（l=1 时为 vector[1]），最后一层不加。"""
    steerer = act_steerer(model, vector, 2.0, 0)
    subs = steerer._steerers
    assert [s.layer_idx for s in subs] == list(range(N_LAYERS - 1))
    torch.testing.assert_close(subs[0].vector, vector[1])
    for s in subs[1:]:
        l = s.layer_idx + 1
        torch.testing.assert_close(s.vector, vector[l] - vector[l - 1])
    assert all(s.coeff == 2.0 and s.positions == "response" for s in subs)


def test_zero_coeff_is_noop(model, vector):
    assert isinstance(act_steerer(model, vector, 0.0, 2), nullcontext)


def test_greedy_config_keeps_repetition_penalty(model):
    cfg = greedy_config(model, 50)
    assert cfg.do_sample is False and cfg.max_new_tokens == 50 and cfg.repetition_penalty == 1.1
    assert model.generation_config.max_new_tokens != 50, "不应改动模型自带的 generation_config"
