from contextlib import nullcontext

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from manifold_guard.layers import LayerTap
from manifold_guard.steering import (MaskedSteerer, absolute_coeff, act_steerer, greedy_config, parse_choice,
                                     steer_mask)

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


@pytest.mark.parametrize("layer", [2, 0])
def test_masked_steerer_full_mask_matches_act_steerer(model, vector, layer):
    ids = torch.randint(0, 128, (2, 7), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        with act_steerer(model, vector, 1.5, layer, positions="all"):
            ref = model(input_ids=ids).logits
        with MaskedSteerer(model, vector, 1.5, layer) as s:
            s.mask = torch.ones_like(ids, dtype=torch.bool)
            got = model(input_ids=ids).logits
        plain = model(input_ids=ids).logits
    assert torch.equal(got, ref)
    assert not torch.allclose(plain, ref), "退出后 hook 应已移除"


def test_steer_mask():
    am = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    m = steer_mask(am, [3, 1])
    assert m.tolist() == [[False, False, True, True, True, False], [True] * 6]


def test_masked_steerer_reproduces_steered_generation(model, vector):
    """生成时 positions="response" 的 steering，等于把 prompt + 回答一次喂进去、按 steer_mask 加 steering。"""
    prompt = torch.randint(0, 128, (1, 5), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        with act_steerer(model, vector, 4.0, 2):
            gen = model.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
                                 max_new_tokens=6, output_logits=True, return_dict_in_generate=True,
                                 pad_token_id=0)
        seq = gen.sequences
        with MaskedSteerer(model, vector, 4.0, 2) as s:
            s.mask = steer_mask(torch.ones_like(seq), [prompt.shape[1]])
            tf = model(input_ids=seq).logits
    gen_logits = torch.stack(gen.logits, dim=1)                   # 每步生成用的 logits
    tf_logits = tf[:, prompt.shape[1] - 1: seq.shape[1] - 1]      # 同样位置的 teacher-forcing logits
    torch.testing.assert_close(tf_logits, gen_logits, rtol=1e-4, atol=1e-4)


def test_greedy_config_keeps_repetition_penalty(model):
    cfg = greedy_config(model, 50)
    assert cfg.do_sample is False and cfg.max_new_tokens == 50 and cfg.repetition_penalty == 1.1
    assert model.generation_config.max_new_tokens != 50, "不应改动模型自带的 generation_config"
