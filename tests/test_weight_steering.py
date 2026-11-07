import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from manifold_guard.weight_steering import WeightSteerer, compute_delta, merged_params


def _tiny(tied=True):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=tied)
    return LlamaForCausalLM(cfg).eval()


def _adapter(base_dir, out_dir, seed):
    """随机初始化的 LoRA（B 不为 0），共享的 embedding 也加一点扰动，模拟训练后的适配器。"""
    torch.manual_seed(seed)
    model = LlamaForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", modules_to_save=["embed_tokens", "lm_head"],
                     ensure_weight_tying=True, init_lora_weights=False, task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    with torch.no_grad():
        pm.base_model.model.model.embed_tokens.modules_to_save["default"].weight.add_(0.01 * torch.randn(128, 32))
    pm.save_pretrained(out_dir)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    _tiny().save_pretrained(root / "base")
    _adapter(root / "base", root / "pos", seed=1)
    _adapter(root / "base", root / "neg", seed=2)
    return root


def test_delta_is_pos_minus_neg_and_keeps_tying(setup):
    delta = compute_delta(str(setup / "base"), str(setup / "pos"), str(setup / "neg"))
    pos = merged_params(str(setup / "base"), str(setup / "pos"))
    neg = merged_params(str(setup / "base"), str(setup / "neg"))
    assert "lm_head.weight" not in pos, "合并后共享的 embedding 应只出现一次"
    assert "model.embed_tokens.weight" in delta
    assert any(".q_proj." in n for n in delta)
    for n, d in delta.items():
        torch.testing.assert_close(d, pos[n] - neg[n], rtol=0, atol=0)


def test_steerer_k0_is_identity_and_restores(setup):
    delta = compute_delta(str(setup / "base"), str(setup / "pos"), str(setup / "neg"))
    model = LlamaForCausalLM.from_pretrained(setup / "base", dtype=torch.float32).eval()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ids = torch.randint(0, 128, (2, 8), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
        with WeightSteerer(model, delta, 0):
            same = model(input_ids=ids).logits
        with WeightSteerer(model, delta, 1.5):
            steered = dict(model.named_parameters())
            for n, d in delta.items():
                torch.testing.assert_close(steered[n], before[n] + 1.5 * d)
            assert not torch.allclose(model(input_ids=ids).logits, ref)
    assert torch.equal(same, ref)
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), f"{n} 没有恢复"
    # 共享权重：改 embed 时 lm_head 跟着变，恢复后也一起恢复
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_steerer_missing_param_raises(setup):
    model = _tiny()
    with pytest.raises(KeyError):
        WeightSteerer(model, {"no.such.weight": torch.zeros(1)}, 1.0)
