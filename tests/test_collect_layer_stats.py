import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from manifold_guard.corpus import ROLES, Sequence
from manifold_guard.layers import LayerTap
from scripts.collect_layer_stats import collect, save

LAYERS = [0, 2, 3]
D = 32


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=D, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def seqs():
    """两种来源的随机序列，三个切分都要有。"""
    g = torch.Generator().manual_seed(0)
    out, i = [], 0
    while not all(sum(s.source == src and s.split == sp for s in out) >= 3
                  for src in ("fineweb", "chat") for sp in ("calib", "holdout")):
        n = int(torch.randint(4, 30, (1,), generator=g))
        ids = torch.randint(0, 128, (n,), generator=g).tolist()
        if i % 2:
            out.append(Sequence("chat", f"c{i}", ids, int(torch.randint(1, n, (1,), generator=g))))
        else:
            out.append(Sequence("fineweb", f"f{i}", ids, n))
        i += 1
    return out


def _reference(model, seqs):
    """逐条、不 padding 地前向，按 (角色, 切分) 收集各层激活。"""
    ref = {(r, sp): {l: [] for l in LAYERS} for r in ROLES for sp in ("fit", "calib")}
    with torch.no_grad(), LayerTap(model, LAYERS) as tap:
        for s in seqs:
            if s.split == "holdout":
                continue
            model(input_ids=torch.tensor([s.input_ids]))
            n, p = len(s.input_ids), s.prompt_len
            pos = {"text": range(1, n)} if s.source == "fineweb" else {"prompt": range(1, p), "response": range(p, n)}
            for role, idx in pos.items():
                for l in LAYERS:
                    ref[(role, s.split)][l].append(tap.outputs[l][0, list(idx)])
    return {k: {l: torch.cat(v) if v else torch.empty(0, D) for l, v in per.items()} for k, per in ref.items()}


def _collect(model, seqs, reservoir_k=10_000):
    return collect(model, seqs, LAYERS, pad_id=0, batch_tokens=128, reservoir_k=reservoir_k,
                   reservoir_device="cpu")


def _assert_same_rows(a, b, atol=1e-2):
    """a、b 是同一组行的不同排列（允许 float16 的舍入误差）：每行的最近邻一一对应且足够近。"""
    dist = torch.cdist(a.float(), b.float())
    nearest = dist.argmin(dim=1)
    assert sorted(nearest.tolist()) == list(range(len(b))), "行没有一一对应"
    assert dist.min(dim=1).values.max() < atol * a.shape[1] ** 0.5


def test_collect_matches_unpadded_reference(model, seqs):
    moments, reservoirs, counts = _collect(model, seqs)
    ref = _reference(model, seqs)
    for role in ROLES:
        for sp in ("fit", "calib"):
            assert counts[role][sp] == len(ref[(role, sp)][LAYERS[0]]) > 0, (role, sp)
        for l in LAYERS:
            x = ref[(role, "fit")][l].double()
            r = moments[role][l].finalize()
            assert r["n"] == len(x)
            torch.testing.assert_close(r["mean"], x.mean(0), rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(r["cov"], torch.cov(x.T), rtol=1e-4, atol=1e-5)
            # 容量足够时 calib 的激活全部保留（顺序不同）
            c, res = ref[(role, "calib")][l], reservoirs[role][l]
            assert res.seen == len(c) == len(res.data)
            _assert_same_rows(res.data, c.half())


def test_reservoir_positions_aligned_across_layers(model, seqs):
    _, reservoirs, counts = _collect(model, seqs, reservoir_k=5)
    for role in ROLES:
        assert counts[role]["calib"] > 5
        keys = [reservoirs[role][l].keys for l in LAYERS]
        assert all(torch.equal(k, keys[0]) for k in keys[1:]), f"{role} 各层保留的 token 位置不一致"


def test_save_writes_layer_files(model, seqs, tmp_path):
    moments, reservoirs, counts = _collect(model, seqs)
    save(tmp_path, moments, reservoirs, counts)
    for role in ROLES:
        for l in LAYERS:
            st = torch.load(tmp_path / role / f"layer_{l:02d}.pt")
            assert (st["role"], st["layer"], st["n"]) == (role, l, counts[role]["fit"])
            assert st["cov"].dtype == torch.float32 and st["cov"].shape == (D, D)
            assert st["eigvals"].shape == (D,) and st["eigvecs"].shape == (D, D)
            assert set(st["norm_quantiles"]) == {0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99}
            res = torch.load(tmp_path / role / f"reservoir_{l:02d}.pt")
            assert res["samples"].dtype == torch.float16 and res["samples"].shape == (counts[role]["calib"], D)
