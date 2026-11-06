import numpy as np
import pytest
import torch

from manifold_guard.stats import Reservoir, StreamingMoments

D = 16


def _data(n=3000, d=D, seed=0):
    """各维尺度相差三个数量级、均值远离 0，专门考验数值稳定性。"""
    g = torch.Generator().manual_seed(seed)
    scale = torch.logspace(-2, 1, d, dtype=torch.float64)
    return torch.randn(n, d, generator=g, dtype=torch.float64) * scale + 5.0


def _moments(x, chunks=None, **kw):
    m = StreamingMoments(x.shape[1], device="cpu", **kw)
    for part in (x.split(chunks) if chunks else [x]):
        m.update(part)
    return m


def _rel_err(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


# ---------- StreamingMoments ----------

def test_matches_numpy():
    x = _data()
    r = _moments(x).finalize()
    assert r["n"] == len(x)
    assert _rel_err(r["mean"], torch.from_numpy(x.numpy().mean(axis=0))) < 1e-6
    assert _rel_err(r["cov"], torch.from_numpy(np.cov(x.numpy(), rowvar=False))) < 1e-6


def test_chunked_equals_single():
    x = _data()
    one = _moments(x).finalize()
    for chunks in (1, 7, 500):
        r = _moments(x, chunks=chunks).finalize()
        torch.testing.assert_close(r["mean"], one["mean"], rtol=1e-12, atol=0)
        torch.testing.assert_close(r["cov"], one["cov"], rtol=1e-10, atol=1e-14)


def test_internal_chunking_equals_single():
    x = _data()
    torch.testing.assert_close(_moments(x, chunk_rows=64).finalize()["cov"],
                               _moments(x).finalize()["cov"], rtol=1e-10, atol=1e-14)


def test_merge_equals_single():
    x = _data()
    a, b = _moments(x[:1234], seed=1), _moments(x[1234:], seed=2)
    a.merge(b)
    one = _moments(x).finalize()
    r = a.finalize()
    assert r["n"] == len(x)
    torch.testing.assert_close(r["cov"], one["cov"], rtol=1e-10, atol=1e-14)

    empty = StreamingMoments(D, device="cpu")
    empty.merge(a)
    torch.testing.assert_close(empty.finalize()["cov"], one["cov"], rtol=1e-10, atol=1e-14)


def test_low_precision_input():
    """bf16 输入在 float64 里累计，结果等于对量化后数据直接算。"""
    xb = _data().to(torch.bfloat16)
    r = _moments(xb).finalize()
    ref = np.cov(xb.double().numpy(), rowvar=False)
    assert _rel_err(r["cov"], torch.from_numpy(ref)) < 1e-6


def test_empty_update_is_noop():
    x = _data()
    m = _moments(x)
    m.update(x[:0])
    assert m.n == len(x)
    torch.testing.assert_close(m.finalize()["cov"], _moments(x).finalize()["cov"], rtol=0, atol=0)


def test_wrong_shape_raises():
    m = StreamingMoments(D, device="cpu")
    with pytest.raises(ValueError):
        m.update(torch.zeros(10, D + 1))
    with pytest.raises(ValueError):
        m.update(torch.zeros(D))


def test_finalize_needs_two_samples():
    m = StreamingMoments(D, device="cpu")
    m.update(_data(n=1))
    with pytest.raises(ValueError):
        m.finalize()


def test_eigendecomposition():
    r = _moments(_data()).finalize()
    vals, vecs = r["eigvals"], r["eigvecs"]
    assert torch.all(vals[:-1] >= vals[1:]), "特征值应从大到小"
    torch.testing.assert_close(r["cov"] @ vecs, vecs * vals, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(vecs.T @ vecs, torch.eye(D, dtype=torch.float64), rtol=0, atol=1e-10)


def test_norm_quantiles_exact_when_sample_holds_everything():
    x = _data(n=500)
    r = _moments(x, chunks=64, norm_sample=1000).finalize()
    norms = x.norm(dim=1).float()
    for q, v in r["norm_quantiles"].items():
        assert v == pytest.approx(torch.quantile(norms, q).item(), rel=1e-6)


def test_state_dict_roundtrip():
    x = _data()
    a = _moments(x[:1000])
    a = StreamingMoments.from_state_dict(a.state_dict(), device="cpu")
    a.update(x[1000:])
    r, one = a.finalize(), _moments(x).finalize()
    assert r["n"] == one["n"]
    torch.testing.assert_close(r["cov"], one["cov"], rtol=1e-10, atol=1e-14)


# ---------- Reservoir ----------

def _ids(n):
    """每行的内容就是行号，用来检查保留下来的是哪些行、行内容有没有错位。"""
    return torch.arange(n, dtype=torch.float32).view(-1, 1).repeat(1, 3)


def _feed(res, x, sizes):
    for part in x.split(sizes):
        res.update(part)


def _reference_ids(seed, sizes, k):
    """用同样的随机数序列重放键，取键最小的 k 行，作为 bottom-k 抽样的标准答案。"""
    g = torch.Generator().manual_seed(seed)
    keys = torch.cat([torch.rand(s, dtype=torch.float64, generator=g) for s in sizes])
    return set(torch.topk(keys, min(k, len(keys)), largest=False).indices.tolist()), keys


def test_reservoir_keeps_everything_below_capacity():
    x = _ids(40)
    r = Reservoir(100, 3, device="cpu", dtype=torch.float32)
    _feed(r, x, 7)
    assert r.seen == 40
    torch.testing.assert_close(r.data, x, rtol=0, atol=0)


@pytest.mark.parametrize("sizes", [[1000], [7] * 142 + [6], [50, 30, 920]])
def test_reservoir_matches_bottom_k(sizes):
    n, k = sum(sizes), 100
    r = Reservoir(k, 3, device="cpu", dtype=torch.float32, seed=3)
    for part in _ids(n).split(sizes):
        r.update(part)
    kept = r.data[:, 0].long()
    expected, _ = _reference_ids(3, sizes, k)
    assert r.seen == n and len(kept) == k
    assert set(kept.tolist()) == expected
    assert torch.all(r.data == kept.view(-1, 1).float()), "行内容和行号错位"


def test_reservoir_merge_is_bottom_k_of_union():
    k, n = 100, 1000
    x = _ids(2 * n)
    a = Reservoir(k, 3, device="cpu", dtype=torch.float32, seed=1)
    b = Reservoir(k, 3, device="cpu", dtype=torch.float32, seed=2)
    _feed(a, x[:n], 64)
    _feed(b, x[n:], 64)
    a.merge(b)

    sizes = [64] * (n // 64) + [n % 64]
    _, keys_a = _reference_ids(1, sizes, k)
    _, keys_b = _reference_ids(2, sizes, k)
    expected = set(torch.topk(torch.cat([keys_a, keys_b]), k, largest=False).indices.tolist())
    assert a.seen == 2 * n
    assert set(a.data[:, 0].long().tolist()) == expected


def test_reservoir_same_seed_selects_same_positions():
    """同一 seed、同样分块时，不同层保留的是同一批 token，可以逐行对齐。"""
    x = _data(n=2000, d=4)
    r1 = Reservoir(50, 4, device="cpu", dtype=torch.float64, seed=0)
    r2 = Reservoir(50, 4, device="cpu", dtype=torch.float64, seed=0)
    _feed(r1, x, 128)
    _feed(r2, 2 * x, 128)
    torch.testing.assert_close(r2.data, 2 * r1.data, rtol=0, atol=0)


def test_reservoir_is_roughly_uniform():
    n, k = 20000, 2000
    r = Reservoir(k, 3, device="cpu", dtype=torch.float32, seed=0)
    _feed(r, _ids(n), 512)
    ids = r.data[:, 0]
    assert ids.unique().numel() == k
    # 均匀抽样时行号均值的标准差约为 n/sqrt(12k) ≈ 129，这里放宽到 5 倍
    assert abs(ids.mean().item() - (n - 1) / 2) < 5 * n / (12 * k) ** 0.5


def test_reservoir_state_dict_roundtrip():
    sizes = [64] * 30
    n, k = sum(sizes), 100
    x = _ids(n)
    r = Reservoir(k, 3, device="cpu", dtype=torch.float32, seed=5)
    for part in x[:960].split(64):
        r.update(part)
    r = Reservoir.from_state_dict(r.state_dict(), device="cpu")
    for part in x[960:].split(64):
        r.update(part)
    expected, _ = _reference_ids(5, sizes, k)
    assert set(r.data[:, 0].long().tolist()) == expected
