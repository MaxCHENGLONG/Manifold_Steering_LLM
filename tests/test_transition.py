import pytest
import torch

from manifold_guard.calibrate import calibrate
from manifold_guard.metrics import auroc
from manifold_guard.scorers import GaussScorer, make_scorer, required_layers
from manifold_guard.stats import StreamingMoments
from manifold_guard.trajectory import LinearGaussianTransition, fit_transition

D = 8


def _truth(seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.eye(D, dtype=torch.float64) + 0.5 * torch.randn(D, D, generator=g, dtype=torch.float64)
    b = torch.randn(D, generator=g, dtype=torch.float64)
    L = 0.3 * torch.randn(D, D, generator=g, dtype=torch.float64)
    return A, b, L @ L.T + 0.05 * torch.eye(D, dtype=torch.float64)


def _sample(n, A, b, S, seed, Q=None):
    """x ~ N(0, I)，y = A Q x + b + ε，ε ~ N(0, S)。Q 正交时 y 的边缘分布与 Q = I 时相同，但转移不同。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D, generator=g, dtype=torch.float64)
    e = torch.randn(n, D, generator=g, dtype=torch.float64) @ torch.linalg.cholesky(S).T
    xq = x if Q is None else x @ Q.T
    return x, xq @ A.T + b + e


@pytest.fixture(scope="module")
def stats_dir(tmp_path_factory):
    """按 collect_layer_stats / collect_pair_stats 的格式写出两层的逐层统计、保留样本和相邻层联合统计。"""
    root = tmp_path_factory.mktemp("stats")
    (root / "pairs").mkdir()
    A, b, S = _truth()
    x, y = _sample(300_000, A, b, S, seed=1)
    joint = StreamingMoments(2 * D, device="cpu")
    joint.update(torch.cat([x, y], dim=1))
    cov = joint.m2 / (joint.n - 1)
    torch.save({"layer": 1, "n": joint.n, "mean": joint.mean, "cov": (cov + cov.T) / 2}, root / "pairs" / "pair_01.pt")
    for l, h in ((0, x), (1, y)):
        m = StreamingMoments(D, device="cpu")
        m.update(h)
        s = {k: v.float() if torch.is_tensor(v) else v for k, v in m.finalize().items()}
        torch.save({"layer": l, **s}, root / f"layer_{l:02d}.pt")
    xr, yr = _sample(20_000, A, b, S, seed=2)  # 两层的保留样本逐行对齐
    for l, h in ((0, xr), (1, yr)):
        torch.save({"layer": l, "samples": h.half(), "seen": len(h)}, root / f"reservoir_{l:02d}.pt")
    return root


def test_fit_transition_recovers_map(stats_dir):
    A, b, S = _truth()
    s = torch.load(stats_dir / "pairs" / "pair_01.pt")
    A_hat, b_hat, S_hat = fit_transition(s["mean"], s["cov"], ridge=0)
    torch.testing.assert_close(A_hat, A, atol=0.01, rtol=0)
    torch.testing.assert_close(b_hat, b, atol=0.01, rtol=0)
    torch.testing.assert_close(S_hat, S, atol=0.005, rtol=0)


def test_score_is_chi2_over_d_on_natural_pairs(stats_dir):
    sc = LinearGaussianTransition(ridge=0, shrink=0).load(stats_dir, device="cpu")
    assert sc.layers == [1] and sc.needs_prev
    A, b, S = _truth()
    x, y = _sample(50_000, A, b, S, seed=3)
    s = sc.score(y.float(), 1, prev=x.float()).double() * D
    assert s.mean().item() == pytest.approx(D, rel=0.02)
    assert s.var().item() == pytest.approx(2 * D, rel=0.06)
    with pytest.raises(ValueError):
        sc.score(y.float(), 1)


def test_transition_sees_changed_map_that_marginal_misses(stats_dir):
    """Q 正交：第 1 层的边缘分布不变，逐层的高斯打分器看不出来；转移打分器能看出来。"""
    A, b, S = _truth()
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=torch.Generator().manual_seed(5), dtype=torch.float64))
    x0, y0 = _sample(20_000, A, b, S, seed=6)
    x1, y1 = _sample(20_000, A, b, S, seed=7, Q=Q)
    gauss = GaussScorer(shrink=0).load(stats_dir, device="cpu")
    trans = make_scorer("trans:ridge=0,shrink=0").load(stats_dir, device="cpu")
    assert auroc(gauss.score(y0.float(), 1), gauss.score(y1.float(), 1)) == pytest.approx(0.5, abs=0.02)
    assert auroc(trans.score(y0.float(), 1, x0.float()), trans.score(y1.float(), 1, x1.float())) > 0.95


def test_calibrate_reads_previous_layer(stats_dir):
    trans = LinearGaussianTransition(ridge=0, shrink=0).load(stats_dir, device="cpu")
    c = calibrate(trans, stats_dir)
    assert list(c["layers"]) == [1] and "trajectory" in c
    # d·分数 服从 χ²_8，中位数约 7.34
    assert c["layers"][1]["median"] * D == pytest.approx(7.34, rel=0.03)
    assert required_layers([trans]) == [0, 1]
    assert required_layers([GaussScorer().load(stats_dir, device="cpu"), trans]) == [0, 1]
