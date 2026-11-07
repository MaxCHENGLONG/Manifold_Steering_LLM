import pytest
import torch

from manifold_guard.calibrate import (calibrate, load_calibration, save_calibration, standardize,
                                      trajectory_score)
from manifold_guard.metrics import auroc
from manifold_guard.scorers import GaussScorer, NormScorer, PCAScorer, make_scorer
from manifold_guard.stats import StreamingMoments

D, LAYERS = 16, (0, 3)


def _sample(n, mean, A, g):
    return mean + torch.randn(n, D, generator=g, dtype=torch.float64) @ A.T


@pytest.fixture(scope="module")
def stats_dir(tmp_path_factory):
    """两层合成高斯数据，按 collect_layer_stats.save 的格式写出 layer_XX.pt 和 reservoir_XX.pt。"""
    root = tmp_path_factory.mktemp("stats")
    g = torch.Generator().manual_seed(0)
    for l in LAYERS:
        mean = 5 * torch.randn(D, generator=g, dtype=torch.float64)
        A = torch.randn(D, D, generator=g, dtype=torch.float64) * torch.logspace(0, -2, D, dtype=torch.float64)
        m = StreamingMoments(D, device="cpu")
        m.update(_sample(200_000, mean, A, g))
        s = {k: v.float() if torch.is_tensor(v) else v for k, v in m.finalize().items()}
        torch.save({"role": "response", "layer": l, **s}, root / f"layer_{l:02d}.pt")
        res = _sample(20_000, mean, A, g).half()
        torch.save({"role": "response", "layer": l, "samples": res, "seen": 20_000}, root / f"reservoir_{l:02d}.pt")
        torch.save({"mean": mean, "A": A}, root / f"truth_{l:02d}.pt")
    return root


def _fresh(stats_dir, l, n=50_000, seed=1):
    t = torch.load(stats_dir / f"truth_{l:02d}.pt")
    return _sample(n, t["mean"], t["A"], torch.Generator().manual_seed(seed)).float()


def test_make_scorer_names():
    assert make_scorer("norm").name == "norm"
    assert make_scorer("gauss").name == "gauss:shrink=0.01"
    assert make_scorer("gauss:shrink=0.1").shrink == 0.1
    assert make_scorer("pca:var=0.95").name == "pca:var=0.95"
    with pytest.raises(ValueError):
        make_scorer("glp")


def test_gauss_is_chi2_over_d(stats_dir):
    """λ=0：d·分数 服从 χ²_d，均值 d、方差 2d。"""
    sc = GaussScorer(shrink=0).load(stats_dir, device="cpu")
    for l in LAYERS:
        s = sc.score(_fresh(stats_dir, l), l).double() * D
        assert s.mean().item() == pytest.approx(D, rel=0.02)
        assert s.var().item() == pytest.approx(2 * D, rel=0.05)


def test_gauss_full_shrink_is_scaled_euclidean(stats_dir):
    sc = GaussScorer(shrink=1).load(stats_dir, device="cpu")
    s = torch.load(stats_dir / "layer_03.pt")
    h = _fresh(stats_dir, 3, n=100)
    expected = (h - s["mean"]).pow(2).sum(-1) / (s["eigvals"].double().mean().item() * D)
    torch.testing.assert_close(sc.score(h, 3), expected.float(), rtol=1e-4, atol=1e-6)


def test_pca_residual(stats_dir):
    s = torch.load(stats_dir / "layer_00.pt")
    sc = PCAScorer(var=0.9).load(stats_dir, device="cpu")
    k = sc.params[0]["k"]
    assert 1 <= k < D
    inside = s["mean"] + torch.randn(100, k) @ s["eigvecs"][:, :k].T * 3
    assert sc.score(inside, 0).abs().max().item() < 1e-4, "主子空间内的点残差应为 0"
    assert sc.score(_fresh(stats_dir, 0), 0).mean().item() == pytest.approx(1, rel=0.03)


def test_norm(stats_dir):
    sc = NormScorer().load(stats_dir, device="cpu")
    med = torch.load(stats_dir / "layer_03.pt")["norm_quantiles"][0.5]
    h = torch.randn(4, D)
    h = h / h.norm(dim=-1, keepdim=True) * med * torch.tensor([[0.5], [1], [2], [3]])
    torch.testing.assert_close(sc.score(h, 3), torch.tensor([0.5, 1, 2, 3]).log().abs())


def test_calibrate_standardizes_and_roundtrips(stats_dir, tmp_path):
    sc = GaussScorer().load(stats_dir, device="cpu")
    calib = calibrate(sc, stats_dir)
    assert set(calib["layers"]) == set(LAYERS)
    for l in LAYERS:
        res = torch.load(stats_dir / f"reservoir_{l:02d}.pt")["samples"]
        z = standardize(sc.score(res, l).double(), calib, l)
        q25, q50, q75 = torch.quantile(z, torch.tensor([0.25, 0.5, 0.75], dtype=z.dtype)).tolist()
        assert q50 == pytest.approx(0, abs=1e-6) and q75 - q25 == pytest.approx(1, abs=1e-6)
        zq = list(calib["layers"][l]["z_quantiles"].values())
        assert zq == sorted(zq) and zq[0] > 0
    save_calibration(calib, tmp_path / "c.json")
    assert load_calibration(tmp_path / "c.json") == calib


def test_trajectory_score():
    per = {0: torch.tensor([1.0, 5.0]), 2: torch.tensor([3.0, -1.0])}
    torch.testing.assert_close(trajectory_score(per, "max"), torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(trajectory_score(per, "mean"), torch.tensor([2.0, 2.0]))


def test_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    g = torch.Generator().manual_seed(0)
    pos = torch.randint(0, 20, (300,), generator=g).float()
    neg = torch.randint(3, 25, (200,), generator=g).float()
    y = [0] * 300 + [1] * 200
    assert auroc(pos, neg) == pytest.approx(roc_auc_score(y, torch.cat([pos, neg]).numpy()), abs=1e-12)
    assert auroc(pos, pos.clone()) == 0.5
    assert auroc(torch.zeros(5), torch.ones(3)) == 1.0
