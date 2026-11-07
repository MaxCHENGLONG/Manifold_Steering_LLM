from .base import LayerwiseScorer


class GaussScorer(LayerwiseScorer):
    """收缩协方差 (1−λ)Σ + λ·(tr Σ / d)·I 下的平方马氏距离除以 d。

    λ=0 时自然样本上的均值约为 1（check_layer_stats.py 的 maha_mean）。λ 越大，越少依赖小特征值方向，
    对这些方向上的偏离也越不敏感。白化矩阵用 float64 算好再转成 float32 放在显存里。
    """

    def __init__(self, shrink=0.01):
        self.shrink = float(shrink)
        self.name = f"gauss:shrink={self.shrink:g}"

    def _prepare(self, s):
        ev, V = s["eigvals"].double(), s["eigvecs"].double()
        ev = ev.clamp_min(ev[0].item() * 1e-12)  # eigh 可能给出极小的负特征值
        ev = (1 - self.shrink) * ev + self.shrink * ev.mean()
        return {"mean": s["mean"].float(), "W": (V / ev.sqrt()).float()}

    def _score(self, h, p):
        return ((h - p["mean"]) @ p["W"]).pow(2).sum(dim=-1) / p["W"].shape[1]
