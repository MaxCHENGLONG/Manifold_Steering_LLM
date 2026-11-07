from .base import LayerwiseScorer


class PCAScorer(LayerwiseScorer):
    """去掉前 k 个主成分后的剩余能量，除以自然样本上的期望值（其余特征值之和），自然样本上均值约为 1。

    k 取累计方差达到 var 的最小维数。直接投到剩余的 d−k 个主方向上算，不用 ‖h−μ‖² 减去主子空间能量，
    避免两个大数相减在 float32 里丢精度。
    """

    def __init__(self, var=0.99):
        self.var = float(var)
        self.name = f"pca:var={self.var:g}"

    def _prepare(self, s):
        ev, V = s["eigvals"].double(), s["eigvecs"]
        frac = ev.cumsum(0) / ev.sum()
        k = int((frac < self.var).sum().item()) + 1
        if k >= len(ev):
            raise ValueError(f"var={self.var} 时 k={k}，没有剩余子空间")
        return {"mean": s["mean"].float(), "V": V[:, k:].float(), "expected": ev[k:].sum().item(), "k": k}

    def _score(self, h, p):
        return ((h - p["mean"]) @ p["V"]).pow(2).sum(dim=-1) / p["expected"]
