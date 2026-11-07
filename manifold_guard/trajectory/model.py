"""M3 跨层先验：不问"h_l 像不像第 l 层的自然激活"，而问"从 h_{l-1} 走到 h_l 这一步自不自然"。

线性高斯版：h_l = A_l h_{l-1} + b_l + ε，ε ~ N(0, S_l)。A_l、b_l、S_l 由相邻两层的联合均值与协方差
（collect_pair_stats.py 在 fit 切分上累计）闭式求出，不需要训练。分数是残差 ε 在 S_l 下的平方马氏距离除以 d，
与 GaussScorer 同一量纲：自然样本上均值约为 1。

为什么需要它：权重 steering 改的是层与层之间的映射。每层的状态单看仍可能落在自然分布里，
逐层的边缘打分器（norm、gauss、pca）看不出来，但"由 h_{l-1} 预测 h_l"的残差会变大。
"""
from pathlib import Path

import torch


def pair_layers(stats_dir):
    """pairs/ 目录里有哪些转移（层号 l 表示从第 l-1 层到第 l 层）。"""
    return sorted(int(p.stem.split("_")[1]) for p in (Path(stats_dir) / "pairs").glob("pair_*.pt"))


def fit_transition(mean, cov, ridge=1e-3):
    """由 [h_{l-1}; h_l] 的联合均值 (2d,) 与协方差 (2d, 2d) 求 A、b 和残差协方差 S（float64）。

    A = C_yx (C_xx + ridge·tr(C_xx)/d·I)^{-1}。ridge > 0 时 A 不是最小二乘解，S 按 Cov(y − Ax) 的完整公式算。
    """
    mean, cov = mean.double(), cov.double()
    d = len(mean) // 2
    mx, my = mean[:d], mean[d:]
    cxx, cxy, cyy = cov[:d, :d], cov[:d, d:], cov[d:, d:]
    reg = cxx + ridge * cxx.diagonal().mean() * torch.eye(d, dtype=cov.dtype, device=cov.device)
    A = torch.linalg.solve(reg, cxy).T  # reg 对称：A = C_yx reg^{-1} = (reg^{-1} C_xy)^T
    S = cyy - A @ cxy - cxy.T @ A.T + A @ cxx @ A.T
    S = (S + S.T) / 2
    return A, my - A @ mx, S


class LinearGaussianTransition:
    """打分器接口与 manifold_guard.scorers 相同，但 score 需要上一层的激活 prev。"""
    needs_prev = True

    def __init__(self, ridge=1e-3, shrink=0.01):
        self.ridge, self.shrink = float(ridge), float(shrink)
        self.name = f"trans:ridge={self.ridge:g},shrink={self.shrink:g}"

    def load(self, stats_dir, layers=None, device="cuda"):
        self.device = device
        layers = pair_layers(stats_dir) if layers is None else list(layers)
        if not layers:
            raise FileNotFoundError(f"{Path(stats_dir) / 'pairs'} 里没有 pair_*.pt，先跑 collect_pair_stats.py")
        self.params = {}
        for l in layers:
            s = torch.load(Path(stats_dir) / "pairs" / f"pair_{l:02d}.pt", map_location=device)
            A, b, S = fit_transition(s["mean"], s["cov"], self.ridge)
            ev, V = torch.linalg.eigh(S)
            ev = ev.clamp_min(ev.max().item() * 1e-12)
            ev = (1 - self.shrink) * ev + self.shrink * ev.mean()
            self.params[l] = {"At": A.T.float(), "b": b.float(), "W": (V / ev.sqrt()).float()}
        return self

    @property
    def layers(self):
        return sorted(self.params)

    def score(self, h, layer, prev=None):
        if prev is None:
            raise ValueError("转移打分器需要上一层的激活 prev")
        p = self.params[layer]
        r = h.to(self.device, torch.float32) - prev.to(self.device, torch.float32) @ p["At"] - p["b"]
        return (r @ p["W"]).pow(2).sum(dim=-1) / p["W"].shape[1]
