"""离流形打分器。用 make_scorer("gauss:shrink=0.1") 这样的字符串构造，再 .load(统计目录)。"""
from .base import LayerwiseScorer, Scorer, stats_layers
from .gauss import GaussScorer
from .norm import NormScorer
from .pca import PCAScorer
from ..trajectory.model import LinearGaussianTransition

SCORERS = {"norm": NormScorer, "gauss": GaussScorer, "pca": PCAScorer, "trans": LinearGaussianTransition}


def make_scorer(spec):
    """'norm'、'gauss'、'gauss:shrink=0.1'、'pca:var=0.95'、'trans:ridge=1e-3' → 还没加载统计量的打分器。"""
    name, _, args = spec.partition(":")
    if name not in SCORERS:
        raise ValueError(f"未知打分器 {name}，可选 {sorted(SCORERS)}")
    kw = {k: float(v) for k, v in (a.split("=") for a in args.split(",") if a)}
    return SCORERS[name](**kw)


def required_layers(scorers):
    """前向时要取出的层：各打分器用到的层，转移打分器（needs_prev）还要上一层。"""
    need = {l for sc in scorers for l in sc.layers}
    return sorted(need | {l - 1 for sc in scorers if sc.needs_prev for l in sc.layers})


__all__ = ["Scorer", "LayerwiseScorer", "stats_layers", "make_scorer", "required_layers", "SCORERS",
           "NormScorer", "GaussScorer", "PCAScorer", "LinearGaussianTransition"]
