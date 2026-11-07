"""离流形打分器。用 make_scorer("gauss:shrink=0.1") 这样的字符串构造，再 .load(统计目录)。"""
from .base import LayerwiseScorer, Scorer, stats_layers
from .gauss import GaussScorer
from .norm import NormScorer
from .pca import PCAScorer

SCORERS = {"norm": NormScorer, "gauss": GaussScorer, "pca": PCAScorer}


def make_scorer(spec):
    """'norm'、'gauss'、'gauss:shrink=0.1'、'pca:var=0.95' → 还没加载统计量的打分器。"""
    name, _, args = spec.partition(":")
    if name not in SCORERS:
        raise ValueError(f"未知打分器 {name}，可选 {sorted(SCORERS)}")
    kw = {k: float(v) for k, v in (a.split("=") for a in args.split(",") if a)}
    return SCORERS[name](**kw)


__all__ = ["Scorer", "LayerwiseScorer", "stats_layers", "make_scorer", "SCORERS",
           "NormScorer", "GaussScorer", "PCAScorer"]
