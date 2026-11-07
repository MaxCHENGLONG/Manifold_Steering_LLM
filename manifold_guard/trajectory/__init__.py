"""M3 跨层先验。目前只有线性高斯版；条件 flow 版见计划模块 5。"""
from .model import LinearGaussianTransition, fit_transition, pair_layers

__all__ = ["LinearGaussianTransition", "fit_transition", "pair_layers"]
