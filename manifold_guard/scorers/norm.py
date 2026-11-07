from .base import LayerwiseScorer


class NormScorer(LayerwiseScorer):
    """|log(‖h‖ / 自然样本的范数中位数)|：范数偏大、偏小都算偏离。

    基线：其他打分器如果不比它好，说明只测到了范数。取两侧，是因为权重 steering 会让范数变小。
    """
    name = "norm"

    def _prepare(self, s):
        return {"median": s["norm_quantiles"][0.5]}

    def _score(self, h, p):
        return (h.norm(dim=-1) / p["median"]).log().abs()
