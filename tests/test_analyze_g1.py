import math

import pytest

from scripts.analyze_g1 import analyze, half_point


def test_half_point_interpolates_in_log_strength():
    # 0.5 → 0.1，中点 0.3 落在 |强度| 2 与 8 之间、正好一半处 → log 中点 4
    assert half_point([0, 1, 2, 8], [0.5, 0.5, 0.5, 0.1], 0.1) == pytest.approx(4.0)
    assert half_point([0, 1, 2], [0.5, 0.48, 0.49], 0.1) is None          # 没有明显变化
    assert half_point([0, 1, 2], [0.0, 0.9, 1.0], 0.05) == 1               # 第一个点就过半
    assert half_point([0, 1, 4], [0.0, 0.25, 1.0], 0.05) == pytest.approx(math.exp(math.log(4) / 3))


def test_analyze_ratio_and_alignment():
    coarse = {"base": {"method": "base", "strength": 0.0}}
    acc, score = {"base": 0.5}, {"base": 0.01}
    for k, a, s in ((1, 0.5, 0.02), (2, 0.5, 0.9), (4, 0.0, 1.0)):
        for sign in (1, -1):
            n = f"weight_k{sign * k:+g}"
            coarse[n] = {"method": "weight", "strength": float(sign * k)}
            acc[n], score[n] = a, s if sign > 0 else 0.01
    r = analyze(coarse, {"accuracy": acc, "score": score})["weight"]
    plus, minus = r["sides"]["+"], r["sides"]["-"]
    assert plus["half_points"]["accuracy"] == pytest.approx(math.sqrt(8))  # log 中点 between 2 and 4
    assert plus["ratio_score_to_accuracy"] < 1 and plus["aligned"]
    assert minus["half_points"]["score"] is None and not minus["aligned"]
    assert r["spearman_score_vs_accuracy"] < 0
