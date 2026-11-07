"""在 calib 切分的自然激活（reservoir_XX.pt）上校准打分器：每层分数的中位数、四分位距和分位数阈值。

标准化分数 z = (s − 中位数) / 四分位距，不同层、不同打分器可以直接比较；阈值 τ_l 取 z 的分位数。
"""
import json
from pathlib import Path

import torch

QS = (0.9, 0.99, 0.999)


def load_reservoir(stats_dir, layer):
    return torch.load(Path(stats_dir) / f"reservoir_{layer:02d}.pt", map_location="cpu")["samples"]


def score_batched(scorer, x, layer, batch=8192):
    return torch.cat([scorer.score(x[i: i + batch], layer).cpu() for i in range(0, len(x), batch)])


def calibrate(scorer, reservoir_dir, layers=None, qs=QS) -> dict:
    """返回 {"scorer", "qs", "layers": {层: {n, median, iqr, quantiles, z_quantiles}}, "trajectory"}，
    分位数键为字符串。

    各层保留样本逐行对齐（同一批 token，见 stats.Reservoir），所以还能在同一批 token 上算轨迹分数
    （各层 z 的最大值、平均值）的分位数，存在 "trajectory": {"max": {...}, "mean": {...}}。
    """
    layers = scorer.layers if layers is None else layers
    out = {"scorer": scorer.name, "qs": list(qs), "layers": {}}
    z = {}
    for l in layers:
        s = score_batched(scorer, load_reservoir(reservoir_dir, l), l).double()
        out["layers"][l] = _summary(s, qs)
        z[l] = standardize(s, out, l)
        out["layers"][l]["z_quantiles"] = {str(q): (v - out["layers"][l]["median"]) / out["layers"][l]["iqr"]
                                           for q, v in zip(qs, out["layers"][l]["quantiles"].values())}
    if len({len(v) for v in z.values()}) == 1:
        out["trajectory"] = {agg: _summary(trajectory_score(z, agg), qs) for agg in ("max", "mean")}
    return out


def _summary(s, qs):
    q25, med, q75, *rest = torch.quantile(s, torch.tensor([0.25, 0.5, 0.75, *qs], dtype=s.dtype)).tolist()
    return {"n": len(s), "median": med, "iqr": q75 - q25, "quantiles": {str(q): v for q, v in zip(qs, rest)}}


def standardize(s, calib, layer):
    c = calib["layers"][layer]
    return (s - c["median"]) / c["iqr"]


def trajectory_score(per_layer: dict, agg="max"):
    """把各层标准化分数 {层: (N,)} 聚合成每个 token 一个分数。"""
    z = torch.stack([per_layer[l] for l in sorted(per_layer)])
    if agg == "max":
        return z.max(dim=0).values
    if agg == "mean":
        return z.mean(dim=0)
    raise ValueError(f"未知聚合方式 {agg}")


def calib_path(stats_dir, scorer_name):
    return Path(stats_dir) / "calib" / f"{scorer_name.replace(':', '_')}.json"


def save_calibration(calib, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2))


def load_calibration(path):
    calib = json.loads(Path(path).read_text())
    calib["layers"] = {int(l): v for l, v in calib["layers"].items()}
    return calib
