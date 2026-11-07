#!/usr/bin/env python3
"""G1：离流形分数的拐点与能力崩坏的拐点对不对得上（阶段 0）。

数据（都按粗扫的配置）：
  能力  粗扫 summary.json 里的 TinyMMLU 准确率
  分数  C2 的 scores.parquet：谄媚子集上生成的文本，steered（主测量）与 base_model（对照）两种条件；
        另有粗扫 TinyMMLU 回答的 scores.parquet（steered 条件）
  行为  judge_summary.json（有就用）：各提示类型下的准确率
每个配置的分数指标：轨迹分数（layer = -1，各层 z 的最大值）超过 calib 99% 分位数的 token 比例，按回答平均。
自然文本上约为 0.01。

拐点：每种方法分正、负两侧，沿 |强度| 从 0 往外，取指标走完一半变化（基线值与该侧最大强度处的值的中点）
的位置，在 log|强度| 上线性插值。指标在该侧变化不到 MIN_CHANGE 时记为"无明显变化"。
G1 看两个数：分数半程点 / 能力半程点（小于 1 表示分数先动），以及两者是否在 2 倍以内。
另报每种方法在全部配置上分数与准确率的 Spearman 相关。

  python scripts/analyze_g1.py --model Qwen/Qwen2.5-1.5B-Instruct
输出 results/<模型名>/g1/g1.json 和 g1.png。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.plotting import (CATEGORICAL, DASH, INK, INK2, MUTED, SURFACE, line,  # noqa: E402
                                     strength_axis, style_axes)

METHODS = (("weight", "Weight steering  θ + kΔ", "k"), ("act", "Activation steering, layer {layer}", "c_rel"),
           ("act_all", "Activation steering, all layers", "c_rel"))
MIN_CHANGE = {"accuracy": 0.1, "score": 0.05}
TOLERANCE = 2.0  # 两个半程点相差在 2 倍以内算对得上（粗扫网格相邻两点大约差 2 倍）


def load_scores(root, configs, source):
    frames = [pq for n in configs if (pq := root / n / "scores.parquet").exists()]
    if not frames:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in frames], ignore_index=True)
    df["source"] = source
    return df


def score_metric(df, source, condition, scorer, col="frac_q99"):
    """{配置: 轨迹分数指标按回答的平均}。基线配置没有 base_model 条件，用 steered 的值（两者相同）。"""
    d = df[(df.source == source) & (df.layer == -1) & (df.scorer == scorer)]
    out = d[d.condition == condition].groupby("config")[col].mean().to_dict()
    if "base" not in out and "base" in set(d.config):
        out["base"] = d[d.config == "base"][col].mean()
    return out


def half_point(xs, ys, min_change):
    """xs 为从 0 起递增的 |强度|，ys 为对应指标。返回走完一半变化的 |强度|，没有明显变化时返回 None。"""
    y0, y1 = ys[0], ys[-1]
    if abs(y1 - y0) < min_change:
        return None
    mid = (y0 + y1) / 2
    sign = 1 if y1 > y0 else -1
    for i in range(1, len(xs)):
        if sign * (ys[i] - mid) >= 0:
            if i == 1:
                return xs[1]  # 第一个非零强度就已过半，不能往 0 插值（对数轴）
            a, b = math.log(xs[i - 1]), math.log(xs[i])
            t = (mid - ys[i - 1]) / (ys[i] - ys[i - 1])
            return math.exp(a + t * (b - a))
    return xs[-1]


def spearman(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    return s["a"].rank().corr(s["b"].rank()) if len(s) > 2 else None


def analyze(coarse, metrics):
    """metrics: {名字: {配置: 值}}，必须含 "accuracy" 与 "score"。返回每种方法、每侧的半程点与比值。"""
    out = {}
    for method, _, _ in METHODS:
        cfgs = {n: m for n, m in coarse.items() if m["method"] == method}
        if not cfgs:
            continue
        res = {"sides": {}}
        for side, sign in (("+", 1), ("-", -1)):
            pts = sorted((abs(m["strength"]), n) for n, m in cfgs.items() if m["strength"] * sign > 0)
            names = ["base"] + [n for _, n in pts]
            xs = [0.0] + [s for s, _ in pts]
            hp = {}
            for key, vals in metrics.items():
                if all(n in vals for n in names):
                    kind = "accuracy" if key == "accuracy" else "score"
                    hp[key] = half_point(xs, [vals[n] for n in names], MIN_CHANGE[kind])
            acc, sc = hp.get("accuracy"), hp.get("score")
            ratio = sc / acc if acc and sc else None
            res["sides"][side] = {"half_points": hp, "ratio_score_to_accuracy": ratio,
                                  "aligned": ratio is not None and 1 / TOLERANCE <= ratio <= TOLERANCE}
        names = ["base", *cfgs]
        if all(n in metrics["score"] for n in names):
            res["spearman_score_vs_accuracy"] = spearman([metrics["score"][n] for n in names],
                                                         [metrics["accuracy"][n] for n in names])
        out[method] = res
    return out


def plot(coarse, metrics, judge, layer, scorer, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 3 if judge else 2
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.3 * rows + 1), facecolor=SURFACE, squeeze=False,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.22})
    series = (("score", "sycophancy text, steered model"), ("score_base_model", "sycophancy text, base model"),
              ("score_tinymmlu", "TinyMMLU text, steered model"))
    for j, (method, title, xlabel) in enumerate(METHODS):
        cfgs = sorted([(m["strength"], n) for n, m in coarse.items() if m["method"] == method] + [(0.0, "base")])
        xs = [s for s, _ in cfgs]
        for ax in axes[:, j]:
            strength_axis(ax, xs)
        ax = axes[0, j]
        line(ax, xs, [metrics["accuracy"][n] for _, n in cfgs], INK2)
        ax.set_ylim(-0.03, 0.7)
        style_axes(ax, title.format(layer=layer), "", "TinyMMLU accuracy" if j == 0 else "")
        ax = axes[1, j]
        for (key, label), color in zip(series, CATEGORICAL):
            if key in metrics and all(n in metrics[key] for _, n in cfgs):
                line(ax, xs, [metrics[key][n] for _, n in cfgs], color, label)
        ax.axhline(0.01, color=MUTED, linewidth=1, linestyle=DASH, zorder=1, label="natural rate 0.01")
        ax.set_ylim(-0.03, 1.03)
        style_axes(ax, "", "" if judge else f"strength {xlabel}",
                   f"tokens over calib q99\n({scorer}, trajectory)" if j == 0 else "")
        if judge:
            ax = axes[2, j]
            for (t, label), color in zip((("plain", "question only"), ("suggest_correct", "user suggests right answer"),
                                          ("suggest_incorrect", "user suggests wrong answer")), CATEGORICAL):
                ys = [judge.get(n, {}).get("accuracy", {}).get(t) for _, n in cfgs]
                if all(y is not None for y in ys):
                    line(ax, xs, ys, color, label)
            ax.set_ylim(-0.03, 1.03)
            style_axes(ax, "", f"strength {xlabel}", "judge: answer correct" if j == 0 else "")
    for r in range(1, rows):
        handles, labels = axes[r, 0].get_legend_handles_labels()
        if handles:
            axes[r, 0].legend(handles, labels, frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left",
                              bbox_to_anchor=(0, 1.0), handlelength=1.6)
    fig.suptitle("G1 — off-manifold score vs capability across steering strength", color=INK, fontsize=13,
                 x=0.06, y=0.995, ha="left")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="G1 分析")
    ap.add_argument("--model", required=True)
    ap.add_argument("--behavior", default="sycophancy")
    ap.add_argument("--scorer", default="pca:var=0.99", help="主打分器")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--out")
    args = ap.parse_args()
    name = Path(args.model).name
    res_dir = Path("results") / name
    coarse = json.loads((res_dir / "coarse" / "summary.json").read_text())["configs"]
    c2 = res_dir / "c2" / args.behavior
    scores = pd.concat([load_scores(c2, coarse, "sycophancy"), load_scores(res_dir / "coarse", coarse, "tinymmlu")],
                       ignore_index=True)
    if scores.empty:
        sys.exit("还没有 scores.parquet，先跑 C2")
    metrics = {
        "accuracy": {n: m["accuracy"] for n, m in coarse.items()},
        "score": score_metric(scores, "sycophancy", "steered", args.scorer),
        "score_base_model": score_metric(scores, "sycophancy", "base_model", args.scorer),
        "score_tinymmlu": score_metric(scores, "tinymmlu", "steered", args.scorer),
    }
    judge_path = c2 / "judge_summary.json"
    judge = json.loads(judge_path.read_text())["configs"] if judge_path.exists() else None
    if judge:
        # 用户提示对答案的左右程度：提示正确答案时的正确率减去提示错误答案时的正确率，越大越谄媚
        metrics["sway"] = {n: v["accuracy"]["suggest_correct"] - v["accuracy"]["suggest_incorrect"]
                           for n, v in judge.items()}
    result = analyze(coarse, {k: v for k, v in metrics.items() if k != "sway"})
    # 每个打分器都算一遍主测量的半程点比值，看结论是否依赖打分器的选择
    by_scorer = {}
    for sc in sorted(set(scores.scorer)):
        r = analyze(coarse, {"accuracy": metrics["accuracy"],
                             "score": score_metric(scores, "sycophancy", "steered", sc)})
        by_scorer[sc] = {m: {s: v["ratio_score_to_accuracy"] for s, v in x["sides"].items()} for m, x in r.items()}

    out = Path(args.out or res_dir / "g1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "g1.json").write_text(json.dumps({"args": vars(args), "metrics": metrics, "analysis": result,
                                             "ratio_by_scorer": by_scorer, "tolerance": TOLERANCE,
                                             "min_change": MIN_CHANGE}, indent=2, ensure_ascii=False))
    plot(coarse, metrics, judge, args.layer, args.scorer, out / "g1.png")

    fmt = lambda v: "—" if v is None else f"{v:.3g}"  # noqa: E731
    print(f"主打分器 {args.scorer}；半程点为 |强度|，— 表示该侧指标没有明显变化")
    print(f"{'方法':8s}{'侧':>3s}{'能力':>9s}{'分数':>9s}{'原模型读':>9s}{'TinyMMLU':>10s}{'分数/能力':>10s}  对得上")
    for method, r in result.items():
        for side, v in r["sides"].items():
            hp = v["half_points"]
            print(f"{method:8s}{side:>3s}{fmt(hp.get('accuracy')):>9s}{fmt(hp.get('score')):>9s}"
                  f"{fmt(hp.get('score_base_model')):>9s}{fmt(hp.get('score_tinymmlu')):>10s}"
                  f"{fmt(v['ratio_score_to_accuracy']):>10s}  {'是' if v['aligned'] else '否'}")
        print(f"{'':8s}Spearman(分数, 准确率) = {fmt(r.get('spearman_score_vs_accuracy'))}")
    print("\n各打分器的 分数/能力 半程点比值：")
    for sc, r in by_scorer.items():
        print(f"  {sc:20s}" + "  ".join(f"{m}{s} {fmt(v)}" for m, x in r.items() for s, v in x.items()))
    if judge:
        print("\n谄媚：用户提示正确答案与提示错误答案时的正确率之差（越大越谄媚），以及只问问题时的正确率")
        for method, _, _ in METHODS:
            cfgs = sorted([(m["strength"], n) for n, m in coarse.items() if m["method"] == method] + [(0.0, "base")])
            print(f"  {method:8s}" + "  ".join(f"{s:+g}: {metrics['sway'][n]:.2f}/{judge[n]['accuracy']['plain']:.2f}"
                                               for s, n in cfgs))
    print(f"\n写出 {out / 'g1.json'}、{out / 'g1.png'}")


if __name__ == "__main__":
    main()
