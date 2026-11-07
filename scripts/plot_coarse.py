#!/usr/bin/env python3
"""画粗扫结果：三种方法的能力随强度的变化（阶段 0 C1），并在终端打印表格。

读 sweep_capability.py 写出的 summary.json。每种方法一列（强度的量纲不同，不共用横轴）：
上排是 TinyMMLU 准确率和有效回答率，下排是平均新 token 数。横轴为 symlog，强度 0 是不加 steering 的基线。

  python scripts/plot_coarse.py results/Qwen2.5-1.5B-Instruct/coarse
输出 <结果目录>/capability.png（可用 --out 改）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.plotting import (CATEGORICAL, DASH, INK, INK2, MUTED, SURFACE, line,  # noqa: E402
                                     strength_axis, style_axes)

METHODS = (("weight", "Weight steering  θ + kΔ", "k"),
           ("act", "Activation steering, layer {layer}", "c_rel"),
           ("act_all", "Activation steering, all layers", "c_rel"))


def curves(configs):
    """{方法: [(强度, 指标), ...]}，按强度排序，每种方法都带上强度 0 的基线。"""
    base = configs["base"]
    out = {}
    for method, _, _ in METHODS:
        rows = [(m["strength"], m) for m in configs.values() if m["method"] == method]
        if rows:
            out[method] = sorted(rows + [(0.0, base)], key=lambda r: r[0])
    return out


def print_table(data):
    for method, rows in data.items():
        print(f"\n{method}")
        print(f"  {'强度':>7s}  {'准确率':>6s}  {'有效':>5s}  {'平均 token':>10s}  {'到上限':>6s}")
        for s, m in rows:
            print(f"  {s:>+7g}  {m['accuracy']:>6.2f}  {m['valid_rate']:>5.2f}  {m['mean_new_tokens']:>10.0f}"
                  f"  {m['n_hit_max']:>6d}")


def plot(data, summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange = CATEGORICAL[:2]
    args = summary.get("args", {})
    base = summary["configs"]["base"]
    max_new = args.get("max_new_tokens", 3000)
    fig, axes = plt.subplots(2, len(data), figsize=(5 * len(data), 7.2), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": (1.4, 1), "hspace": 0.35, "wspace": 0.22},
                             squeeze=False)
    titles = {m: t.format(layer=args.get("layer", "?")) for m, t, _ in METHODS}
    xlabels = {m: x for m, _, x in METHODS}
    for j, (method, rows) in enumerate(data.items()):
        xs = [s for s, _ in rows]
        top, bottom = axes[0, j], axes[1, j]
        for ax in (top, bottom):
            strength_axis(ax, xs)
        top.axhline(base["accuracy"], color=MUTED, linewidth=1, linestyle=DASH, zorder=1,
                    label=f"baseline accuracy {base['accuracy']:.2f}")
        for key, color, label in (("accuracy", blue, "accuracy"), ("valid_rate", orange, "valid answer rate")):
            line(top, xs, [m[key] for _, m in rows], color, label)
        top.set_ylim(-0.03, 1.03)
        style_axes(top, titles[method], "", "fraction of 100 questions" if j == 0 else "")
        line(bottom, xs, [m["mean_new_tokens"] for _, m in rows], blue)
        bottom.axhline(max_new, color=MUTED, linewidth=1, linestyle=DASH, zorder=1)
        # 中间是基线附近，曲线在下方，标签放这里不会压到线
        bottom.annotate(f"max_new_tokens {max_new}", xy=(0.5, max_new), xycoords=("axes fraction", "data"),
                        xytext=(0, -3), textcoords="offset points", ha="center", va="top", fontsize=7.5, color=MUTED)
        bottom.set_yscale("log")
        style_axes(bottom, "", f"strength {xlabels[method]}", "mean new tokens" if j == 0 else "")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3, loc="upper left",
               bbox_to_anchor=(0.055, 0.955), handlelength=1.8)
    model = args.get("model", "")
    fig.suptitle(f"{model} — coarse sweep on TinyMMLU (greedy, {base['n']} questions)", color=INK, fontsize=13,
                 x=0.06, y=0.995, ha="left")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="画粗扫的能力曲线")
    ap.add_argument("results_dir")
    ap.add_argument("--out", help="默认 <results_dir>/capability.png")
    args = ap.parse_args()
    root = Path(args.results_dir)
    summary = json.loads((root / "summary.json").read_text())
    data = curves(summary["configs"])
    print_table(data)
    out = Path(args.out) if args.out else root / "capability.png"
    plot(data, summary, out)
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
