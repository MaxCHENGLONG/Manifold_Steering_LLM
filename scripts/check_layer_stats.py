#!/usr/bin/env python3
"""检查逐层统计的结果，并画每层的特征值谱（模块 1 的验收）。

检查项：
  1. 文件齐全，计数与 meta.json 一致，没有 NaN/Inf，float16 保留样本没有溢出
  2. calib 样本在 fit 高斯下的平方马氏距离 / d：两份数据同分布时均值约为 1
  3. 大范数 token：保留样本里范数超过中位数 10 倍的条数。attention sink 混进统计时会大于 0
     （2026-11-07 第一次跑 Qwen2.5-1.5B 时 text 角色就是这样，原因见 corpus.iter_fineweb）
另外算每层解释 99% 方差所需的主成分数 k99 和参与比 (Σλ)²/Σλ²，供 PCA 打分器选 k。

输出：终端表格、<stats_dir>/check.json、<stats_dir>/spectra.png；有问题时退出码为 1。

  python scripts/check_layer_stats.py stats/Qwen2.5-1.5B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path

import torch

MAHA_RANGE = (0.9, 1.1)
OUTLIER_RATIO = 10


def check_layer(s, r, device):
    ev = s["eigvals"].double().to(device)
    V = s["eigvecs"].double().to(device)
    x = r["samples"].double().to(device)
    z2 = ((x - s["mean"].double().to(device)) @ V) ** 2 / ev.clamp_min(ev[0].item() * 1e-12)
    maha = z2.sum(dim=1) / x.shape[1]
    norms = x.norm(dim=1)
    ratio = norms / norms.median()
    frac = ev.cumsum(0) / ev.sum()
    finite = all(torch.isfinite(s[k]).all() for k in ("mean", "cov", "eigvals", "eigvecs"))
    return {
        "n": s["n"], "seen": r["seen"], "finite": bool(finite and torch.isfinite(r["samples"]).all()),
        "ev_max": ev[0].item(), "ev_min": ev[-1].item(),
        "k99": int((frac < 0.99).sum().item()) + 1,
        "participation_ratio": (ev.sum() ** 2 / (ev ** 2).sum()).item(),
        "norm_median": s["norm_quantiles"][0.5], "norm_q99": s["norm_quantiles"][0.99],
        "max_norm_ratio": ratio.max().item(), "n_outliers": int((ratio > OUTLIER_RATIO).sum().item()),
        "abs_max": r["samples"].abs().max().item(), "maha_mean": maha.mean().item(),
    }


def plot(results, spectra, meta, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    # 参考色板（dataviz skill 的 palette.md）：浅色表面与文字墨色、蓝色有序梯度、前三个类别色
    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    ramp = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
    role_colors = dict(zip(("text", "prompt", "response"), ("#2a78d6", "#eb6834", "#1baf7a")))
    cmap = LinearSegmentedColormap.from_list("layers", ramp)
    layers = meta["layers"]
    norm = Normalize(min(layers), max(layers))

    def style(ax, title, xlabel, ylabel, pad=6):
        ax.set_facecolor(surface)
        ax.set_title(title, color=ink, fontsize=11, loc="left", pad=pad)
        ax.set_xlabel(xlabel, color=ink2, fontsize=9)
        ax.set_ylabel(ylabel, color=ink2, fontsize=9)
        ax.grid(True, color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis)
        ax.tick_params(colors=muted, labelsize=8)

    roles = list(results)
    fig = plt.figure(figsize=(14, 9), facecolor=surface)
    gs = fig.add_gridspec(2, 6, height_ratios=(1.15, 1), hspace=0.38, wspace=0.9)
    top = []
    for i, role in enumerate(roles):
        ax = fig.add_subplot(gs[0, 2 * i: 2 * i + 2], sharey=top[0] if top else None)
        for l in layers:
            ev = spectra[role][l]
            ax.plot(range(1, len(ev) + 1), ev, color=cmap(norm(l)), linewidth=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, f"{role}  (fit n = {results[role][0]['n']:,})", "eigenvalue rank", "eigenvalue" if i == 0 else "")
        top.append(ax)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=top, fraction=0.025, pad=0.02)
    cbar.set_label("layer (model.layers index)", color=ink2, fontsize=9)
    cbar.ax.tick_params(colors=muted, labelsize=8)
    cbar.outline.set_edgecolor(axis)

    d = len(spectra[roles[0]][layers[0]])
    for j, (key, title, ylabel, log) in enumerate((
            ("k99", f"Components for 99% of variance (d = {d})", "k99", False),
            ("participation_ratio", "Participation ratio (Σλ)² / Σλ²", "effective dimension", True))):
        ax = fig.add_subplot(gs[1, 3 * j: 3 * j + 3])
        ends = []
        for role in roles:
            ys = [r[key] for r in results[role]]
            ax.plot(layers, ys, color=role_colors.get(role, ink2), linewidth=2, label=role)
            ends.append((ys[-1], role))
        if log:
            ax.set_yscale("log")
        ax.set_xlim(layers[0], layers[-1] + 3)
        style(ax, title, "layer", ylabel, pad=22)  # 标题和绘图区之间留出一行放图例
        ax.legend(frameon=False, fontsize=8, labelcolor=ink2, ncol=len(roles), loc="lower left",
                  bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.5)
        # 线尾的直接标签：按屏幕坐标从下往上排，相邻至少隔 11 px，避免重叠
        to_px, gap, last = ax.transData, 11, None
        for y, role in sorted(ends):
            py = to_px.transform((layers[-1], y))[1]
            py = py if last is None else max(py, last + gap)
            last = py
            ax.annotate(role, xy=(layers[-1], y), xytext=(layers[-1] + 0.6, to_px.inverted().transform((0, py))[1]),
                        va="center", fontsize=8, color=ink2, annotation_clip=False)

    fig.suptitle(f"{meta['args']['model']} — per-layer activation statistics", color=ink, fontsize=13,
                 x=0.06, ha="left")
    fig.savefig(out_path, dpi=150, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="检查逐层统计并画特征值谱")
    ap.add_argument("stats_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.stats_dir)
    meta = json.loads((root / "meta.json").read_text())
    layers = meta["layers"]
    results, spectra, problems = {}, {}, []
    for role, counts in meta["counts"].items():
        if not (root / role).is_dir():
            continue
        results[role], spectra[role] = [], {}
        for l in layers:
            s = torch.load(root / role / f"layer_{l:02d}.pt")
            r = torch.load(root / role / f"reservoir_{l:02d}.pt")
            res = {"layer": l, **check_layer(s, r, args.device)}
            results[role].append(res)
            spectra[role][l] = s["eigvals"].clamp_min(1e-12).numpy()
            where = f"{role} 第 {l} 层"
            if (res["n"], res["seen"]) != (counts["fit"], counts["calib"]):
                problems.append(f"{where}：计数与 meta.json 不一致")
            if not res["finite"]:
                problems.append(f"{where}：有 NaN/Inf")
            if not MAHA_RANGE[0] <= res["maha_mean"] <= MAHA_RANGE[1]:
                problems.append(f"{where}：calib 的马氏²/d 均值 {res['maha_mean']:.3f} 超出 {MAHA_RANGE}")
            if res["n_outliers"]:
                problems.append(f"{where}：{res['n_outliers']} 条保留样本的范数超过中位数 {OUTLIER_RATIO} 倍"
                                f"（最大 {res['max_norm_ratio']:.0f} 倍）")

        print(f"\n== {role}：fit n={counts['fit']:,}，calib {counts['calib']:,}")
        print(" 层   特征值max  特征值min    k99    参与比  范数中位  范数q99  最大范数/中位  大范数条数  马氏²/d")
        for r in results[role]:
            print(f" {r['layer']:2d}  {r['ev_max']:9.3g}  {r['ev_min']:9.3g}  {r['k99']:5d}  {r['participation_ratio']:8.1f}"
                  f"  {r['norm_median']:8.1f}  {r['norm_q99']:7.1f}  {r['max_norm_ratio']:12.1f}"
                  f"  {r['n_outliers']:9d}  {r['maha_mean']:7.3f}")

    (root / "check.json").write_text(json.dumps({"results": results, "problems": problems}, indent=2,
                                                ensure_ascii=False))
    plot(results, spectra, meta, root / "spectra.png")
    print(f"\n写出 {root / 'check.json'}、{root / 'spectra.png'}")
    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\n全部检查通过")


if __name__ == "__main__":
    main()
