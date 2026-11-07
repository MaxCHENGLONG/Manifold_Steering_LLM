#!/usr/bin/env python3
"""A2：校准打分器，并检验它们能否把 steering 后的激活和自然激活分开（每层、每个打分器的 AUROC）。

先在 calib 保留样本上校准每个打分器（写到 <stats>/<角色>/calib/）。负样本有两类：
  合成  random：保留样本加随机方向的扰动，长度取 c_rel × 该层范数中位数，与相对系数 c_rel 的激活
        steering 一样长；shuffle：保留样本逐行打乱维度，范数和取值分布不变。正样本是扰动前的同一批样本。
  真实  把 holdout 切分的 on-policy 对话一次喂进被 steering 的模型，取 response token 的各层激活。
        激活 steering 只加在生成时会加的位置（steer_mask），权重 steering 用 WeightSteerer。
        正样本是同一批 token 不加 steering 时的激活。强度、绝对系数和 TinyMMLU 准确率取自粗扫的 summary.json。
轨迹分数是各层标准化分数 z 的最大值（另报平均值）。

验收（计划模块 2）：在 TinyMMLU 准确率比基线低 10% 以上的配置上，至少一个便宜打分器（gauss、pca）的
轨迹 AUROC ≥ 0.9；norm 明显更差，否则说明其他打分器只测到了范数。

  python scripts/validate_scorers.py --model Qwen/Qwen2.5-1.5B-Instruct --stats stats/Qwen2.5-1.5B-Instruct \\
      --coarse results/Qwen2.5-1.5B-Instruct/coarse --chat data/chat_onpolicy/Qwen2.5-1.5B-Instruct.jsonl \\
      --delta ckpt/Qwen2.5-1.5B-Instruct/sycophancy/delta.pt \\
      --vector data/vectors/Qwen2.5-1.5B-Instruct/sycophantic_response_avg_diff.pt
输出 <out>/validate.json 和 <out>/validate.png（默认 results/<模型名>/scorers/）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.calibrate import (calib_path, calibrate, load_reservoir, save_calibration,  # noqa: E402
                                      score_batched, standardize, trajectory_score)
from manifold_guard.corpus import batches, collate, iter_chat  # noqa: E402
from manifold_guard.layers import LayerTap  # noqa: E402
from manifold_guard.metrics import auroc  # noqa: E402
from manifold_guard.runinfo import run_info  # noqa: E402
from manifold_guard.scorers import NormScorer, make_scorer  # noqa: E402
from manifold_guard.steering import MaskedSteerer, steer_mask  # noqa: E402
from manifold_guard.weight_steering import WeightSteerer  # noqa: E402

CHEAP_AUROC = 0.9   # 便宜打分器要达到的轨迹 AUROC
NORM_GAP = 0.05     # 最好的便宜打分器平均要比 norm 高出这么多，才算"norm 明显更差"
CAPABILITY_DROP = 0.1


def synthetic(scorers, role_dir, layers, c_grid, n, seed):
    """{"random": {c: {打分器: {层: AUROC}}}, "shuffle": {打分器: {层: AUROC}}}"""
    medians = NormScorer().load(role_dir, layers, device="cpu").params
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = {"random": {c: {s.name: {} for s in scorers} for c in c_grid}, "shuffle": {s.name: {} for s in scorers}}
    for l in layers:
        x = load_reservoir(role_dir, l)[:n].to("cuda", torch.float32)
        u = torch.randn(x.shape, generator=g, device="cuda")
        u /= u.norm(dim=-1, keepdim=True)
        shuffled = x.gather(1, torch.rand(x.shape, generator=g, device="cuda").argsort(dim=1))
        for s in scorers:
            base = score_batched(s, x, l)
            for c in c_grid:
                out["random"][c][s.name][l] = auroc(base, score_batched(s, x + c * medians[l]["median"] * u, l))
            out["shuffle"][s.name][l] = auroc(base, score_batched(s, shuffled, l))
    return out


def collect_scores(model, data, scorers, layers, steerer=None):
    """把 data 里的批次喂进模型，返回 {打分器: {层: 全部 response token 的分数 (N,)}}。"""
    scores = {s.name: {l: [] for l in layers} for s in scorers}
    with torch.no_grad(), LayerTap(model, layers) as tap:
        for ids, am, resp, smask in data:
            if steerer is not None:
                steerer.mask = smask
            model(input_ids=ids, attention_mask=am)
            for l in layers:
                h = tap.outputs[l][resp]
                for s in scorers:
                    scores[s.name][l].append(s.score(h, l))
    return {name: {l: torch.cat(v) for l, v in per.items()} for name, per in scores.items()}


def compare(base, steered, calibs, layers):
    """每个打分器：各层 AUROC、各层平均 z，以及轨迹分数（z 的最大值与平均值）的 AUROC。"""
    out = {}
    for name, per in steered.items():
        zb = {l: standardize(base[name][l], calibs[name], l) for l in layers}
        zs = {l: standardize(per[l], calibs[name], l) for l in layers}
        out[name] = {
            "layers": {l: auroc(base[name][l], per[l]) for l in layers},
            "z_mean": {l: zs[l].mean().item() for l in layers},
            "trajectory": {agg: auroc(trajectory_score(zb, agg), trajectory_score(zs, agg)) for agg in ("max", "mean")},
        }
    return out


def acceptance(coarse, steering, names):
    base_acc = coarse["base"]["accuracy"]
    cheap = [n for n in names if not n.startswith("norm")]
    rows = []
    for cfg, m in coarse.items():
        if cfg == "base" or m["accuracy"] >= (1 - CAPABILITY_DROP) * base_acc:
            continue
        traj = {n: steering[cfg]["scores"][n]["trajectory"]["max"] for n in names}
        best = max(cheap, key=traj.get)
        rows.append({"config": cfg, "accuracy": m["accuracy"], "best": best, "best_auroc": traj[best],
                     "norm_auroc": traj.get("norm")})
    detect = all(r["best_auroc"] >= CHEAP_AUROC for r in rows)
    # 只用一个打分器也要能覆盖全部能力下降的配置：列出做得到的那些
    single = [n for n in cheap if all(steering[r["config"]]["scores"][n]["trajectory"]["max"] >= CHEAP_AUROC
                                      for r in rows)]
    gaps = [r["best_auroc"] - r["norm_auroc"] for r in rows if r["norm_auroc"] is not None]
    gap = sum(gaps) / len(gaps) if gaps else None
    return {"degraded": rows, "detect_pass": detect, "single_scorer_pass": single,
            "mean_gap_over_norm": gap, "norm_worse_pass": gap is not None and gap >= NORM_GAP,
            "thresholds": {"auroc": CHEAP_AUROC, "norm_gap": NORM_GAP, "capability_drop": CAPABILITY_DROP}}


def plot(result, coarse, names, layer, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from manifold_guard.plotting import CATEGORICAL, DASH, INK, INK2, MUTED, SURFACE, line, strength_axis, style_axes

    methods = (("weight", "Weight steering  θ + kΔ", "k"), ("act", f"Activation steering, layer {layer}", "c_rel"),
               ("act_all", "Activation steering, all layers", "c_rel"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), facecolor=SURFACE, squeeze=False,
                             gridspec_kw={"height_ratios": (1.4, 1), "hspace": 0.35, "wspace": 0.22})
    for j, (method, title, xlabel) in enumerate(methods):
        cfgs = sorted([(m["strength"], c) for c, m in coarse.items() if m["method"] == method] + [(0.0, "base")])
        xs = [s for s, _ in cfgs]
        top, bottom = axes[0, j], axes[1, j]
        for ax in (top, bottom):
            strength_axis(ax, xs)
        top.axhline(CHEAP_AUROC, color=MUTED, linewidth=1, linestyle=DASH, zorder=1, label=f"AUROC {CHEAP_AUROC}")
        for name, color in zip(names, CATEGORICAL):
            ys = [0.5 if c == "base" else result["steering"][c]["scores"][name]["trajectory"]["max"] for _, c in cfgs]
            line(top, xs, ys, color, name)
        top.set_ylim(0.45, 1.02)
        style_axes(top, title, "", "trajectory AUROC (max over layers)" if j == 0 else "")
        line(bottom, xs, [coarse[c]["accuracy"] for _, c in cfgs], INK2)
        bottom.set_ylim(-0.03, 0.7)
        style_axes(bottom, "", f"strength {xlabel}", "TinyMMLU accuracy" if j == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK2, ncol=len(labels), loc="upper left",
               bbox_to_anchor=(0.055, 0.955), handlelength=1.8)
    n_tok = result["steering"]["base"]["n_tokens"]
    fig.suptitle(f"{result['args']['model']} — separating steered from natural response activations "
                 f"({n_tok:,} tokens)", color=INK, fontsize=13, x=0.06, y=0.995, ha="left")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="校准并验证离流形打分器")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats", required=True, help="模型的逐层统计目录（含各角色子目录）")
    ap.add_argument("--role", default="response")
    ap.add_argument("--coarse", required=True, help="sweep_capability.py 的输出目录")
    ap.add_argument("--chat", required=True, help="generate_onpolicy_chat.py 的输出，取 holdout 切分")
    ap.add_argument("--delta", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--layer", type=int, default=17, help="激活 steering 层（weight-steering 约定）")
    ap.add_argument("--scorers", default="norm,gauss,pca:var=0.9,pca:var=0.95,pca:var=0.99")
    ap.add_argument("--n_seqs", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--batch_tokens", type=int, default=32768)
    ap.add_argument("--n_synth", type=int, default=20000, help="合成负样本用的保留样本数（每层）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    t0 = time.time()
    name = Path(args.model).name
    out = Path(args.out or f"results/{name}/scorers")
    out.mkdir(parents=True, exist_ok=True)
    role_dir = Path(args.stats) / args.role

    scorers = [make_scorer(s).load(role_dir) for s in args.scorers.split(",")]
    names = [s.name for s in scorers]
    layers = scorers[0].layers
    calibs = {}
    for s in scorers:
        calibs[s.name] = calibrate(s, role_dir)
        save_calibration(calibs[s.name], calib_path(role_dir, s.name))
    print(f"校准完成：{len(scorers)} 个打分器 × {len(layers)} 层，写到 {role_dir / 'calib'}（{time.time() - t0:.0f} s）")

    coarse = json.loads((Path(args.coarse) / "summary.json").read_text())["configs"]
    c_grid = sorted({abs(m["strength"]) for m in coarse.values() if m["method"].startswith("act")})
    synth = synthetic(scorers, role_dir, layers, c_grid, args.n_synth, args.seed)
    print(f"\n合成负样本（各层 AUROC 的中位数）{'':4s}" + "".join(f"{n:>16s}" for n in names))
    for c in c_grid:
        print(f"  random c_rel={c:<6g}" + "".join(
            f"{torch.tensor(list(synth['random'][c][n].values())).median().item():16.3f}" for n in names))
    print("  shuffle          " + "".join(
        f"{torch.tensor(list(synth['shuffle'][n].values())).median().item():16.3f}" for n in names))

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    seqs = [s for s in iter_chat(args.chat, args.max_len) if s.split == "holdout"][: args.n_seqs]
    data = []
    for b in batches(seqs, args.batch_tokens):
        ids, am, roles = collate(b, pad_id)
        data.append((ids.cuda(), am.cuda(), roles["response"].cuda(), steer_mask(am, [s.prompt_len for s in b]).cuda()))
    delta = {k: v.to("cuda") for k, v in torch.load(args.delta).items()}
    vector = torch.load(args.vector)

    base = collect_scores(model, data, scorers, layers)
    n_tok = len(base[names[0]][layers[0]])
    steering = {"base": {"n_tokens": n_tok}}
    print(f"\n{len(seqs)} 条 holdout 对话，{n_tok:,} 个 response token")
    print(f"真实 steering（轨迹 AUROC，max 聚合）{'':10s}准确率" + "".join(f"{n:>16s}" for n in names))
    for cfg, m in coarse.items():
        if cfg == "base":
            continue
        if m["method"] == "weight":
            ctx, steerer = WeightSteerer(model, delta, m["strength"]), None
        else:
            ctx = steerer = MaskedSteerer(model, vector, m["coeff"], args.layer if m["method"] == "act" else 0)
        with ctx:
            per = collect_scores(model, data, scorers, layers, steerer)
        steering[cfg] = {"method": m["method"], "strength": m["strength"], "coeff": m["coeff"],
                         "accuracy": m["accuracy"], "n_tokens": n_tok, "scores": compare(base, per, calibs, layers)}
        print(f"  {cfg:32s}{m['accuracy']:6.2f}" + "".join(
            f"{steering[cfg]['scores'][n]['trajectory']['max']:16.3f}" for n in names), flush=True)

    result = {"args": vars(args), **run_info(), "scorers": names, "layers": layers,
              "calibration": {n: str(calib_path(role_dir, n)) for n in names},
              "synthetic": synth, "steering": steering, "acceptance": acceptance(coarse, steering, names),
              "elapsed_s": round(time.time() - t0, 1)}
    (out / "validate.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    plot(result, coarse, names, args.layer, out / "validate.png")

    acc = result["acceptance"]
    print(f"\n能力下降 {CAPABILITY_DROP:.0%} 以上的配置 {len(acc['degraded'])} 个：")
    for r in acc["degraded"]:
        norm = f"  norm {r['norm_auroc']:.3f}" if r["norm_auroc"] is not None else ""
        print(f"  {r['config']:16s} 准确率 {r['accuracy']:.2f}  最好 {r['best']} {r['best_auroc']:.3f}{norm}")
    print(f"每个配置都有便宜打分器 AUROC ≥ {CHEAP_AUROC}：{'通过' if acc['detect_pass'] else '未通过'}")
    print(f"单个打分器覆盖全部配置：{acc['single_scorer_pass'] or '没有'}")
    if acc["mean_gap_over_norm"] is not None:
        print(f"比 norm 平均高 {acc['mean_gap_over_norm']:.3f}（要求 ≥ {NORM_GAP}）："
              f"{'通过' if acc['norm_worse_pass'] else '未通过'}")
    print(f"\n写出 {out / 'validate.json'}、{out / 'validate.png'}（共 {result['elapsed_s']:.0f} s）")


if __name__ == "__main__":
    main()
