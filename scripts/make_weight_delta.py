#!/usr/bin/env python3
"""由 τ⁺、τ⁻ 两个 LoRA 适配器算权重 steering 的方向 Δ = θ⁺ − θ⁻（float32），供扫描使用。

--pos、--neg 各可给多个训练目录（不同学习率）；按 meta.json 里的最佳验证 loss 各选一个，
与实施计划"在 2–3 个学习率里按验证 loss 重新选一次"一致。

  python scripts/make_weight_delta.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --pos ckpt/Qwen2.5-1.5B-Instruct/sycophant_lr* --neg ckpt/Qwen2.5-1.5B-Instruct/non-sycophant_lr* \
    --out ckpt/Qwen2.5-1.5B-Instruct/sycophancy_delta.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.runinfo import run_info  # noqa: E402
from manifold_guard.weight_steering import compute_delta  # noqa: E402


def pick_best(dirs):
    """在多个训练目录里选最佳验证 loss 最低的一个，返回 (目录, 各候选的 loss)。"""
    losses = {}
    for d in map(Path, dirs):
        meta = json.loads((d / "meta.json").read_text())
        losses[str(d)] = meta["best_eval_loss"]
    best = min(losses, key=losses.get)
    return Path(best), losses


def main():
    ap = argparse.ArgumentParser(description="由两个 LoRA 适配器算 Δ = θ⁺ − θ⁻")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pos", nargs="+", required=True, help="τ⁺ 的训练目录（可多个，按验证 loss 选）")
    ap.add_argument("--neg", nargs="+", required=True, help="τ⁻ 的训练目录（可多个，按验证 loss 选）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    pos, pos_losses = pick_best(args.pos)
    neg, neg_losses = pick_best(args.neg)
    print(f"τ⁺ 选 {pos}（候选 {pos_losses}）\nτ⁻ 选 {neg}（候选 {neg_losses}）")
    delta = compute_delta(args.model, str(pos / "adapter"), str(neg / "adapter"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta, out)
    norms = {n: d.norm().item() for n, d in delta.items()}
    meta = {"args": vars(args), "pos": str(pos), "neg": str(neg), "pos_candidates": pos_losses,
            "neg_candidates": neg_losses, "n_params": len(delta), "total_norm": sum(v * v for v in norms.values()) ** 0.5,
            "largest": sorted(norms.items(), key=lambda kv: -kv[1])[:10],
            "elapsed_s": round(time.time() - t0, 1), **run_info()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"写出 {out}：{len(delta)} 个参数有变化，‖Δ‖ = {meta['total_norm']:.4f}，耗时 {meta['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()
