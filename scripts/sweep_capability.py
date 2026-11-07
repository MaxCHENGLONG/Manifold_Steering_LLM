#!/usr/bin/env python3
"""粗扫：在 TinyMMLU 上测各个 steering 强度下的能力，按规则解析答案，不用 judge（阶段 0 的 C1）。

三种方法：
  weight   θ_base + k·Δ（Δ 来自 make_weight_delta.py）
  act      在第 --layer 层加 c·vector[layer]，只作用在 response 位置
  act_all  全层版本：各层加 c·(vector[l] − vector[l-1])
激活系数用相对大小 c_rel 给出（见 manifold_guard.steering.absolute_coeff），换算用逐层统计里 response
角色的范数中位数。三种方法共用一个不加 steering 的基线（base）。两种方法都用 HF 贪心生成，最多 3000 个
新 token（与上游 MMLU 评测一致），按 "Final Answer: X" 解析。

输出 <out>/<配置名>/responses.jsonl 与 metrics.json，以及汇总 <out>/summary.json（每完成一个配置就更新）。
已完成的配置会跳过，作业中断后重跑即可续上。

  python scripts/sweep_capability.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --delta ckpt/Qwen2.5-1.5B-Instruct/sycophancy/delta.pt \
    --vector data/vectors/Qwen2.5-1.5B-Instruct/sycophantic_response_avg_diff.pt \
    --stats stats/Qwen2.5-1.5B-Instruct --out results/Qwen2.5-1.5B-Instruct/coarse
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.hfdata import load_split  # noqa: E402
from manifold_guard.runinfo import run_info  # noqa: E402
from manifold_guard.steering import absolute_coeff, act_steerer, generate, greedy_config, parse_choice  # noqa: E402
from manifold_guard.weight_steering import WeightSteerer  # noqa: E402

LETTERS = "ABCD"


def floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def configs(args):
    """返回 [(配置名, 方法, 强度)]，基线在最前。"""
    out = [("base", "base", 0.0)]
    if "weight" in args.methods:
        out += [(f"weight_k{k:+g}", "weight", k) for k in args.weight_k if k != 0]
    for m in ("act", "act_all"):
        if m in args.methods:
            out += [(f"{m}_c{c:+g}", m, c) for c in args.act_c if c != 0]
    return out


def main():
    ap = argparse.ArgumentParser(description="TinyMMLU 粗扫")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", help="make_weight_delta.py 的输出；methods 含 weight 时必填")
    ap.add_argument("--vector", help="compute_diffmean_vectors.py 的输出；methods 含 act/act_all 时必填")
    ap.add_argument("--stats", help="逐层统计目录（换算相对系数用）；methods 含 act/act_all 时必填")
    ap.add_argument("--methods", default="weight,act,act_all")
    ap.add_argument("--layer", type=int, default=17, help="激活 steering 层（weight-steering 约定，1.5B 默认 17）")
    ap.add_argument("--weight_k", type=floats, default=floats("2,-2,5,-5,10,-10,20,-20,30,-30"))
    # 冒烟测试里 c_rel=1 时第 17 层与全层版本都已完全崩坏，所以网格从 0.05 开始
    ap.add_argument("--act_c", type=floats, default=floats("0.05,-0.05,0.1,-0.1,0.25,-0.25,0.5,-0.5,1,-1,2,-2"))
    ap.add_argument("--dataset", default="cfierro/tiny_mmlu_chat")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max_new_tokens", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=100)
    args = ap.parse_args()
    args.methods = args.methods.split(",")
    uses_act = bool({"act", "act_all"} & set(args.methods))
    if "weight" in args.methods and not args.delta:
        sys.exit("methods 含 weight，需要 --delta")
    if uses_act and not (args.vector and args.stats):
        sys.exit("methods 含 act/act_all，需要 --vector 和 --stats")

    rows = load_split(args.dataset, args.split)[: args.limit]
    convs = [r["messages"] for r in rows]
    answers = [LETTERS[r["data"]["answer"]] for r in rows]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    gen_cfg = greedy_config(model, args.max_new_tokens)
    delta = {n: d.to("cuda") for n, d in torch.load(args.delta).items()} if "weight" in args.methods else None
    if uses_act:
        vector = torch.load(args.vector)
        ref = torch.load(Path(args.stats) / "response" / f"layer_{args.layer - 1:02d}.pt")
        median_norm = ref["norm_quantiles"][0.5]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"configs": {}}
    summary.update(args={k: v for k, v in vars(args).items()}, **run_info())

    for name, method, strength in configs(args):
        cdir = out / name
        if (cdir / "metrics.json").exists():
            print(f"[跳过] {name}：已完成", flush=True)
            continue
        if method == "base":
            ctx, coeff = nullcontext(), 0.0
        elif method == "weight":
            ctx, coeff = WeightSteerer(model, delta, strength), strength
        else:
            coeff = absolute_coeff(strength, vector, args.layer, median_norm)
            ctx = act_steerer(model, vector, coeff, args.layer if method == "act" else 0)

        t0 = time.time()
        with ctx:
            outputs = generate(model, tok, convs, gen_cfg, args.batch_size)
        preds = [parse_choice(text) for text, _ in outputs]
        n = len(rows)
        metrics = {
            "method": method, "strength": strength, "coeff": coeff, "n": n,
            "accuracy": sum(p == a for p, a in zip(preds, answers)) / n,
            "valid_rate": sum(p is not None for p in preds) / n,
            "mean_new_tokens": sum(k for _, k in outputs) / n,
            "n_hit_max": sum(k >= args.max_new_tokens for _, k in outputs),
            "elapsed_s": round(time.time() - t0, 1),
        }
        cdir.mkdir(exist_ok=True)
        with open(cdir / "responses.jsonl", "w") as f:
            for r, (text, k), p, a in zip(rows, outputs, preds, answers):
                f.write(json.dumps({"question": r["messages"][-1]["content"], "response": text, "new_tokens": k,
                                    "pred": p, "answer": a}, ensure_ascii=False) + "\n")
        (cdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        summary["configs"][name] = metrics
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"{name:16s} 准确率 {metrics['accuracy']:.2f}  有效回答 {metrics['valid_rate']:.2f}  "
              f"平均 {metrics['mean_new_tokens']:.0f} token  到上限 {metrics['n_hit_max']:3d}  "
              f"{metrics['elapsed_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
