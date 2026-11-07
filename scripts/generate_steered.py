#!/usr/bin/env python3
"""C2：在粗扫的每个配置下，对谄媚评测子集采样生成回答，保存 token id 供 teacher-forcing 打分和 judge 评测。

子集：cfierro/sycophancy_eval_answer 的 test 切分每 4 行是同一个问题的 4 种提示（只有问题；"我不认为是
正确答案"；"我认为是错误答案"；"我认为是正确答案"）。按问题整组抽 n_groups 组（默认 125 组 = 500 条，固定种子）。
采样与上游 steering_inference.py 一致：temperature 1.0、top_p 1.0、最多 600 个新 token，top_k、
repetition_penalty 沿用模型的 generation_config。配置（方法、强度、绝对系数）取自粗扫的 summary.json，
每个配置用由种子和配置名决定的随机种子，可复现。

  python scripts/generate_steered.py --model Qwen/Qwen2.5-1.5B-Instruct \\
      --coarse results/Qwen2.5-1.5B-Instruct/coarse --delta ckpt/Qwen2.5-1.5B-Instruct/sycophancy/delta.pt \\
      --vector data/vectors/Qwen2.5-1.5B-Instruct/sycophantic_response_avg_diff.pt --methods base,weight
输出 <out>/<配置名>/responses.jsonl 和 meta.json（默认 results/<模型名>/c2/sycophancy/）；已完成的配置跳过。
"""
import argparse
import json
import random
import sys
import time
import zlib
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.hfdata import load_split  # noqa: E402
from manifold_guard.runinfo import run_info  # noqa: E402
from manifold_guard.steering import act_steerer, generate, sampling_config  # noqa: E402
from manifold_guard.weight_steering import WeightSteerer  # noqa: E402

GROUP = 4  # 每个问题 4 种提示，在数据集里连续存放


def sycophancy_subset(rows, n_groups, seed):
    """按问题整组抽样，返回行号（升序）。"""
    if len(rows) % GROUP:
        raise ValueError(f"行数 {len(rows)} 不是 {GROUP} 的倍数")
    for g in range(0, len(rows), GROUP):
        if len({r["base"]["question"] for r in rows[g: g + GROUP]}) != 1:
            raise ValueError(f"第 {g}–{g + GROUP - 1} 行不是同一个问题")
    groups = sorted(random.Random(seed).sample(range(len(rows) // GROUP), n_groups))
    return [g * GROUP + i for g in groups for i in range(GROUP)]


def select_configs(coarse, methods, names=None):
    """粗扫的配置，按方法（base、weight、act、act_all）或配置名筛选，保持 summary.json 里的顺序。"""
    return {n: m for n, m in coarse.items() if m["method"] in methods and (names is None or n in names)}


def steering_context(model, m, delta, vector, layer):
    if m["method"] == "base":
        return nullcontext()
    if m["method"] == "weight":
        return WeightSteerer(model, delta, m["strength"])
    return act_steerer(model, vector, m["coeff"], layer if m["method"] == "act" else 0)


def main():
    ap = argparse.ArgumentParser(description="C2：steering 后在谄媚评测子集上采样生成")
    ap.add_argument("--model", required=True)
    ap.add_argument("--coarse", required=True, help="sweep_capability.py 的输出目录（取配置）")
    ap.add_argument("--delta", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--layer", type=int, default=17, help="激活 steering 层（weight-steering 约定）")
    ap.add_argument("--methods", default="base,weight,act,act_all")
    ap.add_argument("--configs", help="只跑这些配置（逗号分隔）")
    ap.add_argument("--dataset", default="cfierro/sycophancy_eval_answer")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_groups", type=int, default=125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=250)
    ap.add_argument("--out")
    args = ap.parse_args()
    out = Path(args.out or f"results/{Path(args.model).name}/c2/sycophancy")
    out.mkdir(parents=True, exist_ok=True)

    coarse = json.loads((Path(args.coarse) / "summary.json").read_text())["configs"]
    configs = select_configs(coarse, args.methods.split(","), args.configs and args.configs.split(","))
    rows = load_split(args.dataset, args.split)
    subset = sycophancy_subset(rows, args.n_groups, args.seed)
    convs = [rows[i]["messages"] for i in subset]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    gen_cfg = sampling_config(model, args.max_new_tokens, args.temperature, args.top_p)
    delta = {k: v.to("cuda") for k, v in torch.load(args.delta).items()} if any(
        m["method"] == "weight" for m in configs.values()) else None
    vector = torch.load(args.vector)

    for name, m in configs.items():
        cdir = out / name
        if (cdir / "meta.json").exists():
            print(f"[跳过] {name}：已完成", flush=True)
            continue
        seed = (args.seed * 1_000_003 + zlib.crc32(name.encode())) % 2**31
        torch.manual_seed(seed)
        t0 = time.time()
        with steering_context(model, m, delta, vector, args.layer):
            outputs = generate(model, tok, convs, gen_cfg, args.batch_size, return_ids=True)
        cdir.mkdir(exist_ok=True)
        with open(cdir / "responses.jsonl", "w") as f:
            for k, (i, (text, n, p_ids, r_ids)) in enumerate(zip(subset, outputs)):
                r = rows[i]
                f.write(json.dumps({
                    "example": k, "row": i, "template": r["metadata"]["prompt_template"],
                    "dataset": r["base"]["dataset"], "instruction": r["messages"][-1]["content"],
                    "correct_answer": r["base"]["correct_answer"], "incorrect_answer": r["base"]["incorrect_answer"],
                    "response": text, "new_tokens": n, "prompt_ids": p_ids, "response_ids": r_ids,
                }, ensure_ascii=False) + "\n")
        n_new = [n for _, n, _, _ in outputs]
        meta = {"config": name, **m, "seed": seed, "n": len(outputs),
                "mean_new_tokens": sum(n_new) / len(n_new),
                "n_hit_max": sum(n >= args.max_new_tokens for n in n_new),
                "elapsed_s": round(time.time() - t0, 1), "args": vars(args), **run_info()}
        (cdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"{name:16s} 平均 {meta['mean_new_tokens']:.0f} token  到上限 {meta['n_hit_max']:3d}  "
              f"{meta['elapsed_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
