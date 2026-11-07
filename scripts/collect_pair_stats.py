#!/usr/bin/env python3
"""M3 线性高斯版需要的统计：相邻两层 [h_{l-1}; h_l] 的联合均值与协方差（fit 切分）。

与 collect_layer_stats.py 用同一套语料管线，只读 on-policy 对话、只统计一个角色（默认 response）。
token 预算默认 1e7，与逐层统计里 chat 的份额（2e7 × 0.5）相同，所以累计的是同一批 token；
结束时把联合统计里第 l 层那一块与逐层统计的 layer_XX.pt 对比，确认一致。
calib 保留样本不用另存：逐层统计的保留样本各层逐行对齐，第 l-1 层与第 l 层的同一行就是一对。

输出 <stats>/<role>/pairs/pair_XX.pt（XX = l = 1..L-1；n、mean (2d,)、cov (2d, 2d)，float64）和 pairs/meta.json。

  python scripts/collect_pair_stats.py --model Qwen/Qwen2.5-1.5B-Instruct \\
      --chat data/chat_onpolicy/Qwen2.5-1.5B-Instruct.jsonl --stats stats/Qwen2.5-1.5B-Instruct
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.corpus import batches, collate, iter_chat, split_rows, take_budget  # noqa: E402
from manifold_guard.layers import LayerTap, num_layers  # noqa: E402
from manifold_guard.runinfo import run_info  # noqa: E402
from manifold_guard.stats import StreamingMoments  # noqa: E402


def collect_pairs(model, seqs, role, *, pad_id, batch_tokens, seed=0, log_every=0):
    """返回 {l: StreamingMoments(2d)}（l = 1..L-1，累计 [h_{l-1}; h_l]）和 fit 切分的 token 数。"""
    device = next(model.parameters()).device
    d, L = model.config.hidden_size, num_layers(model)
    moments = {l: StreamingMoments(2 * d, device=device, seed=seed) for l in range(1, L)}
    n_fit, t0 = 0, time.time()
    with torch.no_grad(), LayerTap(model, range(L)) as tap:
        for i, batch in enumerate(batches(seqs, batch_tokens)):
            ids, am, roles = collate(batch, pad_id)
            model(input_ids=ids.to(device), attention_mask=am.to(device), use_cache=False, logits_to_keep=1)
            m = (roles[role] & split_rows(batch, "fit")).to(device)
            n = int(m.sum())
            if n == 0:
                continue
            n_fit += n
            for l, mom in moments.items():
                mom.update(torch.cat([tap.outputs[l - 1][m], tap.outputs[l][m]], dim=1))
            if log_every and (i + 1) % log_every == 0:
                print(f"  batch {i + 1}：fit {n_fit:,} 个 {role} token，{n_fit / (time.time() - t0):,.0f} token/s",
                      flush=True)
    return moments, n_fit


def joint(mom):
    cov = mom.m2 / (mom.n - 1)
    return {"n": mom.n, "mean": mom.mean.cpu(), "cov": ((cov + cov.T) / 2).cpu()}


def check_against_layer_stats(pair, layer_file, d):
    """联合统计里第 l 层那一块与逐层统计比：均值与协方差的相对差。"""
    ref = torch.load(layer_file, map_location="cpu")
    mean, cov = pair["mean"][d:], pair["cov"][d:, d:]
    rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()  # noqa: E731
    return {"n_ref": ref["n"], "mean_rel_diff": rel(mean, ref["mean"]), "cov_rel_diff": rel(cov, ref["cov"])}


def main():
    ap = argparse.ArgumentParser(description="M3：相邻两层的联合统计")
    ap.add_argument("--model", required=True)
    ap.add_argument("--chat", required=True, help="generate_onpolicy_chat.py 的输出")
    ap.add_argument("--stats", required=True, help="逐层统计目录（输出写到 <stats>/<role>/pairs/）")
    ap.add_argument("--role", default="response", choices=("prompt", "response"))
    ap.add_argument("--n_tokens", type=float, default=1e7, help="fit 切分的 token 预算，与逐层统计里 chat 的份额相同")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--batch_tokens", type=int, default=32_768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", help="默认 <stats>/<role>/pairs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    role_dir = Path(args.stats) / args.role
    out = Path(args.out or role_dir / "pairs")
    if (out / "meta.json").exists() and not args.overwrite:
        sys.exit(f"{out} 里已有结果；要重新统计请加 --overwrite")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    d = model.config.hidden_size
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id  # 与 collect_layer_stats.py 相同
    seqs = take_budget(iter_chat(args.chat, args.max_len), args.n_tokens)
    moments, n_fit = collect_pairs(model, seqs, args.role, pad_id=pad_id, batch_tokens=args.batch_tokens,
                                   seed=args.seed, log_every=args.log_every)
    t_collect = time.time() - t0

    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    for l, mom in moments.items():
        pair = joint(mom)
        torch.save({"role": args.role, "layer": l, **pair}, out / f"pair_{l:02d}.pt")
        if (role_dir / f"layer_{l:02d}.pt").exists():
            checks[l] = check_against_layer_stats(pair, role_dir / f"layer_{l:02d}.pt", d)
    worst = max((c["cov_rel_diff"] for c in checks.values()), default=None)
    meta = {"args": vars(args), "n_fit": n_fit, "pairs": list(moments), "check_vs_layer_stats": checks,
            "collect_s": round(t_collect, 1), "elapsed_s": round(time.time() - t0, 1), **run_info()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"写出 {out}：{len(moments)} 对相邻层，fit {n_fit:,} 个 {args.role} token，耗时 {meta['elapsed_s']:.0f} s")
    if checks:
        same_n = all(c["n_ref"] == n_fit for c in checks.values())
        print(f"与逐层统计对比：token 数{'一致' if same_n else '不一致'}，协方差最大相对差 {worst:.2e}")


if __name__ == "__main__":
    main()
