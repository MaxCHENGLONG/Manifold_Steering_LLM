#!/usr/bin/env python3
"""在自然语料上逐层累计激活统计（实施计划模块 1）。

每个角色（text / prompt / response，定义见 manifold_guard/corpus.py）、每一层分别累计：
  fit 切分   → StreamingMoments：n、均值、协方差、特征分解、范数分位数
  calib 切分 → Reservoir：固定数量的随机激活样本（float16），各层逐行对齐
holdout 切分不读取，留给恒等性检查。

输出：
  <out>/<role>/layer_XX.pt       n、mean、cov、eigvals、eigvecs（float32）、norm_quantiles
  <out>/<role>/reservoir_XX.pt   samples（float16）、seen
  <out>/meta.json                参数、各角色与切分的 token 数、镜像、git 哈希、Slurm 作业号、耗时

用法（1 块 GPU；批处理作业里设 HF_HUB_OFFLINE=1，只读预下载的缓存）：
  python scripts/collect_layer_stats.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --chat data/chat_onpolicy/Qwen2.5-1.5B-Instruct.jsonl \
    --n_tokens 2e7 --mix fineweb:0.5,chat:0.5 --out stats/Qwen2.5-1.5B-Instruct
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.corpus import (ROLES, batches, collate, iter_chat, iter_fineweb,  # noqa: E402
                                   split_rows, take_budget)
from manifold_guard.layers import LayerTap, num_layers  # noqa: E402
from manifold_guard.runinfo import run_info  # noqa: E402
from manifold_guard.stats import Reservoir, StreamingMoments  # noqa: E402

FINEWEB = ("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet")


def collect(model, seqs, layers, *, pad_id, batch_tokens, reservoir_k, reservoir_device, seed=0, log_every=0):
    """跑前向并累计统计，返回 moments[role][l]、reservoirs[role][l] 和 counts[role][split]。"""
    device = next(model.parameters()).device
    d = model.config.hidden_size
    moments = {r: {l: StreamingMoments(d, device=device, seed=seed) for l in layers} for r in ROLES}
    reservoirs = {r: {l: Reservoir(reservoir_k, d, device=reservoir_device, seed=seed) for l in layers}
                  for r in ROLES}
    counts = {r: {"fit": 0, "calib": 0} for r in ROLES}
    t0, n_seen = time.time(), 0
    with torch.no_grad(), LayerTap(model, layers) as tap:
        for i, batch in enumerate(batches(seqs, batch_tokens)):
            ids, am, roles = collate(batch, pad_id)
            model(input_ids=ids.to(device), attention_mask=am.to(device), use_cache=False, logits_to_keep=1)
            rows = {sp: split_rows(batch, sp) for sp in ("fit", "calib")}
            for role, rmask in roles.items():
                for sp, acc in (("fit", moments), ("calib", reservoirs)):
                    m = rmask & rows[sp]
                    n = int(m.sum())
                    if n == 0:
                        continue
                    counts[role][sp] += n
                    m = m.to(device)
                    for l in layers:  # 各层按同样的顺序喂同样的 token，Reservoir 选中的位置逐层一致
                        acc[role][l].update(tap.outputs[l][m])
            n_seen += int(am.sum())
            if log_every and (i + 1) % log_every == 0:
                dt = time.time() - t0
                fit = {r: counts[r]["fit"] for r in ROLES}
                print(f"  batch {i + 1}：{n_seen / 1e6:.2f}M token，{n_seen / dt:,.0f} token/s，fit {fit}", flush=True)
    return moments, reservoirs, counts


def save(out, moments, reservoirs, counts):
    for role in ROLES:
        if counts[role]["fit"] < 2:
            print(f"[跳过] {role}：fit 切分只有 {counts[role]['fit']} 个 token")
            continue
        rdir = out / role
        rdir.mkdir(parents=True, exist_ok=True)
        for l, m in moments[role].items():
            r = {k: v.float().cpu() if torch.is_tensor(v) else v for k, v in m.finalize().items()}
            torch.save({"role": role, "layer": l, **r}, rdir / f"layer_{l:02d}.pt")
            res = reservoirs[role][l]
            torch.save({"role": role, "layer": l, "samples": res.data.cpu(), "seen": res.seen},
                       rdir / f"reservoir_{l:02d}.pt")


def parse_mix(s):
    mix = {k: float(v) for k, v in (kv.split(":") for kv in s.split(","))}
    if unknown := set(mix) - {"fineweb", "chat"}:
        raise ValueError(f"--mix 里有未知来源 {unknown}，只支持 fineweb 和 chat")
    return mix


def main():
    ap = argparse.ArgumentParser(description="逐层累计激活统计")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", help="generate_onpolicy_chat.py 的输出；--mix 里有 chat 时必填")
    ap.add_argument("--fineweb", help="FineWeb parquet 路径；默认用 HF 缓存里 sample/10BT 的第 1 个分片")
    ap.add_argument("--n_tokens", type=float, default=2e7, help="fit 切分参与统计的 token 总数")
    ap.add_argument("--mix", default="fineweb:0.5,chat:0.5", help="各来源占 n_tokens 的比例")
    ap.add_argument("--layers", default="all", help="all，或逗号分隔的层号（model.layers 的下标）")
    ap.add_argument("--max_len", type=int, default=2048, help="每条序列截断到的 token 数")
    ap.add_argument("--batch_tokens", type=int, default=32_768, help="每个 batch 的 B × 最大长度上限")
    ap.add_argument("--reservoir", type=int, default=50_000, help="每个角色每层保留的 calib 激活条数")
    ap.add_argument("--reservoir_device", default="cuda", help="保留样本放在哪；7B 全层约 30 GB，显存紧时用 cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "meta.json").exists() and not args.overwrite:
        sys.exit(f"{out} 里已有结果；要重新统计请加 --overwrite")
    mix = parse_mix(args.mix)
    if mix.get("chat") and not args.chat:
        sys.exit("--mix 里有 chat，需要用 --chat 指定 on-policy 对话文件")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    layers = list(range(num_layers(model))) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    budget = {src: frac * args.n_tokens for src, frac in mix.items() if frac > 0}
    sources = []
    if "fineweb" in budget:
        path = args.fineweb or hf_hub_download(FINEWEB[0], FINEWEB[1], repo_type="dataset")
        bos_id = model.config.bos_token_id
        sources.append(take_budget(iter_fineweb(path, tok, args.max_len, bos_id), budget["fineweb"]))
    if "chat" in budget:
        sources.append(take_budget(iter_chat(args.chat, args.max_len), budget["chat"]))

    print(f"模型 {args.model}：{len(layers)} 层，hidden={model.config.hidden_size}；fit 预算 {budget}", flush=True)
    moments, reservoirs, counts = collect(
        model, itertools.chain(*sources), layers, pad_id=pad_id, batch_tokens=args.batch_tokens,
        reservoir_k=args.reservoir, reservoir_device=args.reservoir_device, seed=args.seed,
        log_every=args.log_every,
    )
    t_collect = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    save(out, moments, reservoirs, counts)

    fit = {"fineweb": counts["text"]["fit"], "chat": counts["prompt"]["fit"] + counts["response"]["fit"]}
    short = {src: fit[src] for src in budget if fit[src] < budget[src]}
    if short:
        print(f"[警告] 以下来源的语料不够，没有达到 fit 预算：{short}")
    meta = {
        "args": vars(args), "layers": layers, "counts": counts, "fit_budget": budget, "budget_short": short,
        "collect_s": round(t_collect, 1), "elapsed_s": round(time.time() - t0, 1), **run_info(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"写出 {out}：counts {counts}，耗时 {meta['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()
