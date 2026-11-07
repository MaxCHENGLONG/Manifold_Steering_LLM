#!/usr/bin/env python3
"""用 vLLM 让被测模型回答 UltraChat 的首轮问题，生成逐层统计用的 on-policy 对话。

生成设置与 weight-steering 的评测（inference_and_eval.py 的默认值）一致：只放一条 user 消息，
不加 system prompt（Qwen2.5 的模板会自动补上默认 system prompt）；temperature 1.0、top_p 1.0、
最多 600 个新 token；top_k、repetition_penalty 两边都沿用模型的 generation_config。
输出 jsonl，每行一条：id、prompt、response、finish_reason、prompt_token_ids、response_token_ids。
保存生成时的 token id，统计时直接使用，不重新分词。

用法（1 块 GPU；批处理作业里设 HF_HUB_OFFLINE=1）：
  python scripts/generate_onpolicy_chat.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --out data/chat_onpolicy/Qwen2.5-1.5B-Instruct.jsonl
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import vllm
from huggingface_hub import hf_hub_download
from vllm import LLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.runinfo import run_info  # noqa: E402

ULTRACHAT = ("HuggingFaceH4/ultrachat_200k", "data/train_sft-00000-of-00003-a3ecf92756993583.parquet")


def load_prompts(seed):
    """UltraChat train_sft 第 1 个分片里的 (prompt_id, 首轮问题)，按 seed 打乱。"""
    path = hf_hub_download(ULTRACHAT[0], ULTRACHAT[1], repo_type="dataset")
    t = pq.read_table(path, columns=["prompt_id", "prompt"]).to_pydict()
    rows = list(zip(t["prompt_id"], t["prompt"]))
    random.Random(seed).shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description="生成 on-policy 对话")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="输出的 jsonl 路径")
    ap.add_argument("--n_prompts", type=int, default=30_000)
    ap.add_argument("--max_prompt_tokens", type=int, default=1024, help="套上聊天模板后超过这个长度的问题跳过")
    # 以下三项的默认值与 weight-steering 的 inference_and_eval.py 一致
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} 已存在；要重新生成请加 --overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    llm = LLM(model=args.model, seed=args.seed, max_model_len=args.max_prompt_tokens + args.max_new_tokens)
    tok = llm.get_tokenizer()

    kept, convs = [], []
    for pid, prompt in load_prompts(args.seed):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if len(tok(text, add_special_tokens=False)["input_ids"]) <= args.max_prompt_tokens:
            kept.append((pid, prompt))
            convs.append(msgs)
            if len(convs) == args.n_prompts:
                break
    print(f"选出 {len(convs)} 个问题（目标 {args.n_prompts}）")

    params = llm.get_default_sampling_params()  # 带上 generation_config 里的 top_k、repetition_penalty
    params.temperature, params.top_p, params.max_tokens = args.temperature, args.top_p, args.max_new_tokens
    outputs = llm.chat(convs, params, use_tqdm=True)

    tmp = out.with_name(out.name + ".tmp")  # 写完再改名，作业中断时不会留下半个文件
    n_tokens = n_truncated = 0
    with open(tmp, "w") as f:
        for (pid, prompt), o in zip(kept, outputs):
            c = o.outputs[0]
            n_tokens += len(o.prompt_token_ids) + len(c.token_ids)
            n_truncated += c.finish_reason == "length"
            f.write(json.dumps({
                "id": pid, "prompt": prompt, "response": c.text, "finish_reason": c.finish_reason,
                "prompt_token_ids": list(o.prompt_token_ids), "response_token_ids": list(c.token_ids),
            }, ensure_ascii=False) + "\n")
    tmp.rename(out)

    meta = {
        "args": vars(args), "n_conversations": len(convs), "n_tokens": n_tokens,
        "n_truncated": n_truncated, "sampling_params": repr(params),
        "elapsed_s": round(time.time() - t0, 1), "vllm": vllm.__version__, **run_info(),
    }
    out.with_name(out.name + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"写出 {out}：{len(convs)} 条对话，{n_tokens} 个 token，{n_truncated} 条因长度截断，"
          f"耗时 {meta['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()
