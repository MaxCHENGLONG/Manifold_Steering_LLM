#!/usr/bin/env python3
"""从 D⁺/D⁻ 算激活 steering 向量：回答 token 上的均值差，格式与 persona_vectors 一致。

输出 [L+1, d] 的 float32 张量：第 0 行是 embedding 层输出，第 l 行是 model.layers[l-1] 的输出。
做法与 persona_vectors 的 generate_vec.py 相同：输入是 "prompt（套聊天模板）+ 回答"，每条对话先在
回答 token 上求均值，再对所有对话求均值，最后 D⁺ 减 D⁻。不同的是数据：这里直接用训练 LoRA 的
同一份 D⁺/D⁻，权重 steering 和激活 steering 用的数据一致。

  python scripts/compute_diffmean_vectors.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --pos cfierro/pv-prompts-sycophantic_Qwen2.5-1.5B-Instruct \
    --neg cfierro/pv-prompts-non-sycophantic_Qwen2.5-1.5B-Instruct \
    --out data/vectors/Qwen2.5-1.5B-Instruct/sycophantic_response_avg_diff.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.corpus import Sequence, batches, collate  # noqa: E402
from manifold_guard.hfdata import load_split  # noqa: E402
from manifold_guard.layers import LayerTap, num_layers  # noqa: E402
from manifold_guard.runinfo import run_info  # noqa: E402


def to_sequences(rows, tok):
    """每条对话的 user 轮套上聊天模板作为 prompt，后接回答（都不加特殊 token）。"""
    seqs = []
    for i, r in enumerate(rows):
        *context, answer = r["messages"]
        assert answer["role"] == "assistant"
        prompt = tok.apply_chat_template(context, add_generation_prompt=True, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer["content"], add_special_tokens=False)["input_ids"]
        seqs.append(Sequence("chat", str(i), p_ids + a_ids, len(p_ids)))
    return seqs


@torch.no_grad()
def response_means(model, seqs, pad_id, batch_tokens):
    """返回 [L+1, d]：各层"每条对话回答 token 均值"的平均（float64 累加）。"""
    L, d = num_layers(model), model.config.hidden_size
    sums = torch.zeros(L + 1, d, dtype=torch.float64, device=model.device)
    emb = {}
    handle = model.get_input_embeddings().register_forward_hook(lambda m, i, o: emb.__setitem__(0, o))
    try:
        with LayerTap(model) as tap:
            for batch in batches(iter(seqs), batch_tokens):
                ids, am, roles = collate(batch, pad_id)
                model(input_ids=ids.to(model.device), attention_mask=am.to(model.device), use_cache=False,
                      logits_to_keep=1)
                m = roles["response"].to(model.device, torch.float64).unsqueeze(-1)  # (B, T, 1)
                count = m.sum(dim=1)                                                # (B, 1)
                outs = [emb[0]] + [tap.outputs[l] for l in range(L)]
                for row, h in enumerate(outs):
                    sums[row] += ((h.double() * m).sum(dim=1) / count).sum(dim=0)
    finally:
        handle.remove()
    return sums / len(seqs)


def main():
    ap = argparse.ArgumentParser(description="从 D⁺/D⁻ 算均值差 steering 向量")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pos", required=True, help="D⁺ 数据集（HF 仓库名，读 train 分片）")
    ap.add_argument("--neg", required=True, help="D⁻ 数据集")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", help="模型精度；persona_vectors 用 float32")
    ap.add_argument("--batch_tokens", type=int, default=16_384)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    means, n = {}, {}
    for name in ("pos", "neg"):
        seqs = to_sequences(load_split(getattr(args, name), args.split), tok)
        n[name] = len(seqs)
        means[name] = response_means(model, seqs, pad_id, args.batch_tokens)
    vector = (means["pos"] - means["neg"]).float().cpu()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, out)
    norms = vector.norm(dim=1)
    meta = {"args": vars(args), "shape": list(vector.shape), "n_conversations": n,
            "norm_per_row": [round(x, 4) for x in norms.tolist()],
            "elapsed_s": round(time.time() - t0, 1), **run_info()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"写出 {out}：形状 {tuple(vector.shape)}，D⁺ {n['pos']} 条、D⁻ {n['neg']} 条")
    print("各行范数（第 0 行为 embedding）：", " ".join(f"{x:.2f}" for x in norms.tolist()))


if __name__ == "__main__":
    main()
