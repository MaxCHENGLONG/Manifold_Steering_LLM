#!/usr/bin/env python3
"""把项目要用的模型和数据集预下载到 Hugging Face 缓存（$HF_HOME）。

只依赖 huggingface_hub，可以直接在登录节点上运行，不占用 GPU：
  python3 -m pip install --user -U huggingface_hub
  python3 tools/download_assets.py --set smoke
  python3 tools/download_assets.py --set qwen data
  python3 tools/download_assets.py --set llama          # 需先在 HF 网页上同意 Llama 许可并登录
  python3 tools/download_assets.py --set corpus         # 逐层统计用的自然语料
  python3 tools/download_assets.py --list-cfierro       # 列出 cfierro 名下的全部数据集

可以重复运行：已下载的文件会被跳过，只补缺的部分。
"""
import argparse
import os
import sys
import time

from huggingface_hub import list_datasets, snapshot_download

MODEL, DATASET = "model", "dataset"

# (类型, 仓库名, 忽略的文件模式)
SETS = {
    # 单卡冒烟测试：vLLM 生成、HF hook、GLP 加载
    "smoke": [
        (MODEL, "Qwen/Qwen2.5-0.5B-Instruct", None),
        (MODEL, "generative-latent-prior/glp-llama1b-d12-multi", ["checkpoints/*"]),
    ],
    # Qwen 线主实验
    "qwen": [
        (MODEL, "Qwen/Qwen2.5-1.5B-Instruct", None),
        (MODEL, "Qwen/Qwen2.5-7B-Instruct", None),
    ],
    # Llama 线（meta-llama 是 gated 模型，需要先同意许可）
    "llama": [
        (MODEL, "meta-llama/Llama-3.2-1B-Instruct", ["original/*"]),
        (MODEL, "meta-llama/Llama-3.1-8B-Instruct", ["original/*"]),
        (MODEL, "generative-latent-prior/glp-llama8b-d6", ["checkpoints/*"]),
    ],
    # weight-steering 的 steering 向量训练数据与评测数据
    "data": [
        (DATASET, "cfierro/sycophancy_eval_answer", None),
        (DATASET, "cfierro/gcd", None),
        (DATASET, "cfierro/pv-prompts-sycophantic", None),
        (DATASET, "cfierro/pv-prompts-non-sycophantic", None),
        (DATASET, "cfierro/pv-prompts-evil", None),
        (DATASET, "cfierro/pv-prompts-non-evil", None),
        (DATASET, "cfierro/pv-prompts-sycophantic_Qwen2.5-1.5B-Instruct", None),
        (DATASET, "cfierro/pv-prompts-non-sycophantic_Qwen2.5-1.5B-Instruct", None),
        # evil 评测（World Affecting）：零样本、CoT、带标签 CoT，以及与 MMLU 同指令的 CoT 变体
        (DATASET, "cfierro/ethical_world_affecting", None),
        (DATASET, "cfierro/ethical_world_affecting_cot", None),
        (DATASET, "cfierro/ethical_world_affecting_cot-tags", None),
        (DATASET, "cfierro/ethical_world_affecting_cot-same-mmlu", None),
        # 能力评测：TinyMMLU（以及 weight-steering 代码里的 mmlu_chat）
        (DATASET, "cfierro/tiny_mmlu_chat", None),
        (DATASET, "cfierro/mmlu_chat", None),
        # 分布内 evil 评测问题（persona vectors 的测试问题，论文 G.4）
        (DATASET, "cfierro/persona-vectors-eval-questions", None),
    ],
    # 逐层统计的自然语料（模块 1）：FineWeb 纯文本，和 UltraChat 的首轮问题（回答由被测模型自己生成）
    "corpus": [
        (DATASET, "HuggingFaceFW/fineweb", None),
        (DATASET, "HuggingFaceH4/ultrachat_200k", None),
    ],
    # 拒答实验（论文 4.3 节，本计划中为可选）；weight-steering 代码读取的是 cfierro 的 chat 版本
    "refusal": [
        (DATASET, "cfierro/DirectHarm4-chat", None),
        (DATASET, "cfierro/GSM-Danger_chat", None),
        (DATASET, "cfierro/gsm8k_instr_chat", None),
        (DATASET, "cfierro/safety-tuning-chat", None),
        (DATASET, "cfierro/alignment_faking_harm_answers_chat", None),
    ],
}

# 只下载仓库里的部分文件：FineWeb 取 sample/10BT 的 1 个分片（2.15 GB，远超 1000 万 token 的需要），
# UltraChat 取 train_sft 的第 1 个分片（0.24 GB，约 7 万条对话）
ALLOW = {
    "HuggingFaceFW/fineweb": ["sample/10BT/000_00000.parquet"],
    "HuggingFaceH4/ultrachat_200k": ["data/train_sft-00000-of-00003-*.parquet"],
}


def dir_size_gb(path):
    """快照目录里的文件是指向 blobs 的符号链接，按真实文件大小统计。"""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.realpath(os.path.join(root, f)))
            except OSError:
                pass
    return total / 1e9


def main():
    ap = argparse.ArgumentParser(description="预下载模型与数据集到 $HF_HOME")
    ap.add_argument("--set", nargs="+", choices=sorted(SETS), default=[], help="要下载的分组")
    ap.add_argument("--list-cfierro", action="store_true", help="列出 cfierro 名下的全部数据集")
    ap.add_argument("--workers", type=int, default=8, help="并行下载线程数")
    args = ap.parse_args()

    hf_home = os.environ.get("HF_HOME")
    print(f"HF_HOME = {hf_home}")
    if not hf_home:
        sys.exit("HF_HOME 没有设置：先 source ~/.bashrc，避免把模型下载到家目录。")

    if args.list_cfierro:
        print("cfierro 名下的数据集：")
        for d in list_datasets(author="cfierro"):
            print("  ", d.id)

    if not args.set:
        if not args.list_cfierro:
            ap.print_help()
        return

    failed = []
    for name in args.set:
        print(f"\n===== 分组 {name} =====")
        for kind, repo, ignore in SETS[name]:
            t0 = time.time()
            try:
                path = snapshot_download(
                    repo_id=repo, repo_type=kind, ignore_patterns=ignore, allow_patterns=ALLOW.get(repo),
                    max_workers=args.workers,
                )
                print(f"[OK]   {repo:<62} {dir_size_gb(path):6.1f} GB  {time.time() - t0:5.0f} s")
            except Exception as e:  # noqa: BLE001  逐个记录失败，不中断其余下载
                msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
                print(f"[FAIL] {repo}: {type(e).__name__}: {msg}")
                if any(k in str(e).lower() for k in ("gated", "401", "403", "access")):
                    print("       ↳ 可能是 gated 模型：先在 HF 网页上同意许可，并确认已登录（token 有读权限）。")
                failed.append(repo)

    print()
    if failed:
        print(f"{len(failed)} 个下载失败：")
        for r in failed:
            print("  ", r)
        sys.exit(1)
    print("全部下载完成。")


if __name__ == "__main__":
    main()
