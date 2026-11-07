#!/usr/bin/env python3
"""用 trl + peft 训练 weight steering 的 LoRA（τ⁺ 或 τ⁻），在容器的 train 环境里运行。

超参取自 weight-steering 的 axolotl 配置（axolotl_configs/qwen-2.5-1.5b (GCD)/lora-*-pv_prompts.yml）：
  r=32、α=16、dropout 0、所有线性层，embed_tokens 与 lm_head 全量训练；
  每卡 batch 2 × 梯度累积 4，100 步，warmup 5 步，linear 调度，weight decay 0.01；
  留 5% 做验证，每 20 步评估，保留验证 loss 最好的检查点，连续 2 次不降就停；只在回答部分算 loss。
与原配置的差别：
  - 不做 8-bit 量化，优化器用 adamw_torch_fused（ARM 上没有 bitsandbytes），注意力用 sdpa。
  - embed_tokens 与 lm_head 共享权重的模型（Qwen2.5-1.5B）用 ensure_weight_tying 保持共享，只训一份。
  - 全程 float32，不用 bf16 autocast。参数以 bf16 存时，lr 1e-5 下全量训练的 embedding 每步更新远小于
    bf16 的分辨率，会被舍掉；而 bf16 autocast 加 sdpa 注意力在 batch 里有 padding 时，反向传播会出 NaN
    （2026-11-07 实测：第 2 步起 grad_norm 为 NaN，纯 float32 或 batch 1 都正常）。数据很少，float32 也很快。
  - 梯度检查点默认关闭（只省显存，不影响结果；7B 显存紧时加 --gradient_checkpointing）。
输出 <out>/adapter/（LoRA 与 embedding）和 <out>/meta.json（参数、各步 loss、最佳验证 loss）。

  /opt/venvs/train/bin/python scripts/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset cfierro/pv-prompts-sycophantic_Qwen2.5-1.5B-Instruct --lr 1e-5 \
    --out ckpt/Qwen2.5-1.5B-Instruct/sycophant_lr1e-05
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 直接 python scripts/... 运行时也能导入 manifold_guard
from manifold_guard.hfdata import load_split  # noqa: E402
from manifold_guard.runinfo import run_info  # noqa: E402


def prompt_completion(rows):
    """messages = [..., user, assistant] 拆成对话式的 prompt / completion，trl 只在 completion 上算 loss。"""
    return Dataset.from_list([{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]} for r in rows])


def main():
    ap = argparse.ArgumentParser(description="训练 weight steering 的 LoRA")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="HF 数据集仓库名（读 --split 分片的 messages 字段）")
    ap.add_argument("--split", default="train")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--eval_steps", type=int, default=20)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "meta.json").exists() and not args.overwrite:
        sys.exit(f"{out} 里已有训练结果；要重新训练请加 --overwrite")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    data = prompt_completion(load_split(args.dataset, args.split)).train_test_split(test_size=args.val_frac,
                                                                                   seed=args.seed)

    lora = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear",
                      modules_to_save=["embed_tokens", "lm_head"], ensure_weight_tying=tied,
                      task_type="CAUSAL_LM")
    cfg = SFTConfig(
        output_dir=str(out / "trainer"), max_steps=args.max_steps, learning_rate=args.lr,
        per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4,
        lr_scheduler_type="linear", warmup_steps=5, weight_decay=0.01, optim="adamw_torch_fused",
        bf16=False, gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=5, report_to="none", max_length=4096, seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=data["train"], eval_dataset=data["test"],
                         processing_class=tok, peft_config=lora,
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()

    trainer.save_model(str(out / "adapter"))  # 保存的是验证 loss 最好的检查点
    shutil.rmtree(out / "trainer", ignore_errors=True)  # 中间检查点含优化器状态，体积大，不保留
    state = trainer.state
    meta = {
        "args": vars(args), "tied_embeddings": tied, "n_train": len(data["train"]), "n_val": len(data["test"]),
        "best_eval_loss": state.best_metric, "best_step": int(state.best_model_checkpoint.rsplit("-", 1)[1])
        if state.best_model_checkpoint else None, "global_step": state.global_step,
        "log_history": state.log_history, "elapsed_s": round(time.time() - t0, 1), **run_info(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"写出 {out}：最佳验证 loss {state.best_metric:.4f}（第 {meta['best_step']} 步），"
          f"共 {state.global_step} 步，耗时 {meta['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()
