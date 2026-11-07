#!/bin/bash
# 训练 weight steering 的 LoRA：τ⁺、τ⁻ 各 3 个学习率，作业数组，每个任务 1 块 GH200，用容器的 train 环境。
# 任务 0–2 训 τ⁺、3–5 训 τ⁻，学习率依次为 5e-6、1e-5、2e-5。提交：
#   cd /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/Manifold_Steering_LLM && \
#   sbatch --array=0-5 slurm/train_lora.sbatch Qwen/Qwen2.5-1.5B-Instruct sycophancy \
#       cfierro/pv-prompts-sycophantic_Qwen2.5-1.5B-Instruct cfierro/pv-prompts-non-sycophantic_Qwen2.5-1.5B-Instruct
# 输出 ckpt/<模型名>/<行为>/{pos,neg}_lr<学习率>/（adapter/ 与 meta.json）
#SBATCH -A naiss2026-4-1521-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 00:45:00
#SBATCH -J train-lora
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/logs/%x_%A_%a.out

set -euo pipefail
MODEL=${1:?用法见文件开头}
BEHAVIOR=${2:?用法见文件开头}
POS=${3:?用法见文件开头}
NEG=${4:?用法见文件开头}
LRS=(5e-6 1e-5 2e-5)
i=${SLURM_ARRAY_TASK_ID:?需要用 --array=0-5 提交}
if (( i < 3 )); then ROLE=pos; DATA=$POS; else ROLE=neg; DATA=$NEG; fi
LR=${LRS[$((i % 3))]}

BASE=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
SIF=$BASE/containers/manifold_steering_current.sif
REPO=$BASE/Manifold_Steering_LLM
OUT=$REPO/ckpt/$(basename "$MODEL")/$BEHAVIOR/${ROLE}_lr$LR
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1

echo "节点: $(hostname)   作业: ${SLURM_ARRAY_JOB_ID}_$i   $ROLE  lr=$LR   数据: $DATA   开始: $(date)"
echo "镜像: $(readlink -f "$SIF")"
source "$REPO/slurm/one_gpu.sh"
cd "$REPO"
apptainer exec --nv -B "$BASE" "$SIF" /opt/venvs/train/bin/python scripts/train_lora.py \
  --model "$MODEL" --dataset "$DATA" --lr "$LR" --out "$OUT"
echo "结束: $(date)"
