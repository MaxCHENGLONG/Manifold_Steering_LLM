#!/bin/bash
# C1 粗扫：先由训练好的 τ⁺/τ⁻ 算 Δ（已存在则跳过），再在 TinyMMLU 上扫三种方法。1 块 GH200。
# 等 train_lora 的 6 个任务都成功后提交，或者挂依赖一次提交（afterok 会等数组里全部任务成功）：
#   cd /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/Manifold_Steering_LLM && \
#   sbatch --dependency=afterok:<训练作业号> slurm/sweep_coarse.sbatch Qwen/Qwen2.5-1.5B-Instruct sycophancy sycophantic
# 第 3 个参数是激活向量的文件名前缀（data/vectors/<模型名>/<前缀>_response_avg_diff.pt）；
# 第 4 个参数可选，是激活 steering 层（weight-steering 约定，1.5B 默认 17）。
# 输出 results/<模型名>/coarse/；中断后重新提交会跳过已完成的配置。
#SBATCH -A naiss2026-4-1521-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 03:00:00
#SBATCH -J sweep-coarse
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/logs/%x_%j.out

set -euo pipefail
MODEL=${1:?用法见文件开头}
BEHAVIOR=${2:?用法见文件开头}
TRAIT=${3:?用法见文件开头}
LAYER=${4:-17}

BASE=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
SIF=$BASE/containers/manifold_steering_current.sif
REPO=$BASE/Manifold_Steering_LLM
NAME=$(basename "$MODEL")
CK=$REPO/ckpt/$NAME/$BEHAVIOR
DELTA=$CK/delta.pt
VECTOR=$REPO/data/vectors/$NAME/${TRAIT}_response_avg_diff.pt
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1

echo "节点: $(hostname)   作业: $SLURM_JOB_ID   模型: $MODEL   开始: $(date)"
echo "镜像: $(readlink -f "$SIF")"
source "$REPO/slurm/one_gpu.sh"
cd "$REPO"
run() { apptainer exec --nv -B "$BASE" "$SIF" python "$@"; }

[[ -s $VECTOR ]] || { echo "缺少激活向量 $VECTOR，先运行 scripts/compute_diffmean_vectors.py"; exit 1; }
if [[ -s $DELTA ]]; then
  echo "已有 Δ：$DELTA"
else
  run scripts/make_weight_delta.py --model "$MODEL" --pos "$CK"/pos_lr* --neg "$CK"/neg_lr* --out "$DELTA"
fi
run scripts/sweep_capability.py --model "$MODEL" --delta "$DELTA" --vector "$VECTOR" --layer "$LAYER" \
  --stats "$REPO/stats/$NAME" --out "$REPO/results/$NAME/coarse"
echo "结束: $(date)"
