#!/bin/bash
# 逐层统计（模块 1）：1 块 GH200。先用 vLLM 生成 on-policy 对话（文件已存在就跳过），再逐层累计统计。
# 提交：cd /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/Manifold_Steering_LLM && \
#       sbatch slurm/layer_stats.sbatch Qwen/Qwen2.5-1.5B-Instruct [fit 切分 token 数，默认 2e7]
# 7B 模型建议加大时限：sbatch -t 04:00:00 slurm/layer_stats.sbatch Qwen/Qwen2.5-7B-Instruct
#SBATCH -A naiss2026-4-1521-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J layer-stats
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/logs/%x_%j.out

set -euo pipefail
MODEL=${1:?用法：sbatch slurm/layer_stats.sbatch <模型> [token 数]}
N_TOKENS=${2:-2e7}
BASE=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
SIF=$BASE/containers/manifold_steering_current.sif
REPO=$BASE/Manifold_Steering_LLM
NAME=$(basename "$MODEL")
CHAT=$REPO/data/chat_onpolicy/$NAME.jsonl
OUT=$REPO/stats/$NAME

# 只读预下载的缓存；镜像里没有 nvcc，禁用需要现场编译的 FlashInfer 采样器
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0

echo "节点: $(hostname)   作业: $SLURM_JOB_ID   模型: $MODEL   开始: $(date)"
echo "镜像: $(readlink -f "$SIF")"
source "$REPO/slurm/one_gpu.sh"
cd "$REPO"
run() { apptainer exec --nv -B "$BASE" "$SIF" python "$@"; }

if [[ -s $CHAT ]]; then
  echo "已有 on-policy 对话：$CHAT"
else
  run scripts/generate_onpolicy_chat.py --model "$MODEL" --out "$CHAT"
fi
run scripts/collect_layer_stats.py --model "$MODEL" --chat "$CHAT" --n_tokens "$N_TOKENS" --out "$OUT"
echo "结束: $(date)"
