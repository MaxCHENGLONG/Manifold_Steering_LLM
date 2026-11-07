#!/bin/bash
# C2：沿用粗扫的配置，在谄媚评测子集（125 组 × 4 种提示 = 500 条）上采样生成，再把生成的文本
# teacher-forcing 喂回模型逐层打分（steered 与 base_model 两种条件）；同时给粗扫存下的 TinyMMLU 回答打分。
# 作业数组按方法分 3 个任务，每个任务 1 块 GH200：0 = 基线与权重 steering，1 = 第 17 层激活，2 = 全层激活。
# 等粗扫完成后提交：
#   cd /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/Manifold_Steering_LLM && \
#   sbatch --array=0-2 slurm/c2_score.sbatch Qwen/Qwen2.5-1.5B-Instruct sycophancy sycophantic
# 参数与 sweep_coarse.sbatch 相同：模型、行为、激活向量的文件名前缀、可选的激活 steering 层（默认 17）。
# 输出 results/<模型名>/c2/<行为>/<配置名>/{responses.jsonl,meta.json,scores.parquet}，
# 以及 results/<模型名>/coarse/<配置名>/scores.parquet；中断后重新提交会跳过已完成的部分。
# 只给已有的回答补一个新打分器时，用环境变量指定打分器和输出文件名（生成会因已完成而跳过），例如 M3：
#   SCORERS=trans OUT_NAME=scores_trans.parquet sbatch --array=0-2 slurm/c2_score.sbatch Qwen/Qwen2.5-1.5B-Instruct sycophancy sycophantic
#SBATCH -A naiss2026-4-1521-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J c2-score
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/logs/%x_%A_%a.out

set -euo pipefail
MODEL=${1:?用法见文件开头}
BEHAVIOR=${2:?用法见文件开头}
TRAIT=${3:?用法见文件开头}
LAYER=${4:-17}
METHOD_GROUPS=("base,weight" "act" "act_all")
METHODS=${METHOD_GROUPS[${SLURM_ARRAY_TASK_ID:?需要用 --array=0-2 提交}]}

BASE=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
SIF=$BASE/containers/manifold_steering_current.sif
REPO=$BASE/Manifold_Steering_LLM
NAME=$(basename "$MODEL")
COARSE=$REPO/results/$NAME/coarse
OUT=$REPO/results/$NAME/c2/$BEHAVIOR
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1

echo "节点: $(hostname)   作业: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}   方法: $METHODS   开始: $(date)"
echo "镜像: $(readlink -f "$SIF")"
source "$REPO/slurm/one_gpu.sh"
cd "$REPO"
run() { apptainer exec --nv -B "$BASE" "$SIF" python "$@"; }

[[ -s $COARSE/summary.json ]] || { echo "缺少粗扫结果 $COARSE/summary.json，先跑 sweep_coarse.sbatch"; exit 1; }
COMMON=(--model "$MODEL" --coarse "$COARSE" --delta "$REPO/ckpt/$NAME/$BEHAVIOR/delta.pt"
        --vector "$REPO/data/vectors/$NAME/${TRAIT}_response_avg_diff.pt" --layer "$LAYER" --methods "$METHODS")
SCORE_ARGS=()
[[ -n ${SCORERS:-} ]] && SCORE_ARGS+=(--scorers "$SCORERS")
[[ -n ${OUT_NAME:-} ]] && SCORE_ARGS+=(--out_name "$OUT_NAME")
echo "打分参数: ${SCORE_ARGS[*]:-默认}"
run scripts/generate_steered.py "${COMMON[@]}" --out "$OUT"
run scripts/score_teacher_forced.py "${COMMON[@]}" "${SCORE_ARGS[@]}" --kind generated --responses "$OUT" \
  --stats "$REPO/stats/$NAME"
run scripts/score_teacher_forced.py "${COMMON[@]}" "${SCORE_ARGS[@]}" --kind tinymmlu --stats "$REPO/stats/$NAME"
echo "结束: $(date)"
