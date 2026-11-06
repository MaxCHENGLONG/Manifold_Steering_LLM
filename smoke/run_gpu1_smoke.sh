#!/bin/bash
# 单卡冒烟测试：在 1 卡交互会话里运行
#   bash smoke/run_gpu1_smoke.sh
# 依次运行 vLLM 生成、HF 前向 hook、GLP 加载三项测试，日志写到 $PROJ/logs/。
set -uo pipefail

PROJ=${PROJ:-/nobackup/proj/disk/naiss2025-22-1730/personal/licheng}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF=${SIF:-$PROJ/containers/manifold_steering_current.sif}
LOG=$PROJ/logs/smoke_gpu1_$(date +%Y%m%d-%H%M).log
mkdir -p "$PROJ/logs"
# 镜像里没有 nvcc，FlashInfer 采样器需要现场编译会失败；改用 PyTorch 原生采样（宿主机变量会传进容器）
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
cd "$REPO"

echo "节点: $(hostname)   镜像: $(readlink -f "$SIF")" | tee "$LOG"

declare -A RESULT
run() {
  local name=$1; shift
  echo -e "\n===== $name =====" | tee -a "$LOG"
  if apptainer exec --nv -B "$PROJ" "$SIF" "$@" 2>&1 | tee -a "$LOG"; then
    RESULT[$name]=PASS
  else
    RESULT[$name]=FAIL
  fi
}

run "1-vLLM生成"  python smoke/gpu1_vllm.py
run "2-HF-hook"   python smoke/gpu2_hf_hook.py
run "3-GLP加载"   python smoke/gpu3_glp.py

echo -e "\n===== 汇总 =====" | tee -a "$LOG"
fail=0
for name in "1-vLLM生成" "2-HF-hook" "3-GLP加载"; do
  echo "  $name: ${RESULT[$name]}" | tee -a "$LOG"
  [[ ${RESULT[$name]} == PASS ]] || fail=1
done
echo "日志：$LOG"
exit $fail
