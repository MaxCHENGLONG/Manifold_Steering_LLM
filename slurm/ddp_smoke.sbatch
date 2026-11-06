#!/bin/bash
# 4 卡 DDP 冒烟测试：申请 1 个整节点（4 块 GH200），在容器的 train 环境里用 torchrun 运行。
# 提交：cd /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/Manifold_Steering_LLM && sbatch slurm/ddp_smoke.sbatch
#SBATCH -A naiss2026-4-1521-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=4
##SBATCH --exclusive
#SBATCH -t 00:20:00
#SBATCH -J ddp-smoke
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/logs/%x_%j.out

set -euo pipefail
BASE=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng
SIF=$BASE/containers/manifold_steering_current.sif
PROJ=$BASE/Manifold_Steering_LLM            # 所有代码都在这里
SCRIPT=$PROJ/smoke/ddp_smoke.py

echo "节点: $(hostname)   作业: $SLURM_JOB_ID   开始: $(date)"
echo "镜像: $(readlink -f "$SIF")"
echo "---- GPU 互连拓扑 ----"
nvidia-smi topo -m || true
echo "----------------------"

# 需要排查通信问题时，提交前 export NCCL_DEBUG=INFO 即可看到 NCCL 选用的传输方式
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

apptainer exec --nv -B "$BASE" "$SIF" \
  /opt/venvs/train/bin/torchrun --standalone --nproc_per_node=4 "$SCRIPT"

echo "结束: $(date)"
