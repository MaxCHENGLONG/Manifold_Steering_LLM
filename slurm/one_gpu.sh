# 由 slurm/*.sbatch source（需要先定义 SIF）：让进程只看到一块 GPU。
# 在 Arrhenius 上，带 -N 的作业会分到整台节点的 4 块 GPU（不带 -N 时 --gpus=1 只分 1 块），
# trl 看到多块会直接拒绝训练。sbatch 模板已去掉 -N，这里再兜底：看到多块时只用分到的第一块。
n=$(apptainer exec --nv "$SIF" nvidia-smi -L | wc -l)
echo "GPU: 可见 $n 块   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-未设置}   SLURM_JOB_GPUS=${SLURM_JOB_GPUS-未设置}"
if (( n > 1 )); then
  ids=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:?看到多块 GPU，但没有 SLURM_JOB_GPUS，不知道分到的是哪块}}
  export CUDA_DEVICE_ORDER=PCI_BUS_ID  # 和 nvidia-smi、Slurm 的编号一致
  export CUDA_VISIBLE_DEVICES=${ids%%,*}
  echo "GPU: 改为只用 $CUDA_VISIBLE_DEVICES 号"
fi
