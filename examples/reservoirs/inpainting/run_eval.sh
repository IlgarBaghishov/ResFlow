#!/bin/bash
# Launch eval_losses.py inside an existing salloc allocation:
#   srun --jobid=<JOBID> --nodes=2 --ntasks-per-node=1 bash run_eval.sh
set -e

source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate genflows

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29501

cd $WORK/codes/ResFlow/examples/reservoirs/inpainting

torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=3 \
  --rdzv_id="$SLURM_JOB_ID-eval" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  eval_losses.py "$@"
