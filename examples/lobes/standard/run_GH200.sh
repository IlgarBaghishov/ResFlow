#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -p gh-dev
#SBATCH -t 2:00:00
#SBATCH -o ll_out
#SBATCH -A ALLOCATION_NAME  # set to your HPC allocation

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500

date
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=1 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py
date

python sample.py
date
