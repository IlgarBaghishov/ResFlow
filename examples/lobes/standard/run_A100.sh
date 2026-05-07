#!/bin/bash
#SBATCH -J resflow
#SBATCH --account=ALLOCATION_NAME  # set to your HPC allocation
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm_%j.log
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 08:00:00

pwd; hostname -f; date
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500

date
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py
date

python sample.py
date
