#!/bin/bash
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -p gh
#SBATCH -t 36:00:00
#SBATCH -o ll_out_%j
#SBATCH -A ALLOCATION_NAME  # set to your HPC allocation

# GH200 HPC launcher for reservoir well-conditioning (inpainting) training.

# --- environment ----------------------------------------------------------
# Project env (created with: conda create -n genflows python=3.12 -y && pip install -e .)
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate genflows

# --- DDP rendezvous -------------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500

# --- run directory on $SCRATCH -------------------------------------------
RUN_DIR=$SCRATCH/genflows_runs/reservoirs_inpainting
mkdir -p "$RUN_DIR"

TRAIN_PY="$SLURM_SUBMIT_DIR/train.py"

# train.py defaults to $SCRATCH/SiliciclasticReservoirs and auto-downloads
# if missing. Override only if you keep the dataset somewhere else:
# export RESERVOIR_DATA_DIR=/path/to/SiliciclasticReservoirs

cd "$RUN_DIR"
echo "RUN_DIR  = $RUN_DIR"
echo "TRAIN_PY = $TRAIN_PY"
date
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=1 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  "$TRAIN_PY"
date
