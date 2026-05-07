#!/bin/bash
#SBATCH -J resflow-res-inp
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-a100
#SBATCH -t 24:00:00
#SBATCH -o ll_out_%j
#SBATCH -A ALLOCATION_NAME  # set to your HPC allocation

# 3-A100-per-node HPC launcher for reservoir well-conditioning (inpainting) training.
# 3x A100-40GB per node, 4 nodes default.
#
# Sizing (extrapolated from a 3-A100 single-node measurement, BS=32/rank,
# in-channels=3 inpainting model, ~1.95 it/s):
#   - ~80 min/epoch on 1 node, 40 epochs => ~53 h on 1 node, ~14 h on 4 nodes
#   - Total cost roughly constant: ~55-60 node-hours regardless of node count
# Default: 4 nodes / 16 h wall (some headroom over the ~14 h estimate).
# Training is resumable via $RUN_DIR/checkpoints/training_state.pt: just
# resubmit the same script and it picks up at the last save_every boundary
# (every 5 epochs ≈ every 1.75 h on 4 nodes).
#
# To launch:
#   1. Fill in -A above with your HPC allocation
#   2. sbatch run_A100.sh

# --- environment ----------------------------------------------------------
# Project env (created with: conda create -n genflows python=3.12 -y && pip install -e .)
source $WORK/miniforge3/etc/profile.d/conda.sh
conda activate genflows

# --- DDP rendezvous -------------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500

# --- run directory on $SCRATCH -------------------------------------------
# Outputs (checkpoints/, results/) land alongside the dataset on the fast
# parallel filesystem instead of $WORK (which has a per-user quota).
# Stable path, NOT job-id-suffixed, so resume works across resubmits.
RUN_DIR=$SCRATCH/genflows_runs/reservoirs_inpainting
mkdir -p "$RUN_DIR"

# Absolute path to the training script — independent of where we cd to.
# NOTE: ${BASH_SOURCE[0]} resolves to SLURM's per-job spool copy of this
# script (e.g. /var/spool/slurmd/jobNNN/run_A100.sh), which does not contain
# train.py. Use $SLURM_SUBMIT_DIR (the dir you ran `sbatch` from) instead.
TRAIN_PY="$SLURM_SUBMIT_DIR/train.py"

# train.py defaults the data dir to $SCRATCH/SiliciclasticReservoirs and
# auto-downloads if missing (rank 0 only; other ranks wait at the barrier).
# Override only if you keep the dataset somewhere else:
# export RESERVOIR_DATA_DIR=/path/to/SiliciclasticReservoirs

cd "$RUN_DIR"
echo "RUN_DIR  = $RUN_DIR"
echo "TRAIN_PY = $TRAIN_PY"
date
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=3 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  "$TRAIN_PY"
date
