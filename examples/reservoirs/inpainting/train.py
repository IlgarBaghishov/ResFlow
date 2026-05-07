"""Train wells-only Flow-Matching inpainting on SiliciclasticReservoirs.

Mirrors examples/lobes/inpainting/train.py but for the reservoir dataset:
  - 1M samples across 8 architectures
  - cube tensor (1, 64, 64, 32) — z is the LAST axis (surface at z=Z-1)
  - cond is 18-dim (layer-type one-hot + scalars + sin/cos azimuth)
  - mask: 30% empty + 70% wells (1..5 straight-line wells, 50/50
    boundary-to-boundary vs truncated, drilling DOWN from z=Z-1)
  - UNet3D in_channels=3 (noisy_x | known_data | mask), out_channels=1

Resumable: rerun the same `sbatch run_GH200.sh` and it picks up from
`checkpoints/training_state.pt`. To reset, delete that file.
"""
import os
import numpy as np
import torch
from accelerate import Accelerator

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.utils.data_reservoirs import (
    get_reservoir_inpaint_loaders, COND_DIM, LAYER_TYPES,
)
from resflow.utils.plotting import plot_loss
from resflow.utils.training import train_model_inpaint


# Resolution order:
#   1. RESERVOIR_DATA_DIR (explicit override)
#   2. $SCRATCH/SiliciclasticReservoirs (works on any HPC where $SCRATCH is set)
#   3. ./SiliciclasticReservoirs (last-resort fallback)
# Auto-downloaded on first use by ReservoirDataset._ensure_dataset_local
# (no-op when files are already present), so a single `python train.py` is enough.
_scratch = os.environ.get('SCRATCH')
DEFAULT_DATA_DIR = os.environ.get(
    'RESERVOIR_DATA_DIR',
    os.path.join(_scratch, 'SiliciclasticReservoirs') if _scratch
    else os.path.abspath('SiliciclasticReservoirs'),
)
CHECKPOINT_DIR = 'checkpoints'
SAVE_EVERY = 5  # checkpoint every 5 epochs (~1.5 h on an 8-node GH200 HPC)


def main():
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using device: {device}")
    accelerator.print(f"Data dir: {DEFAULT_DATA_DIR}")

    if accelerator.is_main_process:
        get_reservoir_inpaint_loaders(data_dir=DEFAULT_DATA_DIR, batch_size=32,
                                      num_workers=0)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs("results", exist_ok=True)
    accelerator.wait_for_everyone()

    train_loader, val_loader, test_loader, dataset = get_reservoir_inpaint_loaders(
        data_dir=DEFAULT_DATA_DIR, batch_size=32, num_workers=4,
    )
    accelerator.print(f"Dataset: {len(dataset)} train / "
                      f"{len(val_loader.dataset)} val / "
                      f"{len(test_loader.dataset)} test")
    accelerator.print(f"Cond dim: {COND_DIM}  Layer types: {LAYER_TYPES}")

    if accelerator.is_main_process:
        np.savez(
            os.path.join(CHECKPOINT_DIR, 'cond_stats.npz'),
            cont_min=dataset.cont_min,
            cont_max=dataset.cont_max,
            layer_types=np.array(LAYER_TYPES, dtype=object),
        )

    epochs_fm = 40

    accelerator.print("\n--- Training Flow Matching Inpaint (wells-only) ---")
    model_fm = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                      num_time_embs=1, expand_angle_idx=None).to(device)
    method_fm = FlowMatching(model_fm)
    loss_fm = train_model_inpaint(
        method_fm, train_loader, epochs=epochs_fm, accelerator=accelerator,
        checkpoint_dir=CHECKPOINT_DIR, save_every=SAVE_EVERY,
        total_epochs=epochs_fm,
    )
    if accelerator.is_main_process:
        torch.save(method_fm.model.state_dict(),
                   os.path.join(CHECKPOINT_DIR, 'flow_matching.pt'))
        np.save(os.path.join(CHECKPOINT_DIR, 'loss_history_fm.npy'),
                np.array(loss_fm))
        plot_loss(loss_fm, "FM Inpaint (wells) Training Loss",
                  "results/loss_flow_matching.png")

    accelerator.print("\nDone.")
    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
