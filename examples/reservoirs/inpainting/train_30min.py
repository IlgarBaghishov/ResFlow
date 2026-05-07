"""~30-min reservoir-inpaint training run on a 3-A100 HPC node.

Subset (50k train / 1.5k val / 1.5k test) for 5 epochs. Tracks per-epoch
train + val FM loss, plus a final test loss. Saves an EMA-applied
checkpoint + a loss-curves PNG into results_30min/.
"""
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from tqdm import tqdm

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.utils.data_reservoirs import (
    ReservoirDataset, COND_DIM, LAYER_TYPES, VOLUME_SHAPE,
)
from resflow.utils.masking import InpaintDataset
from resflow.utils.training import EMA, _make_scheduler


_scratch = os.environ.get('SCRATCH')
DATA_DIR = os.environ.get(
    'RESERVOIR_DATA_DIR',
    os.path.join(_scratch, 'SiliciclasticReservoirs') if _scratch
    else os.path.abspath('SiliciclasticReservoirs'),
)
N_TRAIN_SUB = 50000
N_VAL_SUB = 1500
N_TEST_SUB = 1500
BATCH = 32
EPOCHS = 5
NUM_WORKERS = 4
EMA_DECAY = 0.99  # short run -> use a fast EMA, not the default 0.9999
OUT_DIR = 'checkpoints_30min'
RES_DIR = 'results_30min'


@torch.no_grad()
def eval_loss(method, loader, accelerator):
    raw = method.model.module if hasattr(method.model, 'module') else method.model
    method.model.eval()
    total = 0.0
    n = 0
    for x, cond, mask in loader:
        x = x.to(accelerator.device)
        cond = cond.to(accelerator.device)
        mask = mask.to(accelerator.device)
        raw.set_inpaint_context(mask, x * mask)
        loss = method.compute_loss(x, cond)
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    raw.clear_inpaint_context()
    method.model.train()
    return total / max(n, 1)


def main():
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Device: {device}  ranks: {accelerator.num_processes}")
    accelerator.print(f"Data dir: {DATA_DIR}")

    # Build the cond cache + dataset on rank 0 first.
    if accelerator.is_main_process:
        ReservoirDataset(DATA_DIR, split='train')
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(RES_DIR, exist_ok=True)
    accelerator.wait_for_everyone()

    train_full = ReservoirDataset(DATA_DIR, split='train')
    val_full = ReservoirDataset(DATA_DIR, split='val',
                                cont_min=train_full.cont_min,
                                cont_max=train_full.cont_max)
    test_full = ReservoirDataset(DATA_DIR, split='test',
                                 cont_min=train_full.cont_min,
                                 cont_max=train_full.cont_max)

    if accelerator.is_main_process:
        np.savez(os.path.join(OUT_DIR, 'cond_stats.npz'),
                 cont_min=train_full.cont_min, cont_max=train_full.cont_max,
                 layer_types=np.array(LAYER_TYPES, dtype=object))

    rng = np.random.default_rng(0)
    train_idx = rng.choice(len(train_full), size=N_TRAIN_SUB, replace=False).tolist()
    val_idx = rng.choice(len(val_full), size=N_VAL_SUB, replace=False).tolist()
    test_idx = rng.choice(len(test_full), size=N_TEST_SUB, replace=False).tolist()
    train_set = Subset(InpaintDataset(train_full, volume_shape=VOLUME_SHAPE), train_idx)
    val_set = Subset(InpaintDataset(val_full, volume_shape=VOLUME_SHAPE), val_idx)
    test_set = Subset(InpaintDataset(test_full, volume_shape=VOLUME_SHAPE), test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False,
                            num_workers=2, persistent_workers=True,
                            pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False,
                             num_workers=2, persistent_workers=True,
                             pin_memory=True)

    accelerator.print(f"Subset sizes: train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    accelerator.print(f"Steps/epoch (per rank): {len(train_loader)}")

    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    method = FlowMatching(model)
    accelerator.print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = _make_scheduler(optimizer, EPOCHS, accelerator.num_processes)
    method.model, optimizer, train_loader, scheduler = accelerator.prepare(
        method.model, optimizer, train_loader, scheduler
    )
    val_loader, test_loader = accelerator.prepare(val_loader, test_loader)
    method.model.train()
    ema = EMA(method.model, decay=EMA_DECAY)

    train_losses = []
    val_losses = []
    t_start = time.time()
    for epoch in range(EPOCHS):
        method.model.train()
        raw = method.model.module if hasattr(method.model, 'module') else method.model
        total = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"E{epoch+1}/{EPOCHS}",
                    disable=not accelerator.is_main_process)
        for x, cond, mask in pbar:
            raw.set_inpaint_context(mask, x * mask)
            optimizer.zero_grad()
            loss = method.compute_loss(x, cond)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(method.model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(method.model)
            total += loss.item()
            n += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        scheduler.step()
        train_losses.append(total / n)

        v = eval_loss(method, val_loader, accelerator)
        val_losses.append(v)
        accelerator.print(
            f"Epoch {epoch+1}: train={train_losses[-1]:.4f}  val={v:.4f}  "
            f"({time.time()-t_start:.0f}s elapsed)"
        )

    raw = method.model.module if hasattr(method.model, 'module') else method.model
    raw.clear_inpaint_context()
    ema.apply(method.model)
    test_loss = eval_loss(method, test_loader, accelerator)
    accelerator.print(f"\nFinal test loss (EMA): {test_loss:.4f}")
    accelerator.print(f"Total wall: {time.time()-t_start:.0f}s")

    if accelerator.is_main_process:
        method.model = accelerator.unwrap_model(method.model)
        torch.save(method.model.state_dict(),
                   os.path.join(OUT_DIR, 'flow_matching.pt'))
        np.savez(os.path.join(RES_DIR, 'losses.npz'),
                 train=np.array(train_losses),
                 val=np.array(val_losses),
                 test=np.array([test_loss]))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs_x = list(range(1, EPOCHS + 1))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_x, train_losses, '-o', label='train')
        ax.plot(epochs_x, val_losses, '-s', label='val')
        ax.axhline(test_loss, color='red', linestyle='--',
                   label=f'test (final={test_loss:.4f})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('FM loss')
        ax.set_title('Reservoir Inpaint FM — 30-min run on 3xA100')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(RES_DIR, 'losses.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        accelerator.print(f"Saved checkpoint -> {OUT_DIR}/flow_matching.pt")
        accelerator.print(f"Saved losses    -> {RES_DIR}/losses.{{npz,png}}")

    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
