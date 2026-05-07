"""Big lobe-only reservoir with varying lobe properties across the grid.

Mirrors the gradient-pattern of examples/lobes/big_reservoir/generate.py
(size-related scalars decrease along Y, azimuth varies along X, NTG fixed)
but uses the SiliciclasticReservoirs 3-channel inpaint model and the
multi-block parallel denoiser in resflow.assembly.

Layer type is fixed = "lobe" everywhere, so the conditioning differs only
in the per-block scalars (width_cells, depth_cells, asp, azimuth).

Three overlap variants are produced (12, 16, 24) with hard transition mode
(no transition needed since every block is the same layer type).

Run from this directory:
    python generate.py
"""
import os
import time

import numpy as np
import torch

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.assembly import (
    BlockSpec, COND_DIM, LAYER_TYPE_TO_IDX,
    generate_big_reservoir_multi,
)


CKPT = os.path.join(os.environ.get('SCRATCH', '.'), 'genflows_runs/reservoirs_inpainting/checkpoints/flow_matching.pt')
COND_STATS = os.path.join(os.environ.get('SCRATCH', '.'), 'genflows_runs/reservoirs_inpainting/checkpoints/cond_stats.npz')
OUT_DIR = os.environ.get('RESERVOIR_OUT_DIR', 'results')

GRID_SHAPE = (10, 10)            # (ny, nx) blocks
BLOCK_SHAPE = (64, 64, 32)
OVERLAPS = [16, 24]
SEED = int(os.environ.get('RESERVOIR_SEED', 42))
N_STEPS = 50
CFG_SCALE = 3.0

# Property gradient (as fractions of the training range; mirrors the
# lobes example's [0.2, 0.8] / [0.1, 0.9] normalized ranges).
WIDTH_NORM_RANGE  = (0.8, 0.2)   # along Y (decrease)
DEPTH_NORM_RANGE  = (0.6, 0.2)   # along Y (decrease, capped to stay safe)
ASP_NORM_RANGE    = (0.7, 0.3)   # along Y (decrease)
AZIMUTH_DEG_RANGE = (45.0, 135.0)  # along X (140-deg fan centered at 90)
NTG_FIXED = 0.7                  # same value as lobes example


def _denorm(norm, lo, hi):
    return float(lo + norm * (hi - lo))


def build_grid_specs(cont_min, cont_max):
    """Build (ny, nx) BlockSpec grid: lobe everywhere, scalars from gradients."""
    ny, nx = GRID_SHAPE
    # Indices into CONT_COLS = ['ntg','width_cells','depth_cells','asp',...]
    w_lo, w_hi = cont_min[1], cont_max[1]
    d_lo, d_hi = cont_min[2], cont_max[2]
    a_lo, a_hi = cont_min[3], cont_max[3]

    grid: list[list[BlockSpec]] = []
    for i in range(ny):
        # Y fraction: 0 at bottom (i=0), 1 at top (i=ny-1) — bottom is large.
        fy = i / max(ny - 1, 1)
        wn = WIDTH_NORM_RANGE[0] + fy * (WIDTH_NORM_RANGE[1] - WIDTH_NORM_RANGE[0])
        dn = DEPTH_NORM_RANGE[0] + fy * (DEPTH_NORM_RANGE[1] - DEPTH_NORM_RANGE[0])
        an = ASP_NORM_RANGE[0]   + fy * (ASP_NORM_RANGE[1]   - ASP_NORM_RANGE[0])

        width = _denorm(wn, w_lo, w_hi)
        depth = _denorm(dn, d_lo, d_hi)
        aspr  = _denorm(an, a_lo, a_hi)

        row: list[BlockSpec] = []
        for j in range(nx):
            fx = j / max(nx - 1, 1)
            az = AZIMUTH_DEG_RANGE[0] + fx * (AZIMUTH_DEG_RANGE[1] - AZIMUTH_DEG_RANGE[0])
            row.append(BlockSpec(
                layer_idx=LAYER_TYPE_TO_IDX['lobe'],
                azimuth_deg=az,
                raw_scalars={
                    'ntg': NTG_FIXED,
                    'width_cells': width,
                    'depth_cells': depth,
                    'asp': aspr,
                },
            ))
        grid.append(row)
    return grid


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {device}  out: {OUT_DIR}")
    print(f"Grid: {GRID_SHAPE[0]}x{GRID_SHAPE[1]} blocks of {BLOCK_SHAPE}")

    stats = np.load(COND_STATS, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']

    grid = build_grid_specs(cont_min, cont_max)

    # Print the scalar gradient for the user to sanity-check.
    print(f"\nProperty gradient (lobe everywhere, NTG={NTG_FIXED} fixed):")
    print(f"  width_cells  Y: {grid[0][0].raw_scalars['width_cells']:6.1f}  "
          f"-> {grid[-1][0].raw_scalars['width_cells']:6.1f}")
    print(f"  depth_cells  Y: {grid[0][0].raw_scalars['depth_cells']:6.1f}  "
          f"-> {grid[-1][0].raw_scalars['depth_cells']:6.1f}")
    print(f"  asp          Y: {grid[0][0].raw_scalars['asp']:6.2f}  "
          f"-> {grid[-1][0].raw_scalars['asp']:6.2f}")
    print(f"  azimuth_deg  X: {grid[0][0].azimuth_deg:6.1f}  "
          f"-> {grid[0][-1].azimuth_deg:6.1f}")

    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    method = FlowMatching(model)

    ny, nx = GRID_SHAPE
    Sx, Sy, Sz = BLOCK_SHAPE

    for overlap in OVERLAPS:
        tag = f"hard_ov{overlap:02d}"
        Tx = Sx + (nx - 1) * (Sx - overlap)
        Ty = Sy + (ny - 1) * (Sy - overlap)
        print(f"\n== {tag} == grid={ny}x{nx}, total={Tx}x{Ty}x{Sz}")
        torch.manual_seed(SEED)
        t0 = time.time()
        x_global, _ = generate_big_reservoir_multi(
            method, grid, cont_min, cont_max,
            block_shape=BLOCK_SHAPE, overlap_xy=overlap,
            n_steps=N_STEPS, cfg_scale=CFG_SCALE,
            max_batch=24, device=device,
        )
        elapsed = time.time() - t0
        volume = x_global.numpy()
        binary = (volume > 0).astype(np.int8)

        out_npz = os.path.join(OUT_DIR, f"reservoir_{tag}.npz")
        np.savez(out_npz,
                 volume=volume.astype(np.float32),
                 binary=binary,
                 mode='hard',
                 overlap=overlap,
                 block_shape=np.array(BLOCK_SHAPE),
                 ny=ny, nx=nx,
                 pure_x_indices=np.array(list(range(nx)), dtype=np.int32),
                 pure_x_layer=np.array(['lobe'] * nx, dtype=object),
                 trans_x_indices=np.array([], dtype=np.int32),
                 trans_kind='hard',
                 row_azimuths_deg=np.array(
                     [grid[i][0].azimuth_deg for i in range(ny)], dtype=np.float32),
                 elapsed_s=elapsed)
        print(f"  saved -> {out_npz}")
        print(f"  shape={binary.shape}  NTG={binary.mean():.3f}  {elapsed:.1f}s")


if __name__ == '__main__':
    main()
