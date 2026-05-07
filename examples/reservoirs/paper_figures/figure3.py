"""NeurIPS Figure 3: well-conditioned FM samples for all 8 reservoir
architectures. Same 3-row x 8-col layout as Figure 2, but each row's
cubes are generated with a different well configuration tailored to
that cross-section, and the well footprints are outlined in black.

Per-row well configuration (3 wells each, one cube per layer type per row):
  Row 0 (XY @ z=18): three HORIZONTAL wells lying flat along the Y axis
    at fixed X in {16, 32, 48} and z=18. In the XY plot they appear as
    3 vertical bars spanning Y at x=16, 32, 48.
  Row 1 (XZ @ y=32): three VERTICAL wells at x in {16, 32, 48}, y=32,
    full z extent. They appear as 3 vertical bars in the XZ plot.
  Row 2 (YZ @ x=32): three VERTICAL wells at x=32, y in {16, 32, 48},
    full z extent. They appear as 3 vertical bars in the YZ plot.

Conditioning per layer type matches Figures 1 & 2 (same 18-D cond vector,
same source row in the train/test cond cache). Well-log values are carved
from the SAME real cube whose row supplied the cond vector.

Cubes are cached to figs/figure3_cubes.npz (24 cubes = 8 layer types x
3 well configs).

Run:
    python examples/reservoirs/paper_figures/figure3.py
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

NEURIPS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NEURIPS_DIR))
import figure2 as fig2  # noqa: E402

from resflow.assembly import COND_DIM, LAYER_TYPE_TO_IDX  # noqa: E402
from resflow.methods.flow_matching import FlowMatching  # noqa: E402
from resflow.models.unet3d import UNet3D  # noqa: E402
from resflow.utils.data_reservoirs import (  # noqa: E402
    ReservoirDataset, VOLUME_SHAPE,
)
from resflow.utils.masking import apply_inpaint_output  # noqa: E402
from resflow.utils.plotting_reservoirs import (  # noqa: E402
    CMAP_FACIES, NORM_FACIES, FACIES_BINARY_LABELS,
    slice_xy, slice_xz, slice_yz, imshow_slice,
)


DEFAULT_CKPT = fig2.DEFAULT_CKPT
DEFAULT_COND_STATS = fig2.DEFAULT_COND_STATS
DEFAULT_DATA_DIR = fig2.DEFAULT_DATA_DIR
COLUMN_ORDER = fig2.COLUMN_ORDER

N_STEPS = fig2.N_STEPS
CFG = fig2.CFG
SAMPLE_SEED = 11

# Entropy ensemble: number of realizations per (row, layer) used to estimate
# the per-voxel sand probability and Bernoulli entropy. Matches the
# big_reservoir wells_entropy.py convention (n-real=50 in run_restore.sh).
N_REAL_ENTROPY = 50
ENTROPY_BATCH = 25

WELL_COLOR = 'black'
WELL_LW = 0.5
WELL_ALPHA = 0.30

# Default slice indices (match Figure 2).
Z_SLICE_DEFAULT = 18
Y_SLICE_DEFAULT = 32
X_SLICE_DEFAULT = 32

# Default = single well through the centre. Override via --well-fractions
# from the CLI to place N wells at arbitrary fractions of each axis.
WELL_FRACTIONS = [0.5]


def well_positions_x():
    return [int(round(VOLUME_SHAPE[0] * f)) for f in WELL_FRACTIONS]


def well_positions_y():
    return [int(round(VOLUME_SHAPE[1] * f)) for f in WELL_FRACTIONS]


# --------------------------- mask construction -------------------------------

def build_xy_mask(z_slice):
    """3 horizontal wells along Y at fixed X={16,32,48} and z=z_slice."""
    X, Y, Z = VOLUME_SHAPE
    mask = torch.zeros(1, X, Y, Z)
    for x in well_positions_x():
        mask[0, x, :, z_slice] = 1.0
    return mask


def build_xz_mask(y_slice):
    """3 vertical wells at x={16,32,48}, y=y_slice, full z."""
    X, Y, Z = VOLUME_SHAPE
    mask = torch.zeros(1, X, Y, Z)
    for x in well_positions_x():
        mask[0, x, y_slice, :] = 1.0
    return mask


def build_yz_mask(x_slice):
    """3 vertical wells at x=x_slice, y={16,32,48}, full z."""
    X, Y, Z = VOLUME_SHAPE
    mask = torch.zeros(1, X, Y, Z)
    for y in well_positions_y():
        mask[0, x_slice, y, :] = 1.0
    return mask


# ------------------------- real-cube fetch -----------------------------------

def fetch_real_cubes(data_dir: Path, picks, cont_min, cont_max):
    """Return one real (1, X, Y, Z) cube per layer type, from the row whose
    cond vector matches Figure 2 (and Figure 1)."""
    train_set = ReservoirDataset(data_dir, split='train',
                                 cont_min=cont_min, cont_max=cont_max)
    test_set = ReservoirDataset(data_dir, split='test',
                                cont_min=cont_min, cont_max=cont_max)
    real_cubes = {}
    for _, lt in COLUMN_ORDER:
        info = picks[lt]
        shard_dir, sample_str = info['idx_info'].split('#')
        sample_idx = int(sample_str)
        ds = train_set if info['source'] == 'fig1-train' else test_set

        target_shard = ds.shard_dirs.index(shard_dir)
        hits = np.where(
            (ds.layer_idx == LAYER_TYPE_TO_IDX[lt])
            & (ds.shard_keys == target_shard)
            & (ds.sample_idx == sample_idx))[0]
        if len(hits) == 0:
            raise RuntimeError(
                f'cube not found in {info["source"]} for {lt}: '
                f'{shard_dir}#{sample_idx}')
        facies, _ = ds[int(hits[0])]
        real_cubes[lt] = facies  # (1, X, Y, Z) tensor in {-1, 1}
    return real_cubes


# ------------------------- sampling pipeline ---------------------------------

ROW_KEYS = ['xy', 'xz', 'yz']


def generate_all_cubes(picks, real_cubes, ckpt, device, *,
                       seed_base, z_slice, y_slice, x_slice, model=None):
    """Generate 24 cubes = 8 layer types x 3 well configurations.

    Returns:
        cubes[row_key][lt] -> (X, Y, Z) np.float32 in {-1, +1} (post-paste).
    """
    if model is None:
        print(f'Loading checkpoint -> {ckpt}', flush=True)
        model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                       num_time_embs=1, expand_angle_idx=None).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
    method = FlowMatching(model)

    X, Y, Z = VOLUME_SHAPE
    masks = {
        'xy': build_xy_mask(z_slice).to(device).unsqueeze(0),
        'xz': build_xz_mask(y_slice).to(device).unsqueeze(0),
        'yz': build_yz_mask(x_slice).to(device).unsqueeze(0),
    }

    cubes = {k: {} for k in ROW_KEYS}
    for ri, row_key in enumerate(ROW_KEYS):
        mask = masks[row_key]
        for ci, (_, lt) in enumerate(COLUMN_ORDER):
            cond = torch.from_numpy(picks[lt]['cond']).float().unsqueeze(0).to(device)
            real = real_cubes[lt].to(device).unsqueeze(0)   # (1, 1, X, Y, Z)
            known = real * mask                              # (1, 1, X, Y, Z)
            with torch.no_grad():
                model.set_inpaint_context(mask, known)
                torch.manual_seed(seed_base + 1000 * ri + ci)
                s = method.sample((1, 1, X, Y, Z), device,
                                  cond=cond, cfg_scale=CFG, n_steps=N_STEPS)
                s = apply_inpaint_output(s, mask, known)
                model.clear_inpaint_context()
            cube = s[0, 0].cpu().numpy().astype(np.float32)
            cubes[row_key][lt] = cube
            ntg = float((cube > 0).mean())
            print(f'  [{ri+1}/3 {row_key:<2s}] '
                  f'[{ci+1}/{len(COLUMN_ORDER)}] {lt:<28s}  '
                  f'NTG={ntg:.3f}', flush=True)
    return cubes


def bernoulli_entropy(p, eps=1e-6):
    """Per-voxel Bernoulli entropy in bits. Matches big_reservoir/.../wells_entropy.py."""
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def generate_entropy_maps(picks, real_cubes, ckpt, device, *,
                           seed_base, n_real, batch_size,
                           z_slice, y_slice, x_slice, model=None):
    """For each (row_key, layer_type), sample `n_real` well-conditioned FM
    realizations (batched), accumulate per-voxel sand probability, and
    convert to Bernoulli entropy in bits. Returns dict[row_key][lt] -> {
        'p_mean': (X, Y, Z) np.float32,  -> mean sand probability in [0, 1]
        'entropy': (X, Y, Z) np.float32,  -> Bernoulli entropy in bits [0, 1]
    }."""
    if model is None:
        print(f'Loading checkpoint -> {ckpt}', flush=True)
        model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                       num_time_embs=1, expand_angle_idx=None).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
    method = FlowMatching(model)

    X, Y, Z = VOLUME_SHAPE
    masks_3d = {
        'xy': build_xy_mask(z_slice).to(device),    # (1, X, Y, Z)
        'xz': build_xz_mask(y_slice).to(device),
        'yz': build_yz_mask(x_slice).to(device),
    }

    out = {k: {} for k in ROW_KEYS}
    for ri, row_key in enumerate(ROW_KEYS):
        mask_single = masks_3d[row_key]                              # (1, X, Y, Z)
        for ci, (_, lt) in enumerate(COLUMN_ORDER):
            real = real_cubes[lt].to(device)                         # (1, X, Y, Z)
            cond_single = torch.from_numpy(picks[lt]['cond']).float().to(device)
            sum_binary = np.zeros(VOLUME_SHAPE, dtype=np.float64)
            n_done = 0
            seed_offset = seed_base + 100000 * ri + 1000 * ci
            while n_done < n_real:
                B = min(batch_size, n_real - n_done)
                cond_batch = cond_single.unsqueeze(0).expand(B, -1)
                mask_batch = mask_single.unsqueeze(0).expand(
                    B, 1, X, Y, Z).contiguous()
                known_batch = (real.unsqueeze(0).expand(
                    B, 1, X, Y, Z) * mask_batch).contiguous()
                with torch.no_grad():
                    model.set_inpaint_context(mask_batch, known_batch)
                    torch.manual_seed(seed_offset + n_done)
                    s = method.sample((B, 1, X, Y, Z), device,
                                      cond=cond_batch, cfg_scale=CFG,
                                      n_steps=N_STEPS)
                    s = apply_inpaint_output(s, mask_batch, known_batch)
                    model.clear_inpaint_context()
                # Binary {0, 1} per voxel; sum across batch -> running count.
                binary = (s > 0).float().sum(dim=(0, 1)).cpu().numpy()
                sum_binary += binary
                n_done += B
            p_mean = (sum_binary / n_real).astype(np.float32)
            entropy = bernoulli_entropy(p_mean).astype(np.float32)
            out[row_key][lt] = {'p_mean': p_mean, 'entropy': entropy}
            print(f'  [{ri+1}/3 {row_key:<2s}] '
                  f'[{ci+1}/{len(COLUMN_ORDER)}] {lt:<28s}  '
                  f'mean H = {float(entropy.mean()):.3f} bits  '
                  f'p in [{float(p_mean.min()):.3f}, {float(p_mean.max()):.3f}]',
                  flush=True)
    return out


def load_or_generate_entropy(cache_path, picks, real_cubes, ckpt, device,
                              *, seed_base, n_real, batch_size,
                              z_slice, y_slice, x_slice, force):
    if cache_path.exists() and not force:
        print(f'Loading cached entropy -> {cache_path}', flush=True)
        d = np.load(cache_path, allow_pickle=True)
        out = {k: {} for k in ROW_KEYS}
        ok = True
        for row_key in ROW_KEYS:
            for _, lt in COLUMN_ORDER:
                k_ent = f'{row_key}__entropy__{lt.replace(":", "_")}'
                k_p = f'{row_key}__pmean__{lt.replace(":", "_")}'
                if k_ent not in d.files or k_p not in d.files:
                    print(f'  cache missing {k_ent}; regen')
                    ok = False
                    break
                out[row_key][lt] = {'entropy': d[k_ent], 'p_mean': d[k_p]}
            if not ok:
                break
        if ok:
            return out

    out = generate_entropy_maps(
        picks, real_cubes, ckpt, device,
        seed_base=seed_base, n_real=n_real, batch_size=batch_size,
        z_slice=z_slice, y_slice=y_slice, x_slice=x_slice)
    save_kwargs = {}
    for row_key in ROW_KEYS:
        for lt, m in out[row_key].items():
            key = lt.replace(':', '_')
            save_kwargs[f'{row_key}__entropy__{key}'] = m['entropy']
            save_kwargs[f'{row_key}__pmean__{key}'] = m['p_mean']
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **save_kwargs)
    print(f'Cached entropy -> {cache_path}', flush=True)
    return out


def load_or_generate(cache_path, picks, real_cubes, ckpt, device,
                      *, seed_base, z_slice, y_slice, x_slice, force):
    if cache_path.exists() and not force:
        print(f'Loading cached -> {cache_path}', flush=True)
        d = np.load(cache_path, allow_pickle=True)
        cubes = {k: {} for k in ROW_KEYS}
        ok = True
        for row_key in ROW_KEYS:
            for _, lt in COLUMN_ORDER:
                key = f'{row_key}__{lt.replace(":", "_")}'
                if key not in d.files:
                    print(f'  cache missing {key}; regen')
                    ok = False
                    break
                cubes[row_key][lt] = d[key]
            if not ok:
                break
        if ok:
            return cubes

    cubes = generate_all_cubes(picks, real_cubes, ckpt, device,
                               seed_base=seed_base,
                               z_slice=z_slice, y_slice=y_slice, x_slice=x_slice)
    save_kwargs = {}
    for row_key in ROW_KEYS:
        for lt, c in cubes[row_key].items():
            save_kwargs[f'{row_key}__{lt.replace(":", "_")}'] = c
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **save_kwargs)
    print(f'Cached -> {cache_path}', flush=True)
    return cubes


# ------------------------------ rendering ------------------------------------

def add_well_outlines(ax, row_key, *, z_slice, y_slice, x_slice, vector=False):
    """Add black outline rectangles around well footprints.

    For `imshow` (pixel-center convention) the voxel at index i spans
    [i-0.5, i+0.5], so outline rectangles use `(x-0.5, -0.5)`. For
    `pcolormesh` (cell-edge convention) the voxel at index i spans
    [i, i+1], so outline rectangles use `(x, 0)`. matches big_reservoir
    overlay style.
    """
    SX, SY, SZ = VOLUME_SHAPE
    x_off = 0.0 if vector else -0.5
    y_off = 0.0 if vector else -0.5
    if row_key == 'xy':
        for x in well_positions_x():
            ax.add_patch(Rectangle((x + x_off, y_off), 1, SY, fill=False,
                                   edgecolor=WELL_COLOR, linewidth=WELL_LW,
                                   alpha=WELL_ALPHA, zorder=6))
    elif row_key == 'xz':
        for x in well_positions_x():
            ax.add_patch(Rectangle((x + x_off, y_off), 1, SZ, fill=False,
                                   edgecolor=WELL_COLOR, linewidth=WELL_LW,
                                   alpha=WELL_ALPHA, zorder=6))
    elif row_key == 'yz':
        for y in well_positions_y():
            ax.add_patch(Rectangle((y + x_off, y_off), 1, SZ, fill=False,
                                   edgecolor=WELL_COLOR, linewidth=WELL_LW,
                                   alpha=WELL_ALPHA, zorder=6))


def _draw_panel(ax, img, *, cmap, norm=None, vmin=None, vmax=None,
                 vector=False):
    """Render a 2D slice. `vector=True` uses pcolormesh with rasterized=False
    (true-vector cells, sharp at any zoom but slow / large PDFs).
    `vector=False` uses imshow (rasterized bitmap)."""
    if not vector:
        return imshow_slice(ax, img, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    H, W = img.shape
    xs = np.linspace(0, W, W + 1)
    ys = np.linspace(0, H, H + 1)
    kw = dict(cmap=cmap, shading='flat', edgecolors='none', linewidth=0,
              antialiased=False, rasterized=False)
    if norm is not None:
        kw['norm'] = norm
    else:
        kw['vmin'] = vmin
        kw['vmax'] = vmax
    artist = ax.pcolormesh(xs, ys, img, **kw)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def render_figure(picks, cubes, entropy_maps, out_png, out_pdf=None, *,
                   z_idx=None, y_idx=None, x_idx=None, vector=False):
    """6 rows x 8 cols.
        Rows 0-2: facies XY/XZ/YZ from cubes[row_key].
        Rows 3-5: Bernoulli entropy XY/XZ/YZ from entropy_maps[row_key]
                   (per-voxel uncertainty across N realizations).
    """
    n_cols = len(COLUMN_ORDER)
    SX, SY, SZ = VOLUME_SHAPE
    z_slice = Z_SLICE_DEFAULT if z_idx is None else int(z_idx)
    y_slice = Y_SLICE_DEFAULT if y_idx is None else int(y_idx)
    x_slice = X_SLICE_DEFAULT if x_idx is None else int(x_idx)

    panel_w = 1.10
    facies_heights = [panel_w, panel_w / 2.0, panel_w / 2.0]
    entropy_heights = [panel_w, panel_w / 2.0, panel_w / 2.0]
    label_col_w = 0.55
    gap_h = 0.18           # vertical space between facies block and entropy block
    fig_w = label_col_w + n_cols * panel_w + 0.20
    fig_h = (0.55 + sum(facies_heights) + gap_h
             + sum(entropy_heights) + 0.55)

    fig = plt.figure(figsize=(fig_w, fig_h))
    master_widths = [label_col_w] + [panel_w] * n_cols
    master_heights = [
        0.45,                        # header
        sum(facies_heights),         # facies block
        gap_h,                       # gap
        sum(entropy_heights),        # entropy block
        0.55,                        # cbar row
    ]
    master_gs = fig.add_gridspec(
        nrows=5, ncols=len(master_widths),
        width_ratios=master_widths,
        height_ratios=master_heights,
        left=0.005, right=0.995, top=0.965, bottom=0.04,
        hspace=0.06, wspace=0.0)

    # Header: column labels (one per layer type).
    for ci, (pretty, _) in enumerate(COLUMN_ORDER):
        ax = fig.add_subplot(master_gs[0, 1 + ci])
        ax.text(0.5, 0.30, pretty, ha='center', va='center',
                fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.set_axis_off()

    # Tick positions differ between imshow (pixel-center) and pcolormesh
    # (cell-edge). For vector mode we use integer cell-edge coordinates;
    # otherwise we use the centre-aligned -0.5 offset as before.
    if vector:
        xt_64, yt_64 = (0, 32, 64), (0, 32, 64)
        yt_32 = (0, 16, 32)
    else:
        xt_64, yt_64 = (-0.5, 31.5, 63.5), (-0.5, 31.5, 63.5)
        yt_32 = (-0.5, 15.5, 31.5)
    row_specs = [
        ('xy', f'XY (z={z_slice})', slice_xy, z_slice,
         xt_64, ('0', '32', '64'),
         yt_64, ('0', '32', '64')),
        ('xz', f'XZ (y={y_slice})', slice_xz, y_slice,
         xt_64, ('0', '32', '64'),
         yt_32, ('0', '16', '32')),
        ('yz', f'YZ (x={x_slice})', slice_yz, x_slice,
         xt_64, ('0', '32', '64'),
         yt_32, ('0', '16', '32')),
    ]

    def _draw_block(block_master_idx, heights, source, *, is_entropy):
        """Draw one 3-row x n_cols block at master_gs[block_master_idx, ...]."""
        sub_gs = master_gs[block_master_idx, 1:].subgridspec(
            nrows=3, ncols=n_cols,
            height_ratios=heights,
            hspace=0.20, wspace=0.10)
        last_im = None
        for ri, (row_key, _, slicer, idx,
                 xticks, xlabels, yticks, ylabels) in enumerate(row_specs):
            for ci, (_, lt) in enumerate(COLUMN_ORDER):
                cube = source[row_key][lt]
                if is_entropy:
                    cube = cube['entropy']
                img = slicer(cube, idx)
                ax = fig.add_subplot(sub_gs[ri, ci])
                if is_entropy:
                    im = _draw_panel(ax, img, cmap='viridis', vmin=0, vmax=1,
                                     vector=vector)
                else:
                    im = _draw_panel(ax, img, cmap=CMAP_FACIES,
                                     norm=NORM_FACIES, vector=vector)
                last_im = im
                add_well_outlines(ax, row_key,
                                  z_slice=z_slice, y_slice=y_slice,
                                  x_slice=x_slice, vector=vector)
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                # X-tick labels only on the bottom row of THIS block.
                if ri == 2:
                    ax.set_xticklabels(xlabels, fontsize=7)
                else:
                    ax.set_xticklabels([])
                ax.set_yticklabels(ylabels, fontsize=7)
                ax.tick_params(axis='both', length=2.5, pad=1.5,
                               direction='in')
                # Axis letter only at the very bottom row (entropy block).
                if is_entropy and ri == 2:
                    ax_letter = 'Y' if slicer is slice_yz else 'X'
                    ax.set_xlabel(ax_letter, fontsize=7, labelpad=1)
        return last_im

    handle_im = _draw_block(1, facies_heights, cubes, is_entropy=False)
    handle_im_entropy = _draw_block(3, entropy_heights, entropy_maps,
                                    is_entropy=True)

    # Row labels for facies block.
    facies_label_sub_gs = master_gs[1, 0].subgridspec(
        nrows=3, ncols=1, height_ratios=facies_heights, hspace=0.20)
    for ri, (_rk, row_label, *_rest) in enumerate(row_specs):
        ax = fig.add_subplot(facies_label_sub_gs[ri, 0])
        ax.set_axis_off()
        ax.text(0.85, 0.5, row_label, ha='right', va='center',
                fontsize=8, fontweight='bold', transform=ax.transAxes)

    # Row labels for entropy block.
    entropy_label_sub_gs = master_gs[3, 0].subgridspec(
        nrows=3, ncols=1, height_ratios=entropy_heights, hspace=0.20)
    for ri, (_rk, row_label, *_rest) in enumerate(row_specs):
        ax = fig.add_subplot(entropy_label_sub_gs[ri, 0])
        ax.set_axis_off()
        ax.text(0.85, 0.5, row_label + '\nentropy',
                ha='right', va='center',
                fontsize=8, fontweight='bold', transform=ax.transAxes)

    # Bottom row: two centred colorbars side by side -- shale/sand and entropy.
    cbar_sub_gs = master_gs[4, 1:].subgridspec(
        nrows=2, ncols=7,
        width_ratios=[1, 3, 1, 1, 1, 3, 1],
        height_ratios=[1, 1],
        hspace=0.0)
    cax_facies = fig.add_subplot(cbar_sub_gs[1, 1])
    cbar = fig.colorbar(handle_im, cax=cax_facies, orientation='horizontal',
                        ticks=[0, 1])
    cbar.set_ticklabels(FACIES_BINARY_LABELS)
    cbar.ax.tick_params(labelsize=7)

    cax_entropy = fig.add_subplot(cbar_sub_gs[1, 5])
    cbar2 = fig.colorbar(handle_im_entropy, cax=cax_entropy,
                         orientation='horizontal', ticks=[0, 0.5, 1])
    cbar2.set_label('entropy (bits)', fontsize=7, labelpad=2)
    cbar2.ax.tick_params(labelsize=7)

    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


# --------------------------------- main --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--cond-stats', default=DEFAULT_COND_STATS)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default=str(NEURIPS_DIR / 'figs'))
    ap.add_argument('--cubes-cache', default=None)
    ap.add_argument('--entropy-cache', default=None)
    ap.add_argument('--regenerate', action='store_true')
    ap.add_argument('--regenerate-entropy', action='store_true',
                    help='ignore entropy cache and re-sample the ensemble')
    ap.add_argument('--seed', type=int, default=SAMPLE_SEED)
    ap.add_argument('--n-real', type=int, default=N_REAL_ENTROPY,
                    help='realizations per (cross-section, layer) for entropy')
    ap.add_argument('--batch-size', type=int, default=ENTROPY_BATCH,
                    help='per-call batch size for entropy ensemble sampling')
    ap.add_argument('--z', type=int, default=None)
    ap.add_argument('--y', type=int, default=None)
    ap.add_argument('--x', type=int, default=None)
    ap.add_argument('--out-name', default='figure3')
    ap.add_argument('--vector', action='store_true',
                    help='render with pcolormesh(rasterized=False) so each '
                         'voxel is a true vector cell (sharp at any zoom; '
                         'slower, much larger PDF). Default uses imshow '
                         '(rasterized bitmap).')
    ap.add_argument('--well-fractions', type=float, nargs='+', default=None,
                    help='well positions as axis fractions (e.g. 0.33 0.66 '
                         'puts two wells at x=21 and x=42 for the XZ row, '
                         'and y=21 and y=42 for the YZ row). Defaults to '
                         'a single well at 0.5.')
    args = ap.parse_args()
    if args.well_fractions is not None:
        global WELL_FRACTIONS
        WELL_FRACTIONS = list(args.well_fractions)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cubes_cache = (Path(args.cubes_cache) if args.cubes_cache is not None
                   else out_dir / f'{args.out_name}_cubes.npz')
    entropy_cache = (Path(args.entropy_cache) if args.entropy_cache is not None
                     else out_dir / f'{args.out_name}_entropy.npz')

    z_slice = Z_SLICE_DEFAULT if args.z is None else int(args.z)
    y_slice = Y_SLICE_DEFAULT if args.y is None else int(args.y)
    x_slice = X_SLICE_DEFAULT if args.x is None else int(args.x)

    print(f'Cond-stats:  {args.cond_stats}')
    stats = np.load(args.cond_stats, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']

    print('Building per-column cond vectors (matched to figs 1 & 2) ...')
    picks = fig2.build_picks(Path(args.data_dir), cont_min, cont_max)
    for _, lt in COLUMN_ORDER:
        p = picks[lt]
        print(f'  {lt:<28s}  {p["source"]:<14s}  {p["idx_info"]}')

    print('Fetching real cubes for well-log conditioning ...')
    real_cubes = fetch_real_cubes(Path(args.data_dir), picks, cont_min, cont_max)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device:      {device}   seed={args.seed}')
    print(f'Slice plan:  XY z={z_slice}  XZ y={y_slice}  YZ x={x_slice}')
    print(f'Wells:       x in {well_positions_x()}, y in {well_positions_y()}')

    cubes = load_or_generate(cubes_cache, picks, real_cubes, args.ckpt, device,
                             seed_base=args.seed,
                             z_slice=z_slice, y_slice=y_slice, x_slice=x_slice,
                             force=args.regenerate)

    print(f'Entropy ensemble: n_real={args.n_real}  '
          f'batch={args.batch_size}  cache={entropy_cache.name}')
    entropy_maps = load_or_generate_entropy(
        entropy_cache, picks, real_cubes, args.ckpt, device,
        seed_base=args.seed + 12345,
        n_real=args.n_real, batch_size=args.batch_size,
        z_slice=z_slice, y_slice=y_slice, x_slice=x_slice,
        force=args.regenerate_entropy)

    out_png = out_dir / f'{args.out_name}.png'
    out_pdf = out_dir / f'{args.out_name}.pdf'
    print(f'Rendering -> {out_png}')
    render_figure(picks, cubes, entropy_maps, out_png, out_pdf,
                  z_idx=z_slice, y_idx=y_slice, x_idx=x_slice,
                  vector=args.vector)
    print(f'Done. {out_png}  /  {out_pdf}')


if __name__ == '__main__':
    main()
