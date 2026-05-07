"""NeurIPS Figure 2: flow-matching-generated single-cube samples for all 8
reservoir architectures, shown as binary-facies XY/XZ/YZ mid-slices.

Layout: 3 rows (XY, XZ, YZ) x 8 cols (one per layer type).

Conditioning source per column:
  - lobe / delta / channel:MEANDER_OXBOW / channel:PV_SHOESTRING:
      figure-1 medoid picks (re-using examples/.../figure1_picks.txt's
      shard_dir + sample_idx, looked up in the TRAIN cond cache) so the
      conditioning matches the real samples shown in Figure 1.
  - channel:CB_JIGSAW / CB_LABYRINTH / SH_DISTAL / SH_PROXIMAL:
      mirrors examples/reservoirs/inpainting/sample.py exactly: open the
      TEST split, np.random.default_rng(0), iterate LAYER_TYPES in canonical
      order, take the first rng.choice() for each — i.e. the same picks the
      inpainting demo uses.

Sampling uses the FM-inpaint checkpoint (in_channels=3) with a zero mask
and zero known-data tensor, so the model runs as a plain conditional
flow-matching generator. Generated cubes are cached to
   figs/figure2_cubes.npz
so re-rendering plot tweaks doesn't re-sample.

Run:
    python examples/reservoirs/paper_figures/figure2.py
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from resflow.assembly import (
    BlockSpec, COND_DIM, LAYER_TYPE_TO_IDX, build_cond_vector,
)
from resflow.methods.flow_matching import FlowMatching
from resflow.models.unet3d import UNet3D
from resflow.utils.data_reservoirs import (
    CONT_COLS, LAYER_TYPES, ReservoirDataset, VOLUME_SHAPE,
)
from resflow.utils.plotting_reservoirs import (
    CMAP_FACIES, NORM_FACIES, FACIES_BINARY_LABELS,
    slice_xy, slice_xz, slice_yz, imshow_slice,
)


VECTOR = False  # set via --vector for true-vector PDF cells.


def _draw_panel(ax, img, *, cmap, norm=None, vmin=None, vmax=None):
    """imshow (rasterized) or pcolormesh (vector) per global VECTOR flag."""
    if not VECTOR:
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


SCRATCH = os.environ.get('SCRATCH', '/tmp')
DEFAULT_CKPT = os.path.join(SCRATCH,
                            'genflows_runs/reservoirs_inpainting/checkpoints/flow_matching.pt')
DEFAULT_COND_STATS = os.path.join(SCRATCH,
                                  'genflows_runs/reservoirs_inpainting/checkpoints/cond_stats.npz')
DEFAULT_DATA_DIR = os.environ.get('RESERVOIR_DATA_DIR',
                                  os.path.join(SCRATCH, 'SiliciclasticReservoirs'))

# Order in which the 8 columns appear in the figure (left -> right). Chosen
# to group similar architectures together visually.
COLUMN_ORDER = [
    ('Lobe',          'lobe'),
    ('Delta',         'delta'),
    ('Meander',       'channel:MEANDER_OXBOW'),
    ('PV\nshoestring','channel:PV_SHOESTRING'),
    ('CB\njigsaw',    'channel:CB_JIGSAW'),
    ('CB\nlabyrinth', 'channel:CB_LABYRINTH'),
    ('SH\nproximal',  'channel:SH_PROXIMAL'),
    ('SH\ndistal',    'channel:SH_DISTAL'),
]

# layer types whose cond comes from the figure-1 picks log (TRAIN split).
FIG1_LAYERS = {'lobe', 'delta',
               'channel:MEANDER_OXBOW', 'channel:PV_SHOESTRING'}

# Explicit TEST-split overrides for layer types where we want a hand-picked
# row instead of the rng=default_rng(0) pick from inpainting/sample.py.
# Empty by default: the rng=default_rng(0) picks already match the
# figure1 SAMPLE_OVERRIDES for the four channel architectures, so Fig 2's
# conds are identical to Fig 1's. Add entries here only if you want to
# override fig1 picks for fig2 (which would desynchronise the two figures).
INPAINT_TEST_OVERRIDES = {}

N_STEPS = 50
CFG = 3.0
SAMPLE_SEED = 7  # default; override via --seed

NEURIPS_DIR = Path(__file__).resolve().parent
PICKS_LOG = NEURIPS_DIR / 'figs' / 'figure1_picks.txt'


# ------------------------------- cond lookup -------------------------------

def parse_picks_log(path: Path):
    """Parse figure1_picks.txt -> dict[layer_type] = (shard_dir, sample_idx)."""
    out = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                cur = None
                continue
            if not line.startswith(' '):
                cur = line.strip()
                out[cur] = {}
            else:
                m = re.match(r'\s*([A-Za-z_]+)\s*=\s*(\S+)', line)
                if m and cur is not None:
                    out[cur][m.group(1)] = m.group(2)
    picks = {}
    for lt, kv in out.items():
        if 'shard_dir' in kv and 'sample_idx' in kv:
            picks[lt] = (kv['shard_dir'], int(kv['sample_idx']))
    return picks


def find_global_idx(cache_data, layer_type, shard_dir, sample_idx):
    """Find the row in the cond cache matching (layer_type, shard_dir, sample_idx)."""
    shard_dirs = list(cache_data['shard_dirs'])
    target_shard = shard_dirs.index(shard_dir)
    target_layer = LAYER_TYPE_TO_IDX[layer_type]
    layer_idx = cache_data['layer_idx']
    sample_arr = cache_data['sample_idx']
    shard_keys = cache_data['shard_keys']
    hit = np.where(
        (layer_idx == target_layer)
        & (shard_keys == target_shard)
        & (sample_arr == sample_idx))[0]
    if len(hit) == 0:
        raise ValueError(
            f'No row for layer_type={layer_type!r} shard={shard_dir!r} '
            f'sample_idx={sample_idx}')
    return int(hit[0])


def cache_row_to_cond_vec(cache_data, global_idx, layer_type, cont_min, cont_max):
    """Pull cont_raw + azimuth from a cond cache row, build the 18-D cond vec."""
    cont_raw = cache_data['cont_raw'][global_idx]
    azimuth = float(cache_data['azimuth'][global_idx])
    raw_scalars = {col: float(cont_raw[k])
                   for k, col in enumerate(CONT_COLS)
                   if not np.isnan(cont_raw[k])}
    spec = BlockSpec(
        layer_idx=LAYER_TYPE_TO_IDX[layer_type],
        azimuth_deg=azimuth,
        raw_scalars=raw_scalars,
    )
    return build_cond_vector(spec, cont_min, cont_max)


# ---------------------------- sampling pipeline ----------------------------

def build_picks(data_dir: Path, cont_min, cont_max):
    """For each of the 8 layer types, return:
        layer_type -> {'cond': np.ndarray (18,), 'source': str, 'idx_info': str}
    """
    train_cache = np.load(data_dir / '_cond_cache' / 'train.npz', allow_pickle=True)

    fig1_picks_raw = parse_picks_log(PICKS_LOG)

    # Mirror inpainting/sample.py: TEST split, default_rng(0) iterated over
    # LAYER_TYPES in canonical order. We need only 4 types' picks (the
    # "rest of channels"), but consume the rng in the same order so the
    # picks are byte-identical to what sample.py would draw.
    test_set = ReservoirDataset(data_dir, split='test',
                                cont_min=cont_min, cont_max=cont_max)
    rng = np.random.default_rng(0)
    inpaint_global_idx = {}
    for li, layer_name in enumerate(LAYER_TYPES):
        hits = np.where(test_set.layer_idx == li)[0]
        if len(hits) == 0:
            continue
        inpaint_global_idx[layer_name] = int(rng.choice(hits))

    # Pull the test cache (already loaded as part of test_set, but re-load
    # the npz so we have direct access to cont_raw / azimuth / shard_dirs).
    test_cache = np.load(data_dir / '_cond_cache' / 'test.npz', allow_pickle=True)

    out = {}
    for _, lt in COLUMN_ORDER:
        if lt in FIG1_LAYERS:
            shard_dir, sample_idx = fig1_picks_raw[lt]
            gi = find_global_idx(train_cache, lt, shard_dir, sample_idx)
            cond_vec = cache_row_to_cond_vec(train_cache, gi, lt,
                                             cont_min, cont_max)
            out[lt] = dict(cond=cond_vec,
                           source='fig1-train',
                           idx_info=f'{shard_dir}#{sample_idx}')
        elif lt in INPAINT_TEST_OVERRIDES:
            shard_dir, sample_idx = INPAINT_TEST_OVERRIDES[lt]
            gi = find_global_idx(test_cache, lt, shard_dir, sample_idx)
            cond_vec = cache_row_to_cond_vec(test_cache, gi, lt,
                                             cont_min, cont_max)
            out[lt] = dict(cond=cond_vec,
                           source='override-test',
                           idx_info=f'{shard_dir}#{sample_idx}')
        else:
            gi = inpaint_global_idx[lt]
            shard_dir = list(test_cache['shard_dirs'])[
                int(test_cache['shard_keys'][gi])]
            sample_idx = int(test_cache['sample_idx'][gi])
            cond_vec = cache_row_to_cond_vec(test_cache, gi, lt,
                                             cont_min, cont_max)
            out[lt] = dict(cond=cond_vec,
                           source='inpaint-test',
                           idx_info=f'{shard_dir}#{sample_idx}')
    return out


def generate_cubes(picks: dict, ckpt: str, device: torch.device,
                   *, seed: int, model=None):
    """Run FM-inpaint as a plain conditional generator (zero mask + zero known
    data) to produce one binary cube per layer type.

    If `model` is given (already-loaded UNet3D + Method instance), reuse it.
    Otherwise load the checkpoint here. Reuse is useful when sweeping seeds
    so the model load + GPU warmup is amortised across runs.

    Returns dict[layer_type] = np.float32 cube of shape VOLUME_SHAPE."""
    if model is None:
        print(f'Loading checkpoint -> {ckpt}', flush=True)
        model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                       num_time_embs=1, expand_angle_idx=None).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
    method = FlowMatching(model)

    X, Y, Z = VOLUME_SHAPE
    zero_mask = torch.zeros(1, 1, X, Y, Z, device=device)
    zero_data = torch.zeros_like(zero_mask)

    cubes = {}
    for i, (_, lt) in enumerate(COLUMN_ORDER):
        cond = torch.from_numpy(picks[lt]['cond']).float().unsqueeze(0).to(device)
        with torch.no_grad():
            model.set_inpaint_context(zero_mask, zero_data)
            torch.manual_seed(seed + i)
            sample = method.sample((1, 1, X, Y, Z), device,
                                   cond=cond, cfg_scale=CFG, n_steps=N_STEPS)
            model.clear_inpaint_context()
        cube = sample[0, 0].cpu().numpy().astype(np.float32)
        cubes[lt] = cube
        ntg = float((cube > 0).mean())
        print(f'  [{i+1}/{len(COLUMN_ORDER)}] {lt:<28s}  NTG={ntg:.3f}',
              flush=True)
    return cubes


def load_or_generate_cubes(cache_path: Path, picks: dict, ckpt: str,
                            device: torch.device, force: bool, *,
                            seed: int):
    """Cache cubes to a single .npz so re-renders are fast."""
    if cache_path.exists() and not force:
        print(f'Loading cached cubes -> {cache_path}', flush=True)
        d = np.load(cache_path, allow_pickle=True)
        cubes = {}
        for _, lt in COLUMN_ORDER:
            key = lt.replace(':', '_')
            if key not in d.files:
                print(f'  cache missing {lt!r}; regenerating')
                return None
            cubes[lt] = d[key]
        return cubes

    if not torch.cuda.is_available():
        print('No CUDA available; sampling on CPU will be slow.')
    cubes = generate_cubes(picks, ckpt, device, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path,
             **{lt.replace(':', '_'): cube for lt, cube in cubes.items()})
    print(f'Cached cubes -> {cache_path}', flush=True)
    return cubes


# ------------------------------- rendering --------------------------------

def render_figure(picks, cubes, out_png, out_pdf=None, *,
                   z_idx=None, y_idx=None, x_idx=None):
    """3 rows (XY/XZ/YZ) x 8 cols (per layer type), binary-facies cmap.

    The z/y/x slice indices for the XY/XZ/YZ rows default to the midpoint
    of each axis; pass z_idx/y_idx/x_idx to override. The row labels show
    the actual slice index used.
    """
    n_cols = len(COLUMN_ORDER)
    SX, SY, SZ = VOLUME_SHAPE
    # Canonical z=18 for the XY row matches figs 1 & 3.
    z_mid = 18 if z_idx is None else int(z_idx)
    y_mid = SY // 2 if y_idx is None else int(y_idx)
    x_mid = SX // 2 if x_idx is None else int(x_idx)

    # Geometry. Each column is panel_w wide. Row 0 is square (XY 64x64);
    # rows 1 & 2 are half-height (XZ/YZ are 64x32, so axes height = panel_w/2).
    panel_w = 1.10
    row_heights_in = [panel_w, panel_w / 2.0, panel_w / 2.0]
    label_col_w = 0.55          # left strip with row labels
    sep_w = 0.0                  # no field-group separators (single field)
    fig_w = label_col_w + n_cols * panel_w + 0.20
    fig_h = 0.55 + sum(row_heights_in) + 0.55  # header + rows + cbar

    fig = plt.figure(figsize=(fig_w, fig_h))
    master_widths = [label_col_w] + [panel_w] * n_cols
    master_heights = [0.45, sum(row_heights_in), 0.45]
    master_gs = fig.add_gridspec(
        nrows=3, ncols=len(master_widths),
        width_ratios=master_widths,
        height_ratios=master_heights,
        left=0.005, right=0.995, top=0.965, bottom=0.04,
        hspace=0.06, wspace=0.0)

    # --- Header row: column labels (one per layer type) ---
    for ci, (pretty, _) in enumerate(COLUMN_ORDER):
        ax = fig.add_subplot(master_gs[0, 1 + ci])
        ax.text(0.5, 0.30, pretty, ha='center', va='center',
                fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.set_axis_off()

    # --- Data: 3-row x n_cols sub-gridspec, sharing height_ratios so XY is
    # tall and XZ/YZ are half-height. wspace > 0 leaves room for the y-tick
    # labels we now show on every panel (not just the leftmost). ---
    data_gs = master_gs[1, 1:].subgridspec(
        nrows=3, ncols=n_cols,
        height_ratios=row_heights_in,
        hspace=0.20, wspace=0.10)

    if VECTOR:
        xt_64 = (0, 32, 64); yt_64 = (0, 32, 64); yt_32 = (0, 16, 32)
    else:
        xt_64 = (-0.5, 31.5, 63.5)
        yt_64 = (-0.5, 31.5, 63.5)
        yt_32 = (-0.5, 15.5, 31.5)
    row_specs = [
        (f'XY (z={z_mid})', slice_xy, z_mid, ('X', 'Y'),
         xt_64, ('0', '32', '64'),
         yt_64, ('0', '32', '64')),
        (f'XZ (y={y_mid})', slice_xz, y_mid, ('X', 'Z'),
         xt_64, ('0', '32', '64'),
         yt_32, ('0', '16', '32')),
        (f'YZ (x={x_mid})', slice_yz, x_mid, ('Y', 'Z'),
         xt_64, ('0', '32', '64'),
         yt_32, ('0', '16', '32')),
    ]

    handle_im = None
    for ri, (row_label, slicer, idx, _, xticks, xlabels,
             yticks, ylabels) in enumerate(row_specs):
        for ci, (_, lt) in enumerate(COLUMN_ORDER):
            cube = cubes[lt]
            img = slicer(cube, idx)
            ax = fig.add_subplot(data_gs[ri, ci])
            im = _draw_panel(ax, img, cmap=CMAP_FACIES, norm=NORM_FACIES)
            handle_im = im
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            # Tick numbers everywhere: x-ticks on the bottom row, y-ticks
            # on every panel. Axis label (X/Y) only at the very bottom row,
            # mirroring figure 1's "label at the bottom only" convention.
            if ri == 2:
                ax.set_xticklabels(xlabels, fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.set_yticklabels(ylabels, fontsize=7)
            ax.tick_params(axis='both', length=2.5, pad=1.5, direction='in')
            if ri == 2:
                ax_letter = 'Y' if slicer is slice_yz else 'X'
                ax.set_xlabel(ax_letter, fontsize=7, labelpad=1)

    # --- Row labels in master_gs col 0 (one per data row) ---
    label_sub_gs = master_gs[1, 0].subgridspec(
        nrows=3, ncols=1, height_ratios=row_heights_in, hspace=0.20)
    for ri, (row_label, *_rest) in enumerate(row_specs):
        ax = fig.add_subplot(label_sub_gs[ri, 0])
        ax.set_axis_off()
        ax.text(0.85, 0.5, row_label, ha='right', va='center',
                fontsize=8, fontweight='bold', transform=ax.transAxes)

    # --- Bottom row: centred shale/sand colorbar with explicit padding ---
    cbar_sub_gs = master_gs[2, 1:].subgridspec(
        nrows=2, ncols=5,
        width_ratios=[1, 1, 4, 1, 1],
        height_ratios=[1, 1],
        hspace=0.0)
    cax = fig.add_subplot(cbar_sub_gs[1, 2])
    cbar = fig.colorbar(handle_im, cax=cax, orientation='horizontal',
                        ticks=[0, 1])
    cbar.set_ticklabels(FACIES_BINARY_LABELS)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


# --------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--cond-stats', default=DEFAULT_COND_STATS)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default=str(NEURIPS_DIR / 'figs'))
    ap.add_argument('--cubes-cache', default=None,
                    help='path to cubes .npz cache; defaults to '
                         'figs/figure2_cubes.npz (or seeds/seed_NN/figure2_cubes.npz '
                         'when --seed is non-default).')
    ap.add_argument('--regenerate', action='store_true',
                    help='ignore cubes cache and re-sample from the model')
    ap.add_argument('--seed', type=int, default=SAMPLE_SEED,
                    help='base sampling seed; the i-th column uses (seed + i)')
    ap.add_argument('--out-name', default='figure2',
                    help='base name for output png/pdf (default: figure2)')
    ap.add_argument('--z', type=int, default=None,
                    help='Z slice index for the XY row (0..31, default: 16)')
    ap.add_argument('--y', type=int, default=None,
                    help='Y slice index for the XZ row (0..63, default: 32)')
    ap.add_argument('--x', type=int, default=None,
                    help='X slice index for the YZ row (0..63, default: 32)')
    ap.add_argument('--vector', action='store_true',
                    help='render with pcolormesh(rasterized=False) so each '
                         'voxel is a true vector cell (sharp at any zoom; '
                         'much larger PDF). Default uses imshow (rasterized).')
    args = ap.parse_args()
    global VECTOR
    VECTOR = args.vector

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cubes_cache is not None:
        cubes_cache = Path(args.cubes_cache)
    else:
        cubes_cache = out_dir / f'{args.out_name}_cubes.npz'

    print(f'Cond-stats:  {args.cond_stats}')
    stats = np.load(args.cond_stats, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']

    print('Building per-column cond vectors ...')
    picks = build_picks(Path(args.data_dir), cont_min, cont_max)
    for _, lt in COLUMN_ORDER:
        p = picks[lt]
        print(f'  {lt:<28s}  {p["source"]:<14s}  {p["idx_info"]}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device:      {device}   seed={args.seed}')
    cubes = load_or_generate_cubes(cubes_cache, picks, args.ckpt,
                                    device, args.regenerate, seed=args.seed)
    if cubes is None:
        cubes = generate_cubes(picks, args.ckpt, device, seed=args.seed)
        np.savez(cubes_cache,
                 **{lt.replace(':', '_'): cube for lt, cube in cubes.items()})

    out_png = out_dir / f'{args.out_name}.png'
    out_pdf = out_dir / f'{args.out_name}.pdf'
    print(f'Rendering -> {out_png}')
    render_figure(picks, cubes, out_png, out_pdf,
                  z_idx=args.z, y_idx=args.y, x_idx=args.x)
    print(f'Done. {out_png}  /  {out_pdf}')


if __name__ == '__main__':
    main()
