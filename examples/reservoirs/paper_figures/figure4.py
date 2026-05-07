"""NeurIPS Figure 4: big-reservoir multi-diffusion samples (overlap=24).

Reuses the existing big_reservoir/<type>/visualize.py rendering convention:
pcolormesh background, block borders (red vertical = X-block edges, blue
horizontal = Y-block edges), and overlap shading (red column-strips for X
overlaps, cyan row-strips for Y overlaps). The only change versus that style
is the colour map -- we use a canonical earth-science scheme (grey for mud,
burnt-orange for sand) discrete cmap from `resflow.utils.plotting_reservoirs`.

Axis convention matches visualize.py: X is the horizontal axis, Y is the
vertical axis. The cube tensor (X, Y, Z) is shown via
`binary[:, :, z].T` + `extent=[0, Tx, 0, Ty]`, which makes the data dim 0
(X) the horizontal extent and dim 1 (Y) the vertical extent.

Layout:
    +-----------+--+--+--+--+--+--+
    |           |c1|c2|c3|c4|c5|c6|
    |   LOBE    |  |  |  |  |  |  |
    |  424x424  |  |  |  |  |  |  |
    |           |  |  |  |  |  |  |
    +-----------+--+--+--+--+--+--+

The 6 channel sub-panels are 64 (X) wide x 424 (Y) tall each, so each is a
tall vertical strip and the whole right block has the same vertical extent
as the square lobe.

Run:
    python examples/reservoirs/paper_figures/figure4.py
    # output -> examples/reservoirs/paper_figures/figs/figure4.{png,pdf}
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from resflow.utils.plotting_reservoirs import (
    CMAP_FACIES, NORM_FACIES, FACIES_BINARY_LABELS,
)


LOBE_DIR = ('Lobe', 'lobes')
# Pretty names use \n line breaks so they render horizontally on multiple
# lines above the narrow channel strips (instead of being rotated/tilted).
CHANNEL_DIRS = [
    # Order matches figs 1/2/3 (delta omitted -- multi-diffusion has no delta).
    ('Meander',        'meander_oxbow_long'),
    ('PV\nshoestring', 'pv_shoestring_long'),
    ('CB\njigsaw',     'cb_jigsaw_long'),
    ('CB\nlabyrinth',  'cb_labyrinth_long'),
    ('SH\nproximal',   'sh_proximal_long'),
    ('SH\ndistal',     'sh_distal_long'),
]

# Canonical Figure-2 source: the seed-11 multi-diffusion run produced by
# gen_seeds.py. The original big_reservoir/ tree (seed 42) remains a valid
# alternative if --results-root is overridden.
DEFAULT_RES_ROOT = (Path(__file__).resolve().parent
                    / 'gen_seeds' / 'seed_11')
RESULT_FILE = 'results/reservoir_hard_ov24.npz'
Z_SLICE = 16   # mid of 32

# Set at the start of main() and used by load_run().
RESULTS_ROOT = DEFAULT_RES_ROOT


# --- Helpers copied verbatim from big_reservoir/lobes/visualize.py so that
#     the look (overlap shading + block borders) matches that script. -----

def block_origins(n, S, overlap):
    stride = S - overlap
    return [k * stride for k in range(n)]


OVERLAP_COLOR = 'blue'


def shade_overlaps(ax, run, axis, lo, hi):
    """Mark overlap bands with diagonal hatching in OVERLAP_COLOR. Forward
    slashes for X overlaps, back slashes for Y overlaps so the two
    directions remain visually distinguishable where they cross (e.g. in
    the 10x10 lobe grid). Both use the same colour for a unified look."""
    Sx, Sy, _ = run['block_shape']
    n = run['nx'] if axis == 'x' else run['ny']
    S = Sx if axis == 'x' else Sy
    overlap = run['overlap']
    if overlap <= 0:
        return
    origins = block_origins(n, S, overlap)
    hatch = '///' if axis == 'x' else '\\\\\\'
    for k in range(n - 1):
        a_lo = origins[k + 1]
        a_hi = origins[k] + S
        if axis == 'x':
            ax.add_patch(Rectangle((a_lo, lo), a_hi - a_lo, hi - lo,
                                   facecolor='none', edgecolor=OVERLAP_COLOR,
                                   hatch=hatch, linewidth=0,
                                   alpha=0.25, zorder=3))
        else:
            ax.add_patch(Rectangle((lo, a_lo), hi - lo, a_hi - a_lo,
                                   facecolor='none', edgecolor=OVERLAP_COLOR,
                                   hatch=hatch, linewidth=0,
                                   alpha=0.25, zorder=3))


def draw_borders(ax, run, draw_x=True, draw_y=True):
    """Block-boundary lines in OVERLAP_COLOR (so the borders defining the
    overlap region match the overlap shading)."""
    Sx, Sy, _ = run['block_shape']
    if draw_x:
        for x0 in block_origins(run['nx'], Sx, run['overlap']):
            ax.axvline(x0, color=OVERLAP_COLOR, lw=0.5, alpha=0.30, zorder=4)
            ax.axvline(x0 + Sx, color=OVERLAP_COLOR, lw=0.5, alpha=0.30, zorder=4)
    if draw_y:
        for y0 in block_origins(run['ny'], Sy, run['overlap']):
            ax.axhline(y0, color=OVERLAP_COLOR, lw=0.5, alpha=0.30, zorder=4)
            ax.axhline(y0 + Sy, color=OVERLAP_COLOR, lw=0.5, alpha=0.30, zorder=4)


VECTOR = False  # set to True via --vector for true-vector PDF cells.


def _pcm(ax, arr_2d, extent):
    """pcolormesh wrapper. By default the background is rasterised
    (rasterized=True) so the PDF stays small. With VECTOR=True each cell
    becomes a true vector quad — sharp at any zoom but slower & much
    larger PDFs."""
    ny, nx = arr_2d.shape
    x_lo, x_hi, y_lo, y_hi = extent
    xs = np.linspace(x_lo, x_hi, nx + 1)
    ys = np.linspace(y_lo, y_hi, ny + 1)
    pcm = ax.pcolormesh(xs, ys, arr_2d, cmap=CMAP_FACIES, norm=NORM_FACIES,
                        shading='flat', edgecolors='none', linewidth=0,
                        antialiased=False, rasterized=not VECTOR)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    return pcm


def load_run(subdir):
    p = RESULTS_ROOT / subdir / RESULT_FILE
    if not p.exists():
        raise FileNotFoundError(f'missing {p}')
    d = np.load(p, allow_pickle=True)
    return {
        'binary': d['binary'],
        'overlap': int(d['overlap']),
        'block_shape': tuple(int(s) for s in d['block_shape']),
        'ny': int(d['ny']),
        'nx': int(d['nx']),
    }


def render_panel(ax, run, z, *, title=None, title_rotation=0,
                 show_xlabel=True, show_ylabel=True, show_yticklabels=True):
    """Render one XY panel at z = `z` with overlap shading and block borders.
    binary[:, :, z].T puts Y on rows, X on cols, so extent=[0, Tx, 0, Ty]
    gives X horizontal and Y vertical (matching visualize.py)."""
    binary = run['binary']
    Tx, Ty, _ = binary.shape
    _pcm(ax, binary[:, :, z].T, [0, Tx, 0, Ty])
    shade_overlaps(ax, run, 'x', 0, Ty)
    shade_overlaps(ax, run, 'y', 0, Tx)
    draw_borders(ax, run)
    if title is not None:
        if title_rotation:
            # Place rotated text above the panel via ax.text rather than
            # set_title (set_title doesn't honour rotation cleanly).
            ax.text(0.5, 1.01, title,
                    transform=ax.transAxes,
                    ha='center', va='bottom',
                    rotation=title_rotation, rotation_mode='anchor',
                    fontsize=14, fontweight='bold')
        else:
            ax.set_title(title, fontsize=14, fontweight='bold')
    if show_xlabel:
        ax.set_xlabel('X', fontsize=12)
    if show_ylabel:
        ax.set_ylabel('Y', fontsize=12)
    else:
        ax.set_ylabel('')
    if not show_yticklabels:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', labelsize=12)


def main():
    global RESULTS_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir',
                    default=str(Path(__file__).resolve().parent / 'figs'))
    ap.add_argument('--z', type=int, default=Z_SLICE)
    ap.add_argument('--results-root',
                    default=str(DEFAULT_RES_ROOT),
                    help='Directory holding <type>/results/reservoir_hard_ov24.npz '
                         'for each of the 7 reservoir types.')
    ap.add_argument('--suffix-z', action='store_true',
                    help='Append _zNN to the output filename (used by gen_seeds.py '
                         'when sweeping z; off by default so the canonical '
                         'figure stays at figs/figure4.png).')
    ap.add_argument('--vector', action='store_true',
                    help='Render with rasterized=False so each pcolormesh '
                         'cell is a true vector quad in the PDF (sharp at '
                         'any zoom, much larger files).')
    args = ap.parse_args()
    global VECTOR
    VECTOR = args.vector
    RESULTS_ROOT = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading binary cubes (overlap=24) for 1 lobe + {len(CHANNEL_DIRS)} channels ...')
    lobe_run = load_run(LOBE_DIR[1])
    print(f'  {LOBE_DIR[0]:<28s} binary shape={lobe_run["binary"].shape}  '
          f'block={lobe_run["block_shape"]}  ny={lobe_run["ny"]} nx={lobe_run["nx"]}')
    channel_runs = []
    for pretty, sub in CHANNEL_DIRS:
        r = load_run(sub)
        channel_runs.append((pretty, r))
        print(f'  {pretty:<28s} binary shape={r["binary"].shape}  '
              f'block={r["block_shape"]}  ny={r["ny"]} nx={r["nx"]}')

    # Figure geometry: lobe is 424x424 square; each channel is 64x424 strip.
    Tx_lobe, Ty_lobe, _ = lobe_run['binary'].shape
    Tx_ch, Ty_ch, _ = channel_runs[0][1]['binary'].shape
    n_ch = len(channel_runs)

    # Geometry. We pin the COMMON panel height (in inches) and derive every
    # panel width from aspect='equal' so the lobe and every channel strip
    # have exactly the same vertical extent on the rendered figure. The
    # channel y-axis is hidden (ticks + label) since the lobe's y-axis
    # already labels the same Y range.
    panel_h_in = 7.5                              # tall enough that channel
                                                  #   strips read well
    lobe_w_in = panel_h_in * Tx_lobe / Ty_lobe    # square -> 7.5
    ch_w_in = panel_h_in * Tx_ch / Ty_ch          # 7.5 / 6.625 ~= 1.13
    ch_wspace = 0.10                               # gap between channel cells
    stack_w_in = n_ch * ch_w_in + (n_ch - 1) * ch_wspace
    fig_w = lobe_w_in + 1.0 + stack_w_in + 0.5    # extra gap before stack
                                                   #   so channel titles don't
                                                   #   collide with lobe
    fig_h = panel_h_in + 1.1                       # compact bottom margin
                                                   #   (just enough for cbar +
                                                   #    overlap legend)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[lobe_w_in, stack_w_in],
        left=0.06, right=0.985, top=0.93, bottom=0.12,
        wspace=0.08)                               # tight enough that channels
                                                    #   sit close to the lobe

    # Lobe panel
    ax_lobe = fig.add_subplot(gs[0, 0])
    render_panel(ax_lobe, lobe_run, args.z,
                 title=f'{LOBE_DIR[0]}  ({Tx_lobe}×{Ty_lobe})')

    # Channel stack: 1 row x n_ch cols, each panel a vertical strip. Share
    # the y-axis so they all show the same Y range (0..Ty_ch=424). Hide
    # y-tick labels on every panel except the leftmost so the axis isn't
    # cluttered.
    ch_gs = gs[0, 1].subgridspec(nrows=1, ncols=n_ch, wspace=ch_wspace)
    axes_ch = []
    for i, (pretty, run) in enumerate(channel_runs):
        ax = fig.add_subplot(ch_gs[0, i])
        render_panel(ax, run, args.z,
                     title=pretty,                    # horizontal multi-line
                     show_xlabel=True,
                     show_ylabel=False,                # lobe already labels Y
                     show_yticklabels=False)
        # Keep tick MARKS on the y-axis (they share the lobe's grid steps);
        # only the tick LABELS are hidden via show_yticklabels=False above.
        axes_ch.append(ax)

    # No suptitle -- figure caption in the paper carries that information.

    # --- Bottom row: colorbar (shale / sand) + hatched-overlap legend,
    # centred horizontally across the whole figure and tucked close to the
    # panels to keep the bottom margin compact.
    lobe_pos = ax_lobe.get_position()
    cbar_w = 0.18
    cbar_h = 0.022
    legend_w = 0.18
    legend_gap = 0.025
    total_w = cbar_w + legend_gap + legend_w
    bottom = lobe_pos.y0 - 0.085
    cbar_left = 0.5 - total_w / 2
    cax = fig.add_axes([cbar_left, bottom, cbar_w, cbar_h])
    cbar = fig.colorbar(ax_lobe.get_images()[0] if ax_lobe.get_images()
                        else ax_lobe.collections[0],
                        cax=cax, orientation='horizontal',
                        ticks=[0, 1])
    cbar.set_ticklabels(FACIES_BINARY_LABELS)
    cbar.ax.tick_params(labelsize=12)

    # Legend axes height matches the cbar height exactly, with the same
    # bottom edge, so the swatches sit on the same level as the cbar.
    legend_left = cbar_left + cbar_w + legend_gap
    lax = fig.add_axes([legend_left, bottom, legend_w, cbar_h])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.set_xticks([])
    lax.set_yticks([])
    for spine in lax.spines.values():
        spine.set_visible(False)
    sw_w = 0.16
    lax.add_patch(Rectangle((0.02, 0.0), sw_w, 1.0,
                            facecolor='none', edgecolor=OVERLAP_COLOR,
                            hatch='///', linewidth=0.5, alpha=0.55))
    lax.add_patch(Rectangle((0.02 + sw_w + 0.02, 0.0), sw_w, 1.0,
                            facecolor='none', edgecolor=OVERLAP_COLOR,
                            hatch='\\\\\\', linewidth=0.5, alpha=0.55))
    lax.text(0.02 + 2 * sw_w + 0.05, 0.5,
             'multi-diffusion overlap region',
             ha='left', va='center', fontsize=12)

    # Plain filename for the canonical Figure 2; keep --suffix-z for sweeps.
    suffix = f'_z{args.z:02d}' if args.suffix_z else ''
    out_png = out_dir / f'figure4{suffix}.png'
    out_pdf = out_dir / f'figure4{suffix}.pdf'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_png}')
    print(f'Saved {out_pdf}')


if __name__ == '__main__':
    main()
