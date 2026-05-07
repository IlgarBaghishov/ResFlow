"""Plotting helpers for SiliciclasticReservoirs cubes.

Cubes are (X, Y, Z) with Z (depth) on the LAST axis. Slice convention:
  XY @ z = z_idx     -> volume[:, :, z_idx].T          (X horizontal, Y vertical)
  XZ @ y = y_idx     -> volume[:, y_idx, :].T          (X horizontal, Z vertical)
  YZ @ x = x_idx     -> volume[x_idx, :, :].T          (Y horizontal, Z vertical)

Color schemes follow a canonical earth-science palette:

    RESERVOIR_CMAP            = grey -> mustard -> burnt-orange (continuous)
                                 used for binary facies, porosity, log-permeability
    ALLUVSIM_FACIES_COLORS    = 6 fixed hex codes for the categorical 6-class
                                facies (-1=FF, 0=FFCH, 1=CS, 2=LV, 3=LA, 4=CH)
"""
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np


# Continuous-field colormap.
RESERVOIR_CMAP = LinearSegmentedColormap.from_list(
    'reservoir', ['#999999', '#e8c840', '#b85a18'], N=256)

# Continuous fields (porosity, permeability) keep the smooth reservoir colormap.
CMAP_PORO = RESERVOIR_CMAP
CMAP_PERM = RESERVOIR_CMAP

# Binary facies (0 = non-reservoir / mud, 1 = reservoir / sand) is *discrete*,
# so render it with a 2-class ListedColormap using the reservoir-cmap endpoints
# (grey for 0, burnt-orange for 1). This matches the categorical look of the
# alluvsim 6-class scheme below.
CMAP_FACIES = ListedColormap(['#999999', '#b85a18'], name='facies_binary')
NORM_FACIES = BoundaryNorm([-0.5, 0.5, 1.5], CMAP_FACIES.N)
FACIES_BINARY_LABELS = ['shale', 'sand']


# 6-class alluvsim categorical scheme.
# Class order along the colormap: -1, 0, 1, 2, 3, 4  ->  FF, FFCH, CS, LV, LA, CH
ALLUVSIM_FACIES_COLORS = {
    -1: '#b4b4b4',   # FF   (floodplain)
     0: '#5b4636',   # FFCH (mud plug)
     1: '#f2d16b',   # CS   (splay)
     2: '#e8a23a',   # LV   (levee)
     3: '#c89b5e',   # LA   (point bar)
     4: '#7a3f14',   # CH   (channel)
}
ALLUVSIM_LABELS = ['FF', 'FFCH', 'CS', 'LV', 'LA', 'CH']
ALLUVSIM_COLORS = [ALLUVSIM_FACIES_COLORS[k] for k in (-1, 0, 1, 2, 3, 4)]
CMAP_ALLUVSIM = ListedColormap(ALLUVSIM_COLORS, name='alluvsim')
NORM_ALLUVSIM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], CMAP_ALLUVSIM.N)


def slice_xy(vol, z):
    return vol[:, :, z].T


def slice_xz(vol, y):
    return vol[:, y, :].T


def slice_yz(vol, x):
    return vol[x, :, :].T


def imshow_slice(ax, img, *, cmap, vmin=None, vmax=None, norm=None):
    """imshow with no ticks, origin='lower', equal aspect.

    Returns the AxesImage so a colorbar can be attached by the caller.
    """
    kw = dict(cmap=cmap, origin='lower', aspect='equal', interpolation='nearest')
    if norm is not None:
        kw['norm'] = norm
    else:
        kw['vmin'] = vmin
        kw['vmax'] = vmax
    im = ax.imshow(img, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def perm_to_logperm(perm):
    """Recommended preprocessing per dataset README (perm spans 5+ decades)."""
    return np.log10(np.maximum(perm.astype(np.float32), 1e-3))
