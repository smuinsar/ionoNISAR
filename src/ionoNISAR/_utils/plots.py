"""One place for the rules that turn a product raster into a picture."""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAXDIM = 2048

# The requested house style.  Every savefig in this module goes through SAVE.
DPI = 300
SHRINK = 0.4
SAVE = dict(bbox_inches="tight", pad_inches=0.0, dpi=DPI)


# =====================================================================
# colormaps -- loaded once, shared by every caller
# =====================================================================
def _load_cm(name, path):
    return ListedColormap(np.loadtxt(path) / 255.0, name=name)


RMG = _load_cm("rmg", os.path.join(SCRIPT_DIR, "rmg.cm"))    # cyclic: wrapped phase
HLS = _load_cm("hls", os.path.join(SCRIPT_DIR, "hls.cm"))    # unwrapped phase


# =====================================================================
# stretch policy: matched on the file stem, longest prefix first, so a name that is a
# prefix of another still resolves to its own rule
# =====================================================================
UNWRAPPED = ("ifg_unw", "ifg_phase_unw", "unw_", "phi_a_unw", "phi_diff_unw", "nondisp")
WRAPPED = ("ifg_phase_raw", "ifg_phase_corrected", "ifg_phase", "filt_ifg_phase",
           "ifg_gslc_phase", "phi_diff_ms", "phi_diff")
COH = ("coh_", "coherence", "ifg_coh", "ifg_gslc_coh", "mai_coh")
# a PRECISION layer is strictly positive and its zero is meaningful, so it gets neither a
# diverging map nor limits centred on the median -- both of which read a 0.25 mm sigma as
# if it swung negative.  Sequential, from zero.
POSITIVE = ("iono_screen_sigma", "mai_screen_sigma")


def kind_of(stem):
    """Which display kind this product is, from its filename."""
    s = stem.lower()
    if s.startswith(UNWRAPPED):
        return "unwrapped"
    if s.startswith(WRAPPED):
        return "wrapped"
    if s.startswith(COH):
        return "coh"
    if s.startswith(POSITIVE):
        return "positive"
    return "signed"


def symmetric_limits(a, pct=98):
    """Colour limits SYMMETRIC about the median, at the `pct` percentile of |deviation|."""
    ok = np.isfinite(a)
    if not ok.any():
        return 0.0, 1.0
    c = float(np.median(a[ok]))
    v = float(np.percentile(np.abs(a[ok] - c), pct))
    return (c - v, c + v) if v > 0 else (c - 1.0, c + 1.0)


def style(stem, a):
    """(cmap, vmin, vmax, colourbar label, cyclic) for an array of this kind."""
    k = kind_of(stem)
    if k == "wrapped":
        return RMG, -np.pi, np.pi, "radians (wrapped)", True
    if k == "coh":
        return "gray", 0.0, 1.0, r"$\gamma$", False
    if k == "unwrapped":
        lo, hi = symmetric_limits(a)
        return HLS, lo, hi, "radians (unwrapped)", False
    if k == "positive":
        ok = np.isfinite(a)
        hi = float(np.percentile(a[ok], 98)) if ok.any() else 1.0
        unit = "mm" if "_mm" in stem.lower() else "radians"
        return "magma", 0.0, hi if hi > 0 else 1.0, rf"1$\sigma$ ({unit})", False
    lo, hi = symmetric_limits(a)
    s = stem.lower()
    lab = ("pixels" if s.startswith("offset_") and s.endswith("_px") else
           "metres" if s.startswith(("offset_", "mai_azimuth")) and s.endswith("_m") else
           "correlation SNR" if s.startswith("offset_snr") else
           "1 = measured" if s.startswith(("iono_screen_valid", "offset_filled",
                                           "mai_screen_valid")) else
           # a coherence DIFFERENCE is signed, so it is stretched like any other signed
           # field, but it is dimensionless -- labelling it "radians" was simply wrong
           r"$\Delta\gamma$" if s.startswith(("dcoh", "coh_change", "delta_coh")) else
           "radians")
    return "jet", lo, hi, lab, False


def cyclic_stem(stem):
    """True when this product must be decimated by nearest neighbour, not averaged."""
    return kind_of(stem) == "wrapped"


# =====================================================================
# reading
# =====================================================================
def read_tif(path, max_dim=MAXDIM, cyclic=None):
    """Decimated read of band 1.  Returns (array, extent, decimation) with nodata as NaN."""
    from osgeo import gdal
    gdal.UseExceptions()

    if cyclic is None:
        cyclic = cyclic_stem(os.path.splitext(os.path.basename(path))[0])
    ds = gdal.Open(path)
    n = max(ds.RasterXSize, ds.RasterYSize)
    d = max(1, int(np.ceil(n / max_dim)))
    nx, ny = max(1, ds.RasterXSize // d), max(1, ds.RasterYSize // d)
    band = ds.GetRasterBand(1)
    alg = gdal.GRIORA_NearestNeighbour if cyclic else gdal.GRIORA_Average
    a = band.ReadAsArray(buf_xsize=nx, buf_ysize=ny, resample_alg=alg)
    if np.iscomplexobj(a):
        a = np.where(np.abs(a) > 0, np.angle(a), np.nan)
    a = a.astype(np.float32)
    nd = band.GetNoDataValue()
    if nd is not None and np.isfinite(nd):
        a = np.where(a == nd, np.nan, a)
    gt = ds.GetGeoTransform()
    ext = [gt[0], gt[0] + ds.RasterXSize * gt[1],
           gt[3] + ds.RasterYSize * gt[5], gt[3]]
    ds = None
    return a, ext, d


def grid_of(path):
    """(geotransform, nx, ny) -- for checking two rasters share a grid before pairing them."""
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(path)
    return ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize


def stats(a):
    """Valid fraction and 1/50/99 percentiles of the displayed array."""
    ok = np.isfinite(a)
    if not ok.any():
        return dict(valid=0.0, p1=np.nan, p50=np.nan, p99=np.nan)
    v = a[ok]
    p1, p50, p99 = np.percentile(v, [1, 50, 99])
    return dict(valid=float(ok.mean()), p1=float(p1), p50=float(p50), p99=float(p99))


def phase_flatness(phi, win_px, mask=None):
    """|<exp(i.phi)>| over a win_px box -- how flat a WRAPPED phase is locally."""
    from scipy.ndimage import uniform_filter

    ok = np.isfinite(phi)
    if mask is not None:
        ok &= mask
    if not ok.any():
        return float("nan")
    w = max(3, int(win_px))
    z = np.where(ok, np.exp(1j * np.nan_to_num(phi)), 0)
    num = np.abs(uniform_filter(z.real, w) + 1j * uniform_filter(z.imag, w))
    den = uniform_filter(ok.astype(float), w)
    full = den > 0.6           # only boxes that are mostly real data
    return float(np.mean((num / np.maximum(den, 1e-9))[full])) if full.any() else float("nan")


# =====================================================================
# rendering
# =====================================================================
def render_array(a, stem, out, extent=None, title=None, cmap=None, vlim=None,
                 label=None, figsize=(9, 8), xlabel="easting [m]", ylabel="northing [m]"):
    """One array, one page, with axes and a colourbar.  Returns its display statistics."""
    if not np.any(np.isfinite(a)):
        print(f"  {stem}: no valid pixels, skipped")
        return None
    cm, vmin, vmax, lab, _ = style(stem, a)
    if cmap is not None:
        cm = cmap
    if vlim is not None:
        vmin, vmax = vlim
    if label is not None:
        lab = label
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(a, cmap=cm, vmin=vmin, vmax=vmax, extent=extent,
                   interpolation="nearest", origin="upper")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title or stem)
    fig.colorbar(im, ax=ax, shrink=SHRINK, label=lab)
    fig.tight_layout()
    fig.savefig(out, **SAVE)
    plt.close(fig)
    return stats(a)


def render_tif(tif, out=None, max_dim=MAXDIM, title=None, quiet=False):
    """Render a product GeoTIFF beside itself as .png.  Returns its display statistics."""
    stem = os.path.splitext(os.path.basename(tif))[0]
    out = out or os.path.splitext(tif)[0] + ".png"
    a, ext, d = read_tif(tif, max_dim=max_dim)
    s = render_array(a, stem, out, extent=ext, title=title or stem)
    if s and not quiet:
        cm, vmin, vmax, lab, _ = style(stem, a)
        print(f"[png] {out}  {getattr(cm, 'name', cm)} {vmin:+.4g}..{vmax:+.4g}  "
              f"1/{d} decimation, valid {100 * s['valid']:.1f} %")
    return s


def render_dir(d, max_dim=MAXDIM):
    """A PNG beside every GeoTIFF in a directory.  Returns {stem: stats}."""
    import glob
    out = {}
    for tif in sorted(glob.glob(os.path.join(d, "*.tif"))):
        out[os.path.splitext(os.path.basename(tif))[0]] = render_tif(tif, max_dim=max_dim)
    return out


def panel_row(panels, title, out, cmap=None, vlim=None, label=None, extent=None,
              share_limits=True, figsize=None):
    """One row of panels on ONE colour scale, for comparing routes side by side."""
    arrs, subs = [], []
    for p, sub in panels:
        if isinstance(p, str):
            a, ext, _ = read_tif(p, cyclic=cyclic_stem(
                os.path.splitext(os.path.basename(p))[0]))
            extent = extent or ext
            stem = os.path.splitext(os.path.basename(p))[0]
        else:
            a, stem = np.asarray(p), title
        arrs.append(a)
        subs.append(sub)
    stem_kind = os.path.splitext(os.path.basename(panels[0][0]))[0] \
        if isinstance(panels[0][0], str) else title
    cm, vmin, vmax, lab, _ = style(stem_kind, np.concatenate([a.ravel() for a in arrs]))
    if cmap is not None:
        cm = cmap
    if vlim is not None:
        vmin, vmax = vlim
    if label is not None:
        lab = label
    if not share_limits:
        vmin = vmax = None

    n = len(arrs)
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.0 * n, 6.4), squeeze=False)
    for ax, a, sub in zip(axes[0], arrs, subs):
        im = ax.imshow(a, cmap=cm, vmin=vmin, vmax=vmax, extent=extent,
                       interpolation="nearest", origin="upper")
        ax.set_title(sub, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=SHRINK, label=lab)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, **SAVE)
    plt.close(fig)
    print(f"[png] {out}")
    return out


def panel_grid(panels, title, out, ncols=4, extent=None, figsize=None):
    """A grid of panels, each on ITS OWN kind's colour scale."""
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=figsize or (6.0 * ncols, 6.2 * nrows))
    for ax, item in zip(axes.ravel(), panels):
        a, sub, stem = item
        if a is None or not np.any(np.isfinite(a)):
            ax.set_axis_off()
            continue
        cm, vmin, vmax, lab, _ = style(stem, a)
        im = ax.imshow(a, cmap=cm, vmin=vmin, vmax=vmax, extent=extent,
                       interpolation="nearest", origin="upper")
        ax.set_title(sub, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=SHRINK, label=lab)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, **SAVE)
    plt.close(fig)
    print(f"[png] {out}")
    return out


def quicklook(path, arr, gray=False, vlim=None, stem=None):
    """A bare PNG beside a product: the raster, no axes, no colourbar."""
    stem = stem or os.path.splitext(os.path.basename(path))[0]
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        a = np.where(np.abs(a) > 0, np.angle(a), np.nan)
    a = a.astype(np.float32)
    if not np.any(np.isfinite(a)):
        print(f"  {stem}: no valid pixels, skipped")
        return None
    cm, vmin, vmax, _, _ = style(stem, a)
    if gray:
        cm = "gray"
    if vlim is not None:
        vmin, vmax = vlim
    png = os.path.splitext(path)[0] + ".png"
    # cap the figure so a 24704-px raster does not become a 100 MB PNG
    scale = min(0.3, 2400.0 / max(a.shape))
    fig, ax = plt.subplots(figsize=(a.shape[1] * scale / 100.0, a.shape[0] * scale / 100.0))
    ax.imshow(a, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(png, **SAVE)
    plt.close(fig)
    print(f"wrote {png} ({getattr(cm, 'name', cm)}, {vmin:+.4g} .. {vmax:+.4g})")
    return png


def plot_panels(panels, ext, xlabel, ylabel, png, aspect="equal"):
    """The three-panel offsets quicklook the coregistration has always written."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 6.4), squeeze=False)
    for ax, (a, ttl, unit, gray) in zip(axes[0], panels):
        a = np.asarray(a, dtype=np.float32)
        if not np.any(np.isfinite(a)):
            ax.set_axis_off()
            continue
        lo, hi = symmetric_limits(a)
        cm = "gray" if gray else "jet"
        if gray:
            lo, hi = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
        im = ax.imshow(a, cmap=cm, vmin=lo, vmax=hi, extent=ext,
                       interpolation="nearest", origin="upper", aspect=aspect)
        ax.set(title=ttl, xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(im, ax=ax, shrink=SHRINK, label=unit)
    fig.tight_layout()
    fig.savefig(png, **SAVE)
    plt.close(fig)
    print(f"wrote {png}")
    return png
