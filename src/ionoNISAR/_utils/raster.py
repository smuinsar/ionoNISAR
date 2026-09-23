"""The handful of helpers every stage of the ionosphere pipeline needs."""
from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.special import i0 as bessi0            # modified Bessel function of order 0

import isce3
from osgeo import gdal, osr

C = 299_792_458.0
ellip = isce3.core.Ellipsoid()
zero_lut = isce3.core.LUT2d()      # zero-Doppler: the RSLC image grid is zero-Doppler

# =====================================================================
# raster i/o
# =====================================================================
def geotransform_of(gg):
    """(x0, dx, 0, y0, 0, dy) for an isce3 GeoGridParameters."""
    return (gg.start_x, gg.spacing_x, 0.0, gg.start_y, 0.0, gg.spacing_y)


def save_gtiff(path, arr, grid, epsg=None, nodata=None):
    """Write an array to a tiled, DEFLATE-compressed GeoTIFF."""
    a = np.asarray(arr)
    if hasattr(grid, "start_x"):
        gt, epsg = geotransform_of(grid), int(epsg or grid.epsg)
    else:
        gt = tuple(float(v) for v in grid)
        if epsg is None:
            raise ValueError("save_gtiff needs an epsg when `grid` is a geotransform")
        epsg = int(epsg)
    cplx = np.iscomplexobj(a)
    gdt = gdal.GDT_CFloat32 if cplx else gdal.GDT_Float32
    opts = ["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=DEFLATE",
            "ZLEVEL=6", "NUM_THREADS=ALL_CPUS", "BIGTIFF=IF_SAFER",
            "PREDICTOR=1" if cplx else "PREDICTOR=2"]
    length, width = a.shape
    ds = gdal.GetDriverByName("GTiff").Create(path, int(width), int(length), 1, gdt,
                                              options=opts)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None and not cplx:      # preserve NaN (e.g. unwrapped phase); flag it
        band.SetNoDataValue(float(nodata)); band.WriteArray(a.astype(np.float32))
    else:
        band.WriteArray(np.nan_to_num(a).astype(np.complex64 if cplx else np.float32))
    ds.FlushCache(); ds = None
    print("wrote", path)
    return path


def read_gtiff(path):
    """The raster as float64 with nodata already turned back into NaN."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray().astype(np.float64)
    nd = band.GetNoDataValue()
    if nd is not None and not np.isnan(nd):
        a = np.where(a == nd, np.nan, a)
    return np.where(np.isfinite(a), a, np.nan)


class BlockRaster:
    """A read-only 2-D raster that slices like a memmap but reads through GDAL."""

    def __init__(self, path, shape=None, dtype=np.float64):
        self._ds = gdal.Open(path)
        if self._ds is None:
            raise FileNotFoundError(path)
        self._b = self._ds.GetRasterBand(1)
        self.shape = (self._ds.RasterYSize, self._ds.RasterXSize)
        self.dtype = dtype
        self.path = path
        if shape is not None and tuple(shape) != self.shape:
            raise ValueError(f"{path} is {self.shape}, not the {tuple(shape)} expected")

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        rs, cs = key
        if not isinstance(rs, slice) or not isinstance(cs, slice):
            raise TypeError(f"{type(self).__name__} takes slices, not {key!r}")
        r0, r1, _ = rs.indices(self.shape[0])
        c0, c1, _ = cs.indices(self.shape[1])
        if r1 <= r0 or c1 <= c0:
            return np.empty((max(r1 - r0, 0), max(c1 - c0, 0)), self.dtype)
        return self._b.ReadAsArray(c0, r0, c1 - c0, r1 - r0).astype(self.dtype)


def open_raster(path, shape, dtype=np.float64):
    """np.memmap for a raw file, BlockRaster for anything GDAL has to decode."""
    want = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if os.path.exists(path) and os.path.getsize(path) == want:
        return np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))
    r = BlockRaster(path, shape, dtype)
    print(f"  {os.path.basename(path)}: reading through GDAL "
          f"({os.path.getsize(path) / 1e9:.2f} GB on disk vs {want / 1e9:.1f} GB raw, "
          f"so it is compressed)")
    return r


def resample_to_grid(tif, gt, shape, epsg):
    """Read a geocoded raster onto this run's output grid.  NaN where it has no data."""
    gdal.UseExceptions()
    srs = osr.SpatialReference(); srs.ImportFromEPSG(int(epsg))
    ny, nx = shape
    ds = gdal.Warp("", tif, format="MEM", outputType=gdal.GDT_Float32,
                   dstSRS=srs.ExportToWkt(), resampleAlg="bilinear", dstNodata=np.nan,
                   outputBounds=(gt[0], gt[3] + ny * gt[5], gt[0] + nx * gt[1], gt[3]),
                   width=nx, height=ny)
    a = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None
    return a


def load_dem(path):
    """(isce3 Raster, DEMInterpolator with min/max/mean height computed) for a DEM GeoTIFF."""
    raster = isce3.io.Raster(path)
    di = isce3.geometry.DEMInterpolator()
    di.load_dem(raster)
    di.compute_min_max_mean_height()
    print(f"DEM {path}: terrain {di.min_height:.0f}..{di.max_height:.0f} m "
          f"(mean {di.mean_height:.0f} m)")
    return raster, di


# =====================================================================
# Goldstein filter + coherence.  Kept verbatim: do not "tidy" the arithmetic, the
# delivered products were made with exactly these steps.
# =====================================================================
def _kaiser_window_2d(n, beta=2.12):
    """Create a 2D Kaiser window (separable product of 1D Kaiser windows)."""
    m = 0.5 * (n - 1)
    idx = np.arange(n)
    arg = beta * np.sqrt(1.0 - ((idx - m) / m) ** 2)
    w1d = bessi0(arg) / bessi0(beta)
    return np.outer(w1d, w1d)


def _coherence_weights(cc_win, nfft):
    """Coherence estimation weighting function (triangular, distance-based)."""
    half = (cc_win - 1) / 2.0
    sigma = (np.sqrt(cc_win * cc_win) + 1.0) / (np.sqrt(cc_win * cc_win) - 1.0)
    idx = np.arange(cc_win)
    dy = (idx - half) / half
    dx = (idx - half) / half
    DX, DY = np.meshgrid(dx, dy)
    norm_dist = np.sqrt(DX**2 + DY**2)
    return np.maximum(1.0 - np.abs(norm_dist / sigma), 0.0)


def _estimate_coherence_fast(sm, cc_win, nfft=32):
    """Vectorized coherence estimation using the triangular distance-based kernel."""
    from scipy.ndimage import convolve
    wcc = _coherence_weights(cc_win, nfft).astype(np.float64)
    amp = np.abs(sm).astype(np.float64)
    real_part = sm.real.astype(np.float64)
    imag_part = sm.imag.astype(np.float64)
    valid_mask = (amp > 0).astype(np.float64)
    sum_amp = convolve(amp * valid_mask, wcc, mode='constant', cval=0.0)
    sum_re = convolve(real_part * valid_mask, wcc, mode='constant', cval=0.0)
    sum_im = convolve(imag_part * valid_mask, wcc, mode='constant', cval=0.0)
    cc = np.zeros_like(amp, dtype=np.float32)
    valid = sum_amp > 0
    cc[valid] = (np.sqrt(sum_re[valid] ** 2 + sum_im[valid] ** 2) / sum_amp[valid]).astype(np.float32)
    return cc


def goldstein_filter_fast(ifg, alpha=0.5, nfft=32, step=None, cc_win=7,
                          wfrac_min=0.7, beta=2.12, nan_zero=False):
    """Goldstein adaptive spectral filter (Goldstein & Werner 1998) -- vectorized."""
    nlines, width = ifg.shape
    if step is None:
        step = max(nfft // 8, 1)

    win = _kaiser_window_2d(nfft, beta)
    half = nfft // 2

    # Pad input so patches at boundaries are zero-padded naturally.
    extra = step // 2
    padded = np.pad(ifg, ((half, half + extra), (half, half + extra)),
                    mode='constant', constant_values=0)

    sm = np.zeros_like(ifg)
    weight = np.zeros(ifg.shape, dtype=np.float32)

    i_centers = np.arange(0, nlines + step // 2, step)
    j_centers = np.arange(0, width + step // 2, step)

    nz_thresh = int(wfrac_min * nfft * nfft)

    pj_arr = j_centers
    n_j = len(pj_arr)

    # Chunk size for j-patches to cap memory (~40 MB per chunk)
    CHUNK_J = max(1, int(40e6 / (nfft * nfft * 16)))  # 16 bytes per complex128

    for i in i_centers:
        pi = i
        row_strip = padded[pi:pi + nfft, :]

        for jc_start in range(0, n_j, CHUNK_J):
            jc_end = min(jc_start + CHUNK_J, n_j)
            pj_chunk = pj_arr[jc_start:jc_end]

            col_idx = pj_chunk[:, None] + np.arange(nfft)
            patches = row_strip[:, col_idx].transpose(1, 0, 2).copy()

            nz_counts = np.count_nonzero(patches, axis=(1, 2))
            valid_mask = nz_counts >= nz_thresh

            if not np.any(valid_mask):
                continue

            valid_idx = np.where(valid_mask)[0]
            valid_patches = patches[valid_idx]

            valid_patches *= win[np.newaxis, :, :]
            specs = np.fft.fft2(valid_patches)

            psd = specs.real ** 2 + specs.imag ** 2
            specs *= psd ** (alpha / 2.0)

            filtered_all = np.fft.ifft2(specs)

            for ki, k in enumerate(valid_idx):
                j = j_centers[jc_start + k]

                out_i0 = i - step // 2
                out_i1 = out_i0 + step
                out_j0 = j - step // 2
                out_j1 = out_j0 + step

                ci0 = max(out_i0, 0)
                ci1 = min(out_i1, nlines)
                cj0 = max(out_j0, 0)
                cj1 = min(out_j1, width)

                if ci0 >= ci1 or cj0 >= cj1:
                    continue

                fi0 = half - step // 2 + (ci0 - out_i0)
                fi1 = fi0 + (ci1 - ci0)
                fj0 = half - step // 2 + (cj0 - out_j0)
                fj1 = fj0 + (cj1 - cj0)

                block = filtered_all[ki, fi0:fi1, fj0:fj1]

                orig_block = ifg[ci0:ci1, cj0:cj1]
                mask = orig_block != 0
                sm[ci0:ci1, cj0:cj1][mask] = block[mask]
                weight[ci0:ci1, cj0:cj1][mask] = 1.0

    # --- Pass 2: Coherence estimation (vectorized with proper weighting) ---
    cc = _estimate_coherence_fast(sm, cc_win, nfft)

    # --- Pass 3: Renormalize by coherence ---
    amp = np.abs(sm)
    valid = amp > 0
    sm_out = np.zeros_like(sm)
    sm_out[valid] = cc[valid] * sm[valid] / amp[valid]

    if nan_zero:
        zero_mask = sm_out == 0
        sm_out = sm_out.astype(np.complex128)
        sm_out[zero_mask] = np.nan + 1j * np.nan
        cc[zero_mask] = np.nan

    return sm_out, cc


def coherence(a, b, win):
    """|E[a b*]| / sqrt(E[|a|^2] E[|b|^2]) over an odd boxcar window (the InSAR multilook)."""
    a = np.nan_to_num(a); b = np.nan_to_num(b)     # invalid (off-swath) geocoded pixels are NaN;
    # scipy's uniform_filter carries a running sum, so a single NaN would poison the whole
    # array -> zero coherence.
    num = uniform_filter((a*np.conj(b)).real, win) + 1j*uniform_filter((a*np.conj(b)).imag, win)
    den = np.sqrt(uniform_filter(np.abs(a)**2, win) * uniform_filter(np.abs(b)**2, win))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.abs(num)/den
    return np.clip(np.nan_to_num(c), 0, 1)
