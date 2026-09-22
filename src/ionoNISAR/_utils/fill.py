"""Robust penalized-least-squares gap filling for gridded data."""
from __future__ import annotations

import numpy as np

METHODS = ("pls", "pls_mg", "laplace", "spring", "biharmonic", "idw",
           "idw_smooth", "local", "median_grow", "nearest", "griddata",
           "griddata_cubic", "poly", "rbf")


def fill(y, valid=None, method="pls_mg", **kw):
    """Fill the invalid cells of a 2-D array by `method`; returns (z, info)."""
    if valid is None:
        valid = np.isfinite(y)
    if method == "pls":
        z, s_, rej = smooth_fill(y, valid, **kw)
        return z, {"s": s_, "rejected": rej}
    if method == "pls_mg":
        return _fill_pls_multigrid(y, valid, **kw)
    if method in ("laplace", "spring"):
        return _fill_linear(y, valid, method, **kw)
    if method == "biharmonic":
        return _fill_biharmonic(y, valid, **kw)
    if method == "idw":
        return _fill_idw(y, valid, **kw)
    if method == "local":
        return _fill_local(y, valid, **kw)
    if method in ("griddata", "griddata_cubic"):
        return _fill_griddata(y, valid, cubic=method.endswith("cubic"), **kw)
    if method == "idw_smooth":
        return _fill_idw(y, valid, smoothing_iterations=kw.pop("smoothing", 3), **kw)
    if method == "median_grow":
        return _fill_local(y, valid, median=True, **kw)
    if method == "nearest":
        return _fill_nearest(y, valid, **kw)
    if method == "poly":
        return _fill_poly(y, valid, **kw)
    if method == "rbf":
        return _fill_rbf(y, valid, **kw)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def _fill_pls_multigrid(y, valid, levels=None, s=None, verbose=True, **kw):
    """PLS solved coarse-to-fine, so big holes actually converge."""
    from scipy.ndimage import zoom

    ny, nx = y.shape
    if levels is None:                       # enough levels to make the deepest
        levels = max(1, int(np.log2(max(ny, nx) / 64)))   # hole small at the top
    y = np.asarray(y, float)
    pyr = [(y, valid)]
    for _ in range(levels):
        yc, vc = pyr[-1]
        num = _block_reduce(np.where(vc, yc, 0.0))
        den = _block_reduce(vc.astype(float))
        pyr.append((np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), den > 0))

    z0 = None
    for lev in range(len(pyr) - 1, -1, -1):
        yl, vl = pyr[lev]
        if not vl.any():
            continue
        sl = None if s is None else s * (2.0 ** (-4 * lev))
        zl, s_used, rej = smooth_fill(yl, vl, s=sl, z0=z0, verbose=False, **kw)
        if verbose:
            print(f"[fill] pls_mg level {lev}: {yl.shape}, "
                  f"{100 * vl.mean():.0f} % valid, s={s_used:.4g}")
        if lev:
            f = pyr[lev - 1][0].shape
            z0 = zoom(zl, 2, order=1)[:f[0], :f[1]]
            if z0.shape != f:                       # odd sizes: pad the edge
                z0 = np.pad(z0, ((0, f[0] - z0.shape[0]), (0, f[1] - z0.shape[1])),
                            mode="edge")
    return zl, {"s": s_used, "rejected": rej}


def _block_reduce(a, k=2):
    ny, nx = a.shape
    ap = np.pad(a, ((0, (-ny) % k), (0, (-nx) % k)))
    return ap.reshape(ap.shape[0] // k, k, ap.shape[1] // k, k).sum(axis=(1, 3))


def _fill_linear(y, valid, method, **kw):
    """Laplace or spring-analogy solve over the holes (inpaint_nans 2 and 4)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg

    ny, nx = y.shape
    hole = ~valid
    idx = -np.ones((ny, nx), int)
    idx[hole] = np.arange(hole.sum())
    rows, cols, vals, rhs = [], [], [], np.zeros(hole.sum())
    off = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if method == "spring":                 # springs to the 8 neighbours
        off = off + ((-1, -1), (-1, 1), (1, -1), (1, 1))
    yi, xi = np.where(hole)
    for k, (dy, dx) in enumerate(off):
        ny_, nx_ = yi + dy, xi + dx
        good = (ny_ >= 0) & (ny_ < ny) & (nx_ >= 0) & (nx_ < nx)
        i = idx[yi[good], xi[good]]
        j = idx[ny_[good], nx_[good]]
        rows.append(i); cols.append(i); vals.append(np.ones(i.size))
        inner = j >= 0
        rows.append(i[inner]); cols.append(j[inner]); vals.append(-np.ones(inner.sum()))
        edge = (j < 0)
        np.add.at(rhs, i[edge], y[ny_[good][edge], nx_[good][edge]])
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(hole.sum(), hole.sum())).tocsr()
    x, info = cg(A, rhs, rtol=kw.get("tol", 1e-6), maxiter=kw.get("max_iter", 2000))
    z = y.copy()
    z[hole] = x
    return z, {"cg_info": info}


def _fill_biharmonic(y, valid, **kw):
    from skimage.restoration import inpaint_biharmonic
    z = inpaint_biharmonic(np.where(valid, y, 0.0), ~valid)
    return z, {}


def _fill_idw(y, valid, max_search_distance=0, smoothing_iterations=0, **kw):
    from rasterio.fill import fillnodata
    d = max_search_distance or float(sum(y.shape))
    z = fillnodata(np.asarray(y, np.float32).copy(), mask=valid.astype(np.uint8),
                   max_search_distance=d, smoothing_iterations=smoothing_iterations)
    return z.astype(float), {}


def _fill_local(y, valid, max_iter=500, median=False, **kw):
    from scipy import ndimage

    z = np.asarray(y, float).copy()
    have = valid.copy()
    for _ in range(max_iter):
        todo = ~have
        if not todo.any():
            break
        if median:
            # median of the 8 neighbours, as a stack of shifts rather than
            # generic_filter: the callback version is a Python call per pixel and
            # took 107 s on a 300 x 300 test, which is hours on a real lattice
            w = np.where(have, z, np.nan)
            sh = [np.roll(np.roll(w, dy, 0), dx, 1)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)]
            for a_, (dy, dx) in zip(sh, [(dy, dx) for dy in (-1, 0, 1)
                                         for dx in (-1, 0, 1) if (dy or dx)]):
                if dy > 0: a_[:dy] = np.nan
                elif dy < 0: a_[dy:] = np.nan
                if dx > 0: a_[:, :dx] = np.nan
                elif dx < 0: a_[:, dx:] = np.nan
            with np.errstate(invalid="ignore"):
                est = np.nanmedian(np.stack(sh, 0), axis=0)
            grow = todo & np.isfinite(est)
            z[grow] = est[grow]
        else:
            num = ndimage.uniform_filter(np.where(have, z, 0.0), 3, mode="nearest")
            den = ndimage.uniform_filter(have.astype(float), 3, mode="nearest")
            grow = todo & (den > 0)
            z[grow] = num[grow] / den[grow]
        if not grow.any():
            break
        have |= grow
    return z, {}


def _fill_nearest(y, valid, **kw):
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.asarray(y, float)[tuple(idx)], {}


def _fill_poly(y, valid, order=2, iters=3, **kw):
    """One robust low-order surface over the whole grid."""
    ny, nx = y.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    u, v = (yy / ny - 0.5), (xx / nx - 0.5)
    terms = [np.ones_like(u), u, v]
    if order >= 2:
        terms += [u * u, u * v, v * v]
    A = np.column_stack([t[valid].ravel() for t in terms])
    b = np.asarray(y, float)[valid].ravel()
    w = np.ones_like(b)
    for _ in range(max(1, iters)):
        c, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        r = b - A @ c
        sig = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = 1.0 / np.sqrt(1.0 + (r / (4.685 * max(sig, 1e-9))) ** 2)
    z = sum(ci * t for ci, t in zip(c, terms))
    return np.where(valid, y, z), {"coeff": c}


def _fill_rbf(y, valid, sample=20000, smoothing=1.0, **kw):
    """Thin-plate spline through a random subset of the samples."""
    from scipy.interpolate import RBFInterpolator

    ny, nx = y.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    vi = np.flatnonzero(valid.ravel())
    rng = np.random.default_rng(0)
    pick = rng.choice(vi, size=min(sample, vi.size), replace=False)
    pts = np.column_stack([yy.ravel()[pick], xx.ravel()[pick]]).astype(float)
    f = RBFInterpolator(pts, np.asarray(y, float).ravel()[pick],
                        kernel="thin_plate_spline", smoothing=smoothing,
                        neighbors=64)
    hole = ~valid
    tgt = np.column_stack([yy[hole], xx[hole]]).astype(float)
    z = np.asarray(y, float).copy()
    z[hole] = f(tgt)
    return z, {"n_samples": int(pick.size)}


def smooth_fill(y, valid=None, s=None, robust=3, max_iter=100, tol=1e-3,
                workers=8, z0=None, aniso=1.0, verbose=True):
    """Fill the invalid cells of a 2-D array and return the smoothed field."""
    from functools import partial

    from scipy.fft import dctn as _dctn, idctn as _idctn
    from scipy.ndimage import distance_transform_edt
    from scipy.optimize import minimize_scalar

    dctn = partial(_dctn, norm="ortho", workers=workers)
    idctn = partial(_idctn, norm="ortho", workers=workers)

    y = np.asarray(y, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(y)
    valid = valid & np.isfinite(y)
    if not valid.any():
        raise ValueError("no valid samples to fill from")

    n = y.shape
    # eigenvalues of the 2-D Laplacian under DCT-II boundary conditions, with the
    # last axis weighted by `aniso` so the penalty can be elongated along it
    w = [1.0] * len(n)
    w[-1] = float(aniso)
    lam = sum(
        w[d] * (-2.0 + 2.0 * np.cos(np.arange(k) * np.pi / k)).reshape(
            [-1 if d == i else 1 for i in range(len(n))])
        for d, k in enumerate(n)
    ) ** 2

    # nearest-neighbour seed: the iteration converges from anything, but starting
    # near the answer keeps the GCV search honest and cheap
    if z0 is not None:
        z = np.where(valid, np.nan_to_num(y), z0)
    else:
        idx = distance_transform_edt(~valid, return_distances=False,
                                     return_indices=True)
        z = np.where(valid, np.nan_to_num(y), y[tuple(idx)])
    y0 = np.where(valid, y, 0.0)

    W = valid.astype(np.float64)
    nf = float(valid.sum())

    def solve(log10s, z_in, W_in, iterate):
        """One (or many) DCT relaxation(s) at smoothing 10**log10s."""
        gamma = 1.0 / (1.0 + (10.0 ** log10s) * lam)
        z_ = z_in
        for k in range(max_iter if iterate else 1):
            z_prev = z_
            z_ = idctn(gamma * dctn(W_in * (y0 - z_) + z_))
            if not iterate:
                break
            d = np.linalg.norm(z_ - z_prev) / max(np.linalg.norm(z_), 1e-30)
            if d < tol:
                break
        return z_, gamma

    def gcv(log10s):
        """Generalized cross-validation score, evaluated on the measured cells."""
        z_, gamma = solve(log10s, z, W, iterate=False)
        rss = float(np.sum(((y0 - z_) * W) ** 2))
        tr = float(gamma.sum())
        return rss / nf / (1.0 - tr / y.size) ** 2

    for pas in range(max(1, robust + 1)):
        if s is None:
            r = minimize_scalar(gcv, bounds=(-6, 6), method="bounded",
                                options={"xatol": 0.1})
            log10s = float(r.x)
        else:
            log10s = float(np.log10(s))
        z, gamma = solve(log10s, z, W, iterate=True)

        if pas >= robust:
            break
        # bisquare re-weighting: rescale the residuals of the MEASURED cells by a
        # robust spread and drop the ones the smooth field cannot explain
        r = (y0 - z)[valid]
        mad = np.median(np.abs(r - np.median(r)))
        sigma = 1.4826 * mad * np.sqrt(1.0 - gamma.sum() / y.size)
        if sigma <= 0:
            break
        u = np.zeros_like(y)
        u[valid] = r / (4.685 * sigma)
        w = np.clip(1.0 - u ** 2, 0.0, None) ** 2
        W_new = np.where(valid, w, 0.0)
        if verbose:
            kept = float((W_new[valid] > 0).mean())
            print(f"[fill] robust pass {pas + 1}: log10(s)={log10s:.2f}, "
                  f"sigma={sigma:.3f}, {100 * (1 - kept):.1f} % of measured "
                  f"samples down-weighted to zero")
        W = W_new
        if not W.any():
            raise RuntimeError("robust weighting rejected every sample")

    if verbose:
        print(f"[fill] penalized least squares done, log10(s)={log10s:.2f}")
    return z, 10.0 ** log10s, valid & (W <= 0)


def _fill_griddata(y, valid, cubic=False, **kw):
    """scipy.interpolate.griddata over the valid samples."""
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt

    ny, nx = y.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.column_stack([yy[valid], xx[valid]])
    tgt = np.column_stack([yy[~valid], xx[~valid]])
    v = griddata(pts, y[valid], tgt, method="cubic" if cubic else "linear")
    z = y.astype(float).copy()
    z[~valid] = v
    outside = ~np.isfinite(z)
    if outside.any():
        idx = distance_transform_edt(outside, return_distances=False,
                                     return_indices=True)
        z[outside] = z[tuple(idx)][outside]
    return z, {"outside_hull": int(outside.sum())}


def benchmark_methods(y, valid, hole, methods=METHODS, s=None, verbose=True):
    """Score each method against data withheld inside `hole`."""
    import time

    from scipy.ndimage import distance_transform_edt

    keep = valid & ~hole
    truth = valid & hole
    dist = distance_transform_edt(~keep)
    bands = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 10000)]
    bands = [(lo, hi) for lo, hi in bands if (truth & (dist >= lo) & (dist < hi)).sum() > 50]

    print(f"withheld {int(hole.sum())} cells, {int(truth.sum())} with truth; "
          f"{100 * keep.mean():.1f} % of the grid kept; deepest {dist[hole].max():.0f} cells")
    head = f"{'method':>11} {'time':>7}" + "".join(f"{f'{lo}-{hi}':>12}" for lo, hi in bands)
    print(head + "\n" + "-" * len(head))
    truth_row = f"{'(truth)':>11} {'':>7}" + "".join(
        f"{np.median(y[truth & (dist >= lo) & (dist < hi)]):+12.2f}" for lo, hi in bands)
    print(truth_row)
    out = {}
    for m in methods:
        t = time.time()
        try:
            z, _ = fill(y, keep, method=m, s=s) if m.startswith("pls") else fill(y, keep, method=m)
        except Exception as e:
            print(f"{m:>11} {'--':>7}  failed: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t
        row = f"{m:>11} {dt:6.1f}s"
        for lo, hi in bands:
            msk = truth & (dist >= lo) & (dist < hi)
            row += f"{np.median(np.abs(z[msk] - y[msk])):12.3f}"
        print(row, flush=True)
        out[m] = z
    print("\n(numbers are median |error| in px against the withheld truth)")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="compare gap-filling methods on a "
                                             "dense-offset field")
    ap.add_argument("--offsets", default="outputs_offsets/offsets.npz")
    ap.add_argument("--snr-min", type=float, default=8.0)
    ap.add_argument("--radius", type=int, default=260,
                    help="radius of the disc of data to withhold, in samples "
                         "(default 260, the size of this frame's icefield gap)")
    ap.add_argument("--speckle", type=float, default=0.30,
                    help="also hide this fraction of the rest, as the water and "
                         "glacier masks do")
    ap.add_argument("--cutoff", type=float, default=100.0)
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()

    d = np.load(a.offsets)
    az, snr = d["azimuth"], d["snr"]
    valid = np.isfinite(az) & (snr >= a.snr_min)
    ny, nx = az.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    hole = ((yy - ny // 2) ** 2 + (xx - int(nx * 0.30)) ** 2) < a.radius ** 2
    if a.speckle:
        hole |= np.random.default_rng(0).random(az.shape) < a.speckle
    benchmark_methods(az, valid, hole, methods=tuple(a.methods.split(",")),
                      s=(a.cutoff / (2 * np.pi)) ** 4)
