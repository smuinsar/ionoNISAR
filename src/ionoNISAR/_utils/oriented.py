"""Make the anisotropic hole fill work at ANY band orientation, not just axis-aligned."""
import numpy as np
from scipy import ndimage

from .fill import fill as _fill

TILT_MIN_DEG = 10.0   # under this the band frame IS the lattice frame: identical code path
TILT_R_MIN = 0.30     # angular concentration under this: orientation not trustworthy


def orientation_cells(f, valid, smooth_cells=28.0):
    """Band orientation in LATTICE CELLS, degrees CCW from the range (last) axis, and R."""
    g = np.where(valid, np.nan_to_num(f), 0.0)
    w = valid.astype(np.float64)
    g = ndimage.gaussian_filter(g, smooth_cells)
    w = ndimage.gaussian_filter(w, smooth_cells)
    g = np.where(w > 1e-3, g / np.maximum(w, 1e-3), 0.0)
    g -= g.mean()
    ha = np.hanning(g.shape[0])[:, None] * np.hanning(g.shape[1])[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * ha))) ** 2
    ka = np.fft.fftshift(np.fft.fftfreq(g.shape[0]))[:, None]
    kr = np.fft.fftshift(np.fft.fftfreq(g.shape[1]))[None, :]
    P = np.where((ka ** 2 + kr ** 2) > 0, P, 0.0)
    th = np.arctan2(ka + 0 * kr, kr + 0 * ka)
    z = (P * np.exp(2j * th)).sum() / P.sum()
    return (np.degrees(np.angle(z)) / 2.0 + 180.0) % 180.0 - 90.0, float(abs(z))


def band_tilt(f, valid):
    """The tilt to fill along: 0.0 unless the field is clearly oriented AND clearly off-axis."""
    th, R = orientation_cells(f, valid)
    if R < TILT_R_MIN or abs(th) < TILT_MIN_DEG or abs(th) > 90.0 - TILT_MIN_DEG:
        return 0.0, th, R
    return float(th), th, R


def _rot(a, ang, order=1):
    return ndimage.rotate(a, ang, order=order, reshape=True, mode="constant", cval=0.0)


def to_band_frame(f, valid, theta):
    """Rotate a masked field into the frame where the bands are range-parallel."""
    num = _rot(np.where(valid, np.nan_to_num(f), 0.0), theta)
    den = _rot(valid.astype(np.float64), theta)
    solid = _rot(np.ones(f.shape, dtype=np.float64), theta) > 0.99
    v = den > 0.5
    return np.where(v, num / np.maximum(den, 1e-9), 0.0), v, solid


def from_band_frame(a, theta, shape, order=1):
    """Rotate back and crop to the original shape."""
    b = _rot(a, -theta, order)
    a0, b0 = (b.shape[0] - shape[0]) // 2, (b.shape[1] - shape[1]) // 2
    return b[a0:a0 + shape[0], b0:b0 + shape[1]]


def band_elongation(f, valid, theta, threshold=0.10):
    """(along-track reach, across-track reach, ratio) measured in the BAND frame."""
    from .elongation import reach, structure_function
    g, v, solid = to_band_frame(f, valid, theta)
    ii, jj = np.where(solid)
    m = int(0.05 * min(ii.max() - ii.min(), jj.max() - jj.min()))
    sl = (slice(ii.min() + m, ii.max() - m), slice(jj.min() + m, jj.max() - m))
    gg, vv = g[sl], v[sl] & solid[sl]
    ra = reach(structure_function(gg, vv, 0), threshold)
    rr = reach(structure_function(gg, vv, 1), threshold)
    return ra, rr


def oriented_fill(y, valid, aniso, theta, **kw):
    """smooth_fill with the penalty's long axis put along `theta` instead of along range."""
    if not theta:
        return _fill(y, valid, method="pls", aniso=aniso, **kw)
    n0 = y.shape
    g, v, solid = to_band_frame(y, valid, theta)
    zr, info = _fill(np.where(v, g, np.nan), v, method="pls", aniso=aniso, **kw)
    z = from_band_frame(np.where(solid, zr, 0.0), theta, n0)
    # the robust pass's rejection mask has to come back too: fill_priors takes the UNION over
    # channels from it, and a channel that silently returned an empty one would put the
    # +-64 px failed matches straight back into regional_level
    rej = info.get("rejected")
    if rej is not None:
        info = dict(info)
        info["rejected"] = valid & (from_band_frame(rej.astype(np.float64), theta, n0) > 0.5)
    return np.where(valid, y, z), info
