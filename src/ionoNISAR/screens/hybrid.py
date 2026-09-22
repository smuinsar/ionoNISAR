"""Hybrid screen: long along-track wavelengths from split, short ones from the azimuth offsets."""
import argparse
import os

import numpy as np


def row_metres(offsets_npz):
    """Metres per lattice row along track, from the correlator skip and azimuth spacing."""
    z = np.load(offsets_npz)
    try:
        return float(np.ravel(z["skip"])[0]) * float(np.ravel(z["az_spacing"])[0])
    except KeyError:
        raise SystemExit(f"{offsets_npz} carries no az_spacing; --cut-km is a wavelength and "
                         f"cannot be converted to lattice rows without it")


def build(offsets_npz, split_npz, offsets_lattice, out_npz, cut_km=32.0):
    """Combine the two screens at the given along-track cut and write the result."""
    from scipy.ndimage import gaussian_filter1d

    zo, zp = np.load(offsets_npz), np.load(split_npz)
    So, Sp = zo["screen"].astype(np.float64), zp["screen"].astype(np.float64)
    if So.shape != Sp.shape:
        raise SystemExit(f"lattices differ: offsets {So.shape}, split {Sp.shape}")

    row_m = row_metres(offsets_lattice)
    sig = cut_km * 1000.0 / row_m / 2.355
    lp = gaussian_filter1d(Sp, sig, axis=0, mode="nearest")
    hyb = lp + (So - gaussian_filter1d(So, sig, axis=0, mode="nearest"))

    print(f"lattice {So.shape}, {row_m:.1f} m/row; cut {cut_km:g} km (sigma {sig:.1f} rows)")
    for nm, v in (("offsets", So), ("split", Sp), ("hybrid", hyb)):
        print(f"  {nm:8s} p1..p99 {np.percentile(v, 1):+7.2f} .. {np.percentile(v, 99):+7.2f} rad"
              f"   std {v.std():6.3f}")

    # geometry and validity come from the long-wavelength parent, but not its own
    # uncertainty layers: they do not describe this combination
    out = {k: zp[k] for k in zp.files if not k.startswith("sigma")}
    out["screen"] = hyb.astype(np.float32)
    out["p_hybrid_long_from"] = "split"
    out["p_hybrid_cut_km"] = float(cut_km)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **out)
    print(f"wrote {out_npz}")
    return out_npz


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="the pair's product tag")
    p.add_argument("--unified", required=True,
                   help="the route root holding offsets/ and split/")
    p.add_argument("--offsets", required=True,
                   help="the offset lattice npz, for the along-track row spacing")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cut-km", type=float, default=32.0,
                   help="along-track wavelength above which the screen follows split and "
                        "below which it follows the azimuth offsets (default 32)")
    a = p.parse_args(argv)

    split_npz = os.path.join(a.unified, "split", f"iono_screen_{a.tag}.npz")
    if not os.path.exists(split_npz):
        raise SystemExit(f"no split screen at {split_npz}")
    return build(os.path.join(a.unified, "offsets", f"iono_screen_{a.tag}.npz"),
                 split_npz, a.offsets,
                 os.path.join(a.out_dir, f"iono_screen_{a.tag}.npz"), a.cut_km)


if __name__ == "__main__":
    main()
