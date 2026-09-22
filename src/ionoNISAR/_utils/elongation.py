"""Measure how range-elongated an offset field is, and what --fill-aniso it implies."""
import argparse
import os

import numpy as np

LAGS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])


def structure_function(f, m, ax, lags=LAGS, min_pairs=1000):
    out = []
    for h in lags:
        if h >= f.shape[ax]:
            out.append(np.nan)
            continue
        a = np.take(f, np.arange(f.shape[ax] - h), axis=ax)
        b = np.take(f, np.arange(h, f.shape[ax]), axis=ax)
        mm = (np.take(m, np.arange(m.shape[ax] - h), axis=ax)
              & np.take(m, np.arange(h, m.shape[ax]), axis=ax))
        out.append(np.median(np.abs(b - a)[mm]) if mm.sum() >= min_pairs else np.nan)
    return np.array(out)


def reach(sfv, target, lags=LAGS):
    """Lag in CELLS at which the structure function first reaches `target`, interpolated."""
    ok = np.isfinite(sfv)
    if not ok.any():
        return np.inf
    k = np.where(ok & (sfv >= target))[0]
    if not k.size:
        return np.inf
    i = k[0]
    if i == 0:
        return float(lags[0])
    x0, x1, y0, y1 = lags[i - 1], lags[i], sfv[i - 1], sfv[i]
    return float(x0 + (x1 - x0) * (target - y0) / max(y1 - y0, 1e-12))


def describe(npz, threshold=0.10, channel="azimuth"):
    z = np.load(npz)
    if "filled" not in z.files:
        raise SystemExit(f"{npz} has no `filled` flag -- point this at the _rbsheet.npz")
    good = ~z["filled"]
    # the measurement relative to geometry, not the smoothed surface that was applied
    f = z[channel].astype(np.float64)
    skip = z["skip"]
    az_sp = float(z["az_spacing"])
    raw = npz.replace("_rbsheet.npz", ".npz")
    rg_sp = float(np.load(raw)["rg_spacing"]) if os.path.exists(raw) else np.nan
    cell_az, cell_rg = skip[0] * az_sp, skip[1] * rg_sp

    ra = reach(structure_function(f, good, 0), threshold)
    rr = reach(structure_function(f, good, 1), threshold)
    el = rr / ra if np.isfinite(rr) and ra > 0 else np.inf
    return dict(name=os.path.basename(npz), shape=f.shape, measured=100 * good.mean(),
                reach_az=ra, reach_rg=rr, elong=el, cell_az=cell_az, cell_rg=cell_rg,
                km_az=ra * cell_az / 1000, km_rg=rr * cell_rg / 1000, channel=channel)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.10, metavar="PX",
                    help="the change the correlation length is measured at (default 0.10 px)")
    ap.add_argument("--channel", default="azimuth", choices=("azimuth", "range", "both"),
                    help="which offset component to measure (default azimuth).  The two are "
                         "NOT alike: an ionospheric azimuth field is strongly range-elongated "
                         "and the range field is 1.5 : 1, i.e. a single bilinear ramp with no "
                         "bands in it.  That is why the fill's prior has to be per channel")
    a = ap.parse_args(argv)

    chans = ("azimuth", "range") if a.channel == "both" else (a.channel,)
    print(f"elongation of the measured offset field, at a {a.threshold:g} px change\n")
    rows = [(p, c) for p in a.npz for c in chans]
    print(f"  {'field':40s}{'chan':>8s}{'% meas':>8s}{'lag az':>9s}{'lag rg':>9s}"
          f"{'km az':>8s}{'km rg':>8s}{'elong':>8s}{'--fill-aniso':>14s}")
    for p, c in rows:
        try:
            d = describe(p, a.threshold, c)
        except Exception as exc:                       # a sibling frame may be mid-run
            print(f"  {os.path.basename(p):40s}{c:>8s}  {exc}")
            continue
        rg = f"{d['reach_rg']:9.0f}" if np.isfinite(d["reach_rg"]) else f"{'>2048':>9s}"
        km = f"{d['km_rg']:8.1f}" if np.isfinite(d["km_rg"]) else f"{'-':>8s}"
        el = f"{d['elong']:8.1f}" if np.isfinite(d["elong"]) else f"{'>' + str(int(2048 / max(d['reach_az'], 1))):>8s}"
        if not np.isfinite(d["elong"]):
            rec = f"{'>16':>14s}"
        elif d["elong"] >= 1.0:
            rec = f"{max(2.0, round(d['elong'])):14.0f}"
        else:                       # azimuth-elongated: the reciprocal weight, not a floor
            rec = f"{'1/' + str(max(2, round(1 / d['elong']))):>14s}"
        print(f"  {d['name']:40s}{d['channel']:>8s}{d['measured']:8.1f}{d['reach_az']:9.0f}{rg}"
              f"{d['km_az']:8.1f}{km}{el}{rec}")
    print("\nelongation is in LATTICE CELLS, the unit --fill-aniso lives in.  The rule "
          "--fill-aniso ~= elongation\nwas calibrated by withheld discs; "
          "re-check it with a disc test on a very different geometry.")


if __name__ == "__main__":
    main()
