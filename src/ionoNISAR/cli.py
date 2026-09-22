"""Command line entry point: one subcommand per stage of the chain."""
from __future__ import annotations

import argparse
import csv
import os
import sys

from . import config


def _pair(args):
    """The pair configuration a stage is to run on."""
    if not args.config:
        raise SystemExit("this stage needs --config <pair.yaml>")
    return config.load(args.config)


# --------------------------------------------------------------------------- stage A

def cmd_download(args):
    from . import download
    return download.main(["--track", str(args.track), "--frame", str(args.frame),
                          "--dates", *args.dates, "--level", args.level,
                          "--out", args.out, "--workers", str(args.workers)])


def cmd_stack(args):
    from . import gslc_stack
    gslc_stack.build_cache(args.cache, args.track, args.frame, args.dates,
                           pol=args.pol, spacing=args.spacing, refresh=args.refresh)
    return 0


def cmd_streak_index(args):
    from . import streak_index
    rows = streak_index.measure_pairs(args.cache, args.track, args.frame, args.dates,
                                      pol=args.pol, spacing=args.spacing, rebin=args.rebin,
                                      freqs=tuple(args.freq))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


def cmd_select_pair(args):
    """Rank a streak-index table and name the pair to process."""
    with open(args.csv) as fh:
        rows = [r for r in csv.DictReader(fh) if r["freq"] == args.freq]
    if not rows:
        raise SystemExit(f"{args.csv} has no frequency {args.freq} rows")
    rows.sort(key=lambda r: float(r["R_fft"]), reverse=True)
    print(f"{'rank':>4s} {'d1':>9s} {'d2':>9s} {'R_FFT':>7s} {'theta':>7s} "
          f"{'D_RA':>7s}  fill direction")
    for i, r in enumerate(rows[:args.top], 1):
        d_ra = float(r["D_RA"])
        print(f"{i:4d} {r['d1']:>9s} {r['d2']:>9s} {float(r['R_fft']):7.3f} "
              f"{float(r['theta_fft']):+7.1f} {d_ra:+7.3f}  "
              f"{'range' if d_ra > 0 else 'azimuth'}")
    best = rows[0]
    print(f"\nhighest R_FFT at frequency {args.freq}: {best['d1']} / {best['d2']}")
    if float(best["R_fft"]) < 0.5:
        print("NOTE: R_FFT below about 0.5 usually means the loss is not organised into "
              "bands; look at the coherence before committing to the RSLC chain")
    return 0


# --------------------------------------------------------------------------- inputs

def cmd_dem(args):
    from . import dem, grid
    c = _pair(args)
    if not c.ref:
        raise SystemExit(f"no reference granule for {c.dates[0]} in {c.granules}/")
    lon, lat = zip(*grid.bounding_polygon(c.ref))
    b = [min(lon) - 0.25, min(lat) - 0.25, max(lon) + 0.25, max(lat) + 0.25]
    return dem.main(["--bounds", *[f"{v:.4f}" for v in b], "--output", c.dem])


def cmd_grid(args):
    from . import grid
    c = _pair(args)
    return grid.main(["--from", c.ref, "--out", c.grid, "--epsg", str(c.grid_epsg),
                      "--posting", str(c.posting)])


def cmd_glacier_mask(args):
    """Rasterise the RGI glacier complexes onto the pair's map grid."""
    import numpy as np
    from osgeo import gdal

    from . import masks
    c = _pair(args)
    gdal.UseExceptions()
    a = argparse.Namespace(cache_dir=c.cache, rgi_region=args.rgi_region,
                           mask_buffer=args.buffer_m)
    ice = masks.glacier_mask(c.grid, a)
    src = gdal.Open(c.grid)
    ds = gdal.GetDriverByName("GTiff").Create(
        c.glacier_mask, src.RasterXSize, src.RasterYSize, 1, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(src.GetGeoTransform())
    ds.SetProjection(src.GetProjection())
    ds.GetRasterBand(1).WriteArray(ice.astype(np.uint8))
    ds = None
    print(f"wrote {c.glacier_mask}: {100 * ice.mean():.2f} % ice")
    return 0


def cmd_check(args):
    c = _pair(args)
    return 1 if config.preflight(c, need_gpu=not args.no_gpu) else 0


# --------------------------------------------------------------------------- stage B

def cmd_coregister(args):
    from . import coregister
    c = _pair(args)
    for what, p in (("reference", c.ref), ("secondary", c.sec), ("DEM", c.dem),
                    ("map grid", c.grid)):
        if not p or not os.path.exists(p):
            raise SystemExit(f"the {what} is missing ({p}); see the download/dem/grid stages")
    return coregister.main([
        "--ref", c.ref, "--sec", c.sec, "--dem", c.dem,
        "--grid-like", c.grid, "--crop-to-data",
        "--freq", "A", "--pol", c.pol,
        "--gross", "geom", "geom", "--coreg", "geometric",
        "--search", "64", "64", "--skip", "16", "16", "--winsize", "64", "64",
        "--snr-min", "8", "--units", "both",
        "--mask-water", "--mask-glacier", "--mask-buffer", "160", "--mask-where", "both",
        "--rubbersheet-az", "1", "--iono-screen",
        "--ifg", "--ifg-geocode", "--ifg-posting", str(c.posting), "--ifg-keep-rdr-npz",
        "--ifg-looks", str(c.looks[0]), str(c.looks[1]),
        "--fill-trust-km", "8", "--fill-trust-km-flat", "1.5",
        "--keep-scratch", "--gpus", c.gpus, "--tag", c.tag,
        "--out-dir", c.coreg_out, "--scratch", c.scratch,
    ] + config.coreg_extra(c) + args.extra)


def cmd_screen(args):
    from . import routes
    c = _pair(args)
    argv = [
        "--ref", c.ref, "--sec", c.sec, "--dem", c.dem,
        "--scratch", c.scratch, "--offsets", c.offsets,
        "--tag", c.tag, "--looks", str(c.looks[0]), str(c.looks[1]),
        "--posting", str(c.posting), "--grid-like", c.grid,
        "--mask-water", "--gpu", c.gpus, "--out", c.routes_out,
        "--hybrid-cut-km", str(c.hybrid_cut_km),
        "--routes", *args.routes,
    ]
    if os.path.exists(c.glacier_mask):
        argv += ["--split-glacier-mask", c.glacier_mask]
    if c.split.get("unwrap_diff") is False:
        argv += ["--no-split-unwrap-diff"]
    if c.split.get("gate_bracket"):
        argv += ["--split-gate-bracket", *[str(v) for v in c.split["gate_bracket"]]]
    if c.split.get("coh_thresh_b") is not None:
        argv += ["--split-coh-thresh-b", str(c.split["coh_thresh_b"])]
    if c.split.get("unwrap_method"):
        argv += ["--split-unwrap-method", str(c.split["unwrap_method"])]
    return routes.main(argv + args.extra)


def cmd_refocus(args):
    from . import refocus
    return refocus.main(args.extra or ["--selftest"])


def cmd_run(args):
    """The whole chain for one pair, from the granules to the corrected interferogram."""
    c = _pair(args)
    if config.preflight(c, need_gpu=True):
        raise SystemExit("preflight failed; nothing was run")
    for step, fn in (("coregister", cmd_coregister), ("screen", cmd_screen)):
        print(f"\n{'=' * 70}\n{step}\n{'=' * 70}", flush=True)
        rc = fn(args)
        if rc:
            raise SystemExit(f"{step} failed ({rc})")
    return 0


# --------------------------------------------------------------------------- parser

def build_parser():
    p = argparse.ArgumentParser(prog="ionoNISAR", description=__doc__)
    p.add_argument("--config", metavar="YAML", help="the pair configuration file")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        s.set_defaults(fn=fn)
        return s

    s = add("download", cmd_download, "fetch GSLC or RSLC granules from ASF")
    s.add_argument("--track", type=int, required=True)
    s.add_argument("--frame", type=int, required=True)
    s.add_argument("--dates", nargs="+", required=True)
    s.add_argument("--level", default="RSLC", choices=("GSLC", "RSLC"))
    s.add_argument("--out", default="cache/granules")
    s.add_argument("--workers", type=int, default=8)

    s = add("stack", cmd_stack, "accumulate a GSLC stack onto one lattice")
    s.add_argument("--track", type=int, required=True)
    s.add_argument("--frame", type=int, required=True)
    s.add_argument("--dates", nargs="+", required=True)
    s.add_argument("--pol", default="HH")
    s.add_argument("--cache", default="cache/granules")
    s.add_argument("--spacing", type=float, default=240.0)
    s.add_argument("--refresh", action="store_true")

    s = add("streak-index", cmd_streak_index, "R_FFT and theta for every pair in a stack")
    s.add_argument("--track", type=int, required=True)
    s.add_argument("--frame", type=int, required=True)
    s.add_argument("--dates", nargs="+", required=True)
    s.add_argument("--pol", default="HH")
    s.add_argument("--cache", default="cache/granules")
    s.add_argument("--spacing", type=float, default=240.0)
    s.add_argument("--rebin", type=int, default=2, help="cell = spacing x rebin (default 480 m)")
    s.add_argument("--freq", nargs="+", default=["A", "B"])
    s.add_argument("--out", default="gslc_streak/streak_index.csv")

    s = add("select-pair", cmd_select_pair, "rank a streak-index table and name the pair")
    s.add_argument("csv")
    s.add_argument("--freq", default="A")
    s.add_argument("--top", type=int, default=5)

    add("dem", cmd_dem, "fetch the DEM covering the pair")
    add("grid", cmd_grid, "build the output map grid")

    s = add("glacier-mask", cmd_glacier_mask, "rasterise RGI glaciers onto the map grid")
    s.add_argument("--rgi-region", default=None)
    s.add_argument("--buffer-m", type=float, default=160.0)

    s = add("check", cmd_check, "report what would stop this pair")
    s.add_argument("--no-gpu", action="store_true")

    s = add("coregister", cmd_coregister,
            "geometric coregistration, dense offsets, rubbersheet, refocus, interferogram")
    s.add_argument("extra", nargs="*", help="extra arguments passed straight through")

    s = add("screen", cmd_screen, "estimate the phase screens and apply them")
    s.add_argument("--routes", nargs="+", default=["hybrid"],
                   choices=["offsets", "split", "hybrid"])
    s.add_argument("extra", nargs="*")

    s = add("refocus", cmd_refocus, "Doppler-dependent refocusing (--selftest with no argument)")
    s.add_argument("extra", nargs="*")

    s = add("run", cmd_run, "coregister and screen one pair end to end")
    s.add_argument("--routes", nargs="+", default=["hybrid"],
                   choices=["offsets", "split", "hybrid"])
    s.add_argument("extra", nargs="*")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not hasattr(args, "extra"):
        args.extra = []
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
