"""Download NISAR GSLC or RSLC granules for a track and frame in bulk."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path


def find_granules(track, frame, dates, level="GSLC"):
    """{date: (url, bytes)} from ASF.  Searching needs no credentials; downloading does."""
    import asf_search as asf

    d0, d1 = min(dates), max(dates)
    # end is exclusive-ish, so cover the whole last day; calendar arithmetic, not day+1 on
    # the digits, or a month-end date asks ASF for a 32nd
    end = (datetime.strptime(d1, "%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    res = asf.search(dataset=asf.DATASET.NISAR, relativeOrbit=int(track), frame=int(frame),
                     processingLevel=level,
                     start=datetime.strptime(d0, "%Y%m%d").strftime("%Y-%m-%d"),
                     end=end, maxResults=500)
    out = {}
    for g in res:
        day = str(g.properties["startTime"])[:10].replace("-", "")
        if day in dates and day not in out:
            url = g.properties["url"]
            size = None
            # `bytes` is a dict keyed by filename for NISAR; the .h5 entry is the granule
            b = g.properties.get("bytes")
            if isinstance(b, dict):
                for k, v in b.items():
                    if k.endswith(".h5") and "QA" not in k:
                        size = int(v["bytes"])
            elif isinstance(b, (int, float)):
                size = int(b)
            out[day] = (url, size)
    missing = [d for d in dates if d not in out]
    if missing:
        raise SystemExit(f"no {level} granule for {', '.join(missing)}")
    return out


def download(url, dest, session, chunk=128 * 1024 * 1024, workers=8):
    """Parallel ranged GET with a resumable chunk sidecar.  Returns bytes transferred now."""
    dest = Path(dest)
    if dest.exists():
        return 0
    part, done_f = Path(str(dest) + ".part"), Path(str(dest) + ".done")

    head = session.head(url, allow_redirects=True)
    total = int(head.headers["Content-Length"])
    n = (total + chunk - 1) // chunk

    done = set()
    if part.exists() and done_f.exists():
        done = {int(x) for x in done_f.read_text().split() if x.strip()}
    else:
        done_f.unlink(missing_ok=True)
    with open(part, "r+b" if part.exists() else "wb") as f:
        f.truncate(total)

    todo = [i for i in range(n) if i not in done]
    if not todo:
        part.rename(dest)
        done_f.unlink(missing_ok=True)
        return 0

    lock = threading.Lock()
    state = {"got": 0, "t0": time.time()}
    fd = os.open(part, os.O_WRONLY)

    def grab(i):
        a = i * chunk
        b = min(a + chunk, total) - 1
        for attempt in range(5):
            try:
                r = session.get(url, headers={"Range": f"bytes={a}-{b}"}, stream=True)
                r.raise_for_status()
                off, got = a, 0
                for blk in r.iter_content(8 * 1024 * 1024):
                    os.pwrite(fd, blk, off)
                    off += len(blk)
                    got += len(blk)
                if got != b - a + 1:
                    raise IOError(f"short chunk {i}: {got} != {b - a + 1}")
                with lock:
                    state["got"] += got
                    with open(done_f, "a") as g:
                        g.write(f"{i}\n")
                    el = time.time() - state["t0"]
                    pct = 100.0 * (len(done) + state["got"] / chunk) / n
                    print(f"    {dest.name[:52]}  {pct:5.1f}%  "
                          f"{state['got'] / 1e6:8.0f} MB  {state['got'] / 1e6 / el:5.1f} MB/s",
                          flush=True)
                return
            except Exception as e:                       # noqa: BLE001 -- retry any transport error
                if attempt == 4:
                    raise
                print(f"    chunk {i} retry {attempt + 1}: {e}", flush=True)
                time.sleep(2 * (attempt + 1))

    try:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(grab, todo))
    finally:
        os.close(fd)

    if part.stat().st_size != total:
        raise SystemExit(f"{dest.name}: size {part.stat().st_size} != {total}")
    part.rename(dest)
    done_f.unlink(missing_ok=True)
    return state["got"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", type=int, required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--level", default="GSLC", choices=("GSLC", "RSLC"))
    p.add_argument("--out", default="cache/granules")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel range requests; the endpoint punishes large values")
    p.add_argument("--chunk-mb", type=int, default=128)
    a = p.parse_args(argv)

    import earthaccess

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gran = find_granules(a.track, a.frame, a.dates, a.level)

    earthaccess.login(strategy="netrc")
    session = earthaccess.get_requests_https_session()

    print(f"{a.level} T{a.track:03d} F{a.frame:03d}: {len(gran)} granules -> {out}")
    t0, moved = time.time(), 0
    for d in sorted(gran):
        url, size = gran[d]
        dest = out / url.split("/")[-1]
        if dest.exists():
            print(f"  {d}  cached  {dest.stat().st_size / 1e9:.1f} GB")
            continue
        print(f"  {d}  {size / 1e9 if size else 0:.1f} GB  {dest.name[:60]}")
        moved += download(url, dest, session, a.chunk_mb * 1024 * 1024, a.workers)
    el = time.time() - t0
    print(f"done: {moved / 1e9:.1f} GB in {el / 60:.1f} min"
          f"{f' = {moved / 1e6 / el:.1f} MB/s' if moved else ''}")
    for d in sorted(gran):
        print(f"  {d}  {out / gran[d][0].split('/')[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
