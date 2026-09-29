# ionoNISAR

[![PyPI](https://img.shields.io/pypi/v/ionoNISAR.svg)](https://pypi.org/project/ionoNISAR/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23029458.svg)](https://doi.org/10.5281/zenodo.23029458)

Correcting the ionosphere in NISAR L-band repeat-pass interferograms.

At L band, the ionosphere affects interferometry in two primary ways. Along-track gradients 
in total electron content produce spatially varying azimuth shifts that misregister the image 
pair and can cause pronounced, banded coherence loss across a frame. At the same time, the 
dispersive ionospheric delay introduces a large-scale phase signal into the interferogram. 
ionoNISAR addresses both effects: it corrects spatially varying azimuth misregistration 
using measured azimuth offsets, applies Doppler-dependent refocusing where the shift varies 
significantly over the synthetic aperture, and removes the ionospheric phase contribution 
using a phase screen that combines azimuth-offset and split-spectrum estimates.

## What it does

1. **Screening.** Accumulate a stack of GSLC products, measure the coherence loss left after
   the stack median is removed, and report `R_FFT` — how concentrated that loss is in a
   single orientation — and `theta`, its orientation in radar coordinates. Pairs whose loss
   is organised into bands are the ones worth the RSLC chain, and this settles the question
   before any RSLC is downloaded.
2. **Registration.** Geometric coregistration from the orbit and a DEM, dense offset
   estimation on a 64 × 64 window lattice, masking of water and glaciers, anisotropic
   filling of the gaps that leaves, and an azimuth rubbersheet folded into the resampling.
3. **Refocusing.** Where the applied azimuth field changes by a pixel or more over the few
   kilometres a synthetic aperture pierces, each Doppler sub-aperture needs a different
   shift. The Doppler-dependent refocus applies the field seen at each sub-aperture's pierce
   point, and is applied only where it measurably gains coherence.
4. **Phase screen.** The *hybrid* screen: long along-track wavelengths from the
   split-spectrum solve, short ones from the integrated azimuth offsets. The azimuth route
   integrates along track, so its noise accumulates as a random walk and it drifts at long
   wavelengths; the split-spectrum solve integrates nothing but amplifies band-phase noise
   by the inverse of a small determinant. Each covers the other's weakness.

## Requirements

One pair needs

- about **55 GB** of RSLC downloads (two granules), plus ~25 GB per GSLC date for screening,
- about **300 GB** of free scratch, most of it the exported flat-binary SLC pair and the
  geometry rasters, which the later stages read back,
- a **CUDA GPU**, and
- **3–9 hours** of wall clock, of which roughly 1.5 h is the coregistration.

CUDA is used through ISCE3 for `Rdr2Geo`, `Geo2Rdr`, `ResampSlc` and `PyCuAmpcor`. **The
dense offset estimation is what CUDA accelerates most** — it is the single largest cost in
the chain, and PyCuAmpcor is single-device, so `--gpus 0,1,2,3` shards the window lattice
across several cards, one process each. The screening stage needs no GPU.

`--keep-scratch` is not optional: the phase-screen stages read the exported pair and the
geometry rasters the coregistration leaves behind. Budget the disk accordingly.

### Software

ISCE3 is the one dependency that does not come from PyPI; install it with conda
(`conda install -c conda-forge isce3`), in an environment with CUDA support. Everything else
installs with the package: numpy, scipy, h5py, GDAL, rasterio, pyproj, matplotlib,
geopandas, shapely, pandas, requests, asf_search, earthaccess, dem_stitcher, snaphu-py,
scikit-image, tqdm and PyYAML.

```
pip install -e .
```

### Credentials

An Earthdata Login is needed for the granules (ASF DAAC), the DEM and the RGI glacier
outlines. Put it in `~/.netrc`:

```
machine urs.earthdata.nasa.gov login <username> password <password>
```

Nothing else needs credentials; the ESA WorldCover water mask is public.

## Using it

Work in one directory per frame. A pair is described by a small YAML file — see
`examples/` for two complete ones.

```
ionoNISAR download --track 87 --frame 57 --level GSLC --dates 20260620 20260702 20260714
ionoNISAR stack    --track 87 --frame 57 --dates 20260620 20260702 20260714
ionoNISAR streak-index --track 87 --frame 57 --dates 20260620 20260702 20260714 \
    --rebin 2 --out gslc_streak/streak_index.csv
ionoNISAR select-pair gslc_streak/streak_index.csv --freq A

ionoNISAR download --track 87 --frame 57 --level RSLC --dates 20260714 20260726
ionoNISAR --config pair.yaml dem
ionoNISAR --config pair.yaml grid
ionoNISAR --config pair.yaml glacier-mask
ionoNISAR --config pair.yaml check

ionoNISAR --config pair.yaml run --routes hybrid
```

`run` is `coregister` followed by `screen`; either can be run on its own, and both skip work
that is already on disk. The `coregister`, `screen`, `refocus` and `run` stages forward any
argument they do not recognise to the tool they drive, so
`ionoNISAR --config pair.yaml coregister --gpus 0,1` works. The other stages reject an
unknown argument rather than passing it on.

`ionoNISAR refocus --selftest` checks the refocusing arithmetic against a synthetic pair and
needs no data.

### Reading R_FFT

`R_FFT` measures how concentrated the coherence loss is in one orientation, **not** how much
coherence was lost. It is a screening statistic: a pair above about 0.5 generally shows
pronounced streaking, and that number is an empirical guide rather than a threshold. Two
caveats change the value itself — the measurement returns nothing for a frame whose rotated
interior is too small or too sparsely filled, and mask holes attenuate it — so values are
comparable between frames of similar valid fraction, not across arbitrary ones.

## Choices worth knowing about

**Unwrapping.** The default is snaphu (`snaphu-py`), with `phass` and `icu` selectable. The
split-spectrum stage carries a consistency gate that compares the unwrapped phi_A against
the wrapped interferogram at the screen scale, and **stops the run** if they disagree, rather
than publishing a screen built on a bad unwrap. If it fires, prefer a different unwrapper or
`--split-phia-from gradient` over relaxing the gate.

**Only the hybrid screen is included here.** The estimators it is built from — the azimuth
offsets and the split-spectrum solve — are available as `--routes offsets` and
`--routes split`, because the hybrid needs both and each is worth inspecting. Other
estimators are out of scope for this package.

**The screen's datum is not observable.** Neither estimator can measure the absolute
dispersive phase, so the screen carries an arbitrary constant. That matters when comparing
screens, not when correcting an interferogram.

## Citing

If this code contributes to published work, please cite the accompanying paper.

## Licence

MIT. See `LICENSE`.
