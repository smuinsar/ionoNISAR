"""Read geo2rdr/{range,azimuth}.off, raw ENVI or compressed GeoTIFF alike."""

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


class _Rows:
    """Row-addressable view of a 2-D raster, read through GDAL on demand."""

    def __init__(self, path):
        self._ds = gdal.Open(path)
        if self._ds is None:                      # UseExceptions makes this unreachable,
            raise OSError(f"cannot open {path}")  # but the message is worth keeping
        self._b = self._ds.GetRasterBand(1)
        self.shape = (self._ds.RasterYSize, self._ds.RasterXSize)
        self.dtype = np.dtype(gdal.GetDataTypeName(self._b.DataType).lower())
        self.path = path

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        na, nr = self.shape
        if isinstance(key, tuple):                # off[i0:i1, :] -- the second axis is whole
            key, second = key[0], key[1]
            if second not in (slice(None), Ellipsis):
                raise IndexError(f"{self.path}: only whole rows are read; got {second!r}")
        if isinstance(key, slice):
            i0, i1, step = key.indices(na)
            if step != 1:
                raise IndexError(f"{self.path}: row slices must be contiguous")
            n = max(0, i1 - i0)
            if n == 0:
                return np.empty((0, nr), self.dtype)
            return self._b.ReadAsArray(0, i0, nr, n)
        i = int(key)
        if i < 0:
            i += na
        if not 0 <= i < na:
            raise IndexError(f"{self.path}: row {key} out of range for {na}")
        return self._b.ReadAsArray(0, i, nr, 1)[0]


def rows(path, expect_shape=None):
    """Open `path` for row-wise reading.  `expect_shape`, if given, is asserted."""
    r = _Rows(path)
    if expect_shape is not None and tuple(expect_shape) != r.shape:
        raise ValueError(f"{path} is {r.shape}, not the expected {tuple(expect_shape)}")
    return r
