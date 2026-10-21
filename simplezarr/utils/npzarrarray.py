from __future__ import annotations

from concurrent.futures import Future

import numpy as np
from simplezarr.misc import resolve_fill_value


class NumpyZarrArray:
    """A read-only ZarrArray-compatible view on a numpy array.

    This makes it possible to use a numpy array as a source (e.g. in ``write_ome_zarr_pyramid()``),
    without making a copy of the data: selections are served as views into the given array.
    Reading is the same as on a ZarrArray, but writing is not supported.

    The defaults are the same as for a plain ``ZarrArray.create()``: one chunk
    that covers the whole array, a zero fill_value, and the default codecs.
    """

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"NumpyZarrArray() expected a numpy array, got {data.__class__.__name__} {data}"
            )
        self._data = data

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._data.shape} {self.dtype} at {hex(id(self))}>"

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def dtype(self) -> str:
        return str(self._data.dtype)

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        # ZarrArray.create() uses one chunk that covers the whole array by default
        return self._data.shape

    @property
    def _fill_value(self):
        # The same zero fill_value that ZarrArray.create() resolves to
        return resolve_fill_value(None, self.dtype)[0]

    @property
    def _codecs(self) -> list[dict]:
        # The same default codecs that ZarrArray.create() uses
        return [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 7, "checksum": True}},
        ]

    def __getitem__(self, index) -> _NumpyZarrArrayView:
        return _NumpyZarrArrayView(self._data[index])


class _NumpyZarrArrayView:
    """A selection of a NumpyZarrArray: a view on the underlying numpy data. Read-only."""

    def __init__(self, view: np.ndarray):
        self._view = view

    def __repr__(self):
        return f"<{self.__class__.__name__} at {hex(id(self))}>"

    @property
    def shape(self) -> tuple[int, ...]:
        return self._view.shape

    def get_soon(self) -> Future:
        fut = Future()
        fut.set_result(self._view)
        return fut

    def get_now(self) -> np.ndarray:
        return self._view

    def set_soon(self, value):
        raise TypeError("Cannot write to a NumpyZarrArray; it is read-only.")

    def set_now(self, value):
        raise TypeError("Cannot write to a NumpyZarrArray; it is read-only.")
