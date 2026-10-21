import numpy as np
import pytest

from simplezarr.utils.npzarrarray import NumpyZarrArray
from simplezarr.utils.write_multiscale import PyramidData


def test_numpy_zarr_array():
    # A read-only, zero-copy ZarrArray-compatible view on a numpy array
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    arr = NumpyZarrArray(a)

    assert arr.shape == (2, 3, 4)
    assert arr.ndim == 3
    assert arr.dtype == "float32"
    # One chunk covering the whole array, like ZarrArray.create() defaults to
    assert arr.chunk_shape == (2, 3, 4)
    # The same zero fill_value and default codecs as ZarrArray.create()
    assert arr._fill_value == 0.0
    assert arr._codecs == [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "zstd", "configuration": {"level": 7, "checksum": True}},
    ]

    # Reads return views on the data, no copy is made
    view = arr[...].get_now()
    assert np.shares_memory(view, a)
    assert np.allclose(view, a)
    sub = arr[1, 1:3, 2:4].get_now()
    assert np.shares_memory(sub, a)
    assert np.allclose(sub, a[1, 1:3, 2:4])
    assert arr[...].get_soon().result().shape == (2, 3, 4)

    # Writing is not possible
    with pytest.raises(TypeError):
        arr[...].set_now(a)
    with pytest.raises(TypeError):
        arr[...].set_soon(a)

    with pytest.raises(TypeError):
        NumpyZarrArray([1, 2, 3])


def test_pyramiddata_numpy_no_copy():
    # PyramidData wraps a numpy array in a read-only zero-copy view
    a = np.zeros((8, 8), np.float32)
    p = PyramidData(a)
    assert isinstance(p.src_array, NumpyZarrArray)
    assert np.shares_memory(p.src_array[...].get_now(), a)

    # It is a view: changes in the source data are visible
    a[0, 0] = 42.0
    assert p.src_array[...].get_now()[0, 0] == 42.0

    # The defaults are inherited, just like from a plain ZarrArray
    assert p.dtype == "float32"
    assert p.fill_value == 0.0
    assert isinstance(p.codecs, list)

    # A NumpyZarrArray can also be passed directly
    p2 = PyramidData(NumpyZarrArray(a))
    assert isinstance(p2.src_array, NumpyZarrArray)
    assert np.shares_memory(p2.src_array[...].get_now(), a)


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
