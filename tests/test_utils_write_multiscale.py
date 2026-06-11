from simplezarr import MemoryStore, open_zarr

from simplezarr.utils.write_multiscale import write_ome_zarr_pyramid

import numpy as np
import pytest


def test_write_multiscale_even():
    s = MemoryStore()

    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 8, np.float32)
    write_ome_zarr_pyramid(s, "", a, "yx", nlevels=3)

    a0 = open_zarr(s, "level0")[...].get_now()
    a1 = open_zarr(s, "level1")[...].get_now()
    a2 = open_zarr(s, "level2")[...].get_now()

    with pytest.raises(IOError):
        open_zarr(s, "level3")

    assert a0.shape == (8, 8)
    assert a1.shape == (4, 4)
    assert a2.shape == (2, 2)

    assert np.all(a0[0] == range(1, 9))
    assert np.all(a1[0] == [1.5, 3.5, 5.5, 7.5])
    assert np.all(a2[0] == [2.5, 6.5])


def test_write_multiscale_uneven():
    s = MemoryStore()

    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] * 11, np.float32)
    write_ome_zarr_pyramid(s, "", a, "yx", nlevels=3)

    a0 = open_zarr(s, "level0")[...].get_now()
    a1 = open_zarr(s, "level1")[...].get_now()
    a2 = open_zarr(s, "level2")[...].get_now()

    with pytest.raises(IOError):
        open_zarr(s, "level3")

    assert a0.shape == (11, 11)
    assert a1.shape == (5, 5)
    assert a2.shape == (2, 2)

    assert np.all(a0[0] == range(1, 12))
    assert np.all(a1[0] == [1.5, 3.5, 5.5, 7.5, 9.5])
    assert np.all(a2[0] == [2.5, 6.5])


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
