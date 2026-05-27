from concurrent.futures import Future

from simplezarr import MemoryStore, open_zarr, ZarrArray, ZarrSubArray

import numpy as np


store_data = {
    "zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [40, 28],
            "dimension_names": ["y", "x"],
            "data_type": "uint8",
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": [10, 7]
                }
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {
                    "separator": "/"
                }
            },
            "codecs": [{
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            }],
            "fill_value": "0.0"
        }
        """.encode(),
    "c/0/0": np.full((10, 7), 100, np.uint8).tobytes(),
    "c/0/1": np.full((10, 7), 101, np.uint8).tobytes(),
    "c/0/2": np.full((10, 7), 102, np.uint8).tobytes(),
    "c/0/3": np.full((10, 7), 103, np.uint8).tobytes(),
    "c/1/0": np.full((10, 7), 110, np.uint8).tobytes(),
    "c/1/1": np.full((10, 7), 111, np.uint8).tobytes(),
    "c/1/2": np.full((10, 7), 112, np.uint8).tobytes(),
    "c/1/3": np.full((10, 7), 113, np.uint8).tobytes(),
    "c/2/0": np.full((10, 7), 120, np.uint8).tobytes(),
    "c/2/1": np.full((10, 7), 121, np.uint8).tobytes(),
    "c/2/2": np.full((10, 7), 122, np.uint8).tobytes(),
    "c/2/3": np.full((10, 7), 123, np.uint8).tobytes(),
    "c/3/0": np.full((10, 7), 130, np.uint8).tobytes(),
    "c/3/1": np.full((10, 7), 131, np.uint8).tobytes(),
    "c/3/2": np.full((10, 7), 132, np.uint8).tobytes(),
    "c/3/3": np.full((10, 7), 133, np.uint8).tobytes(),
}


def test_indexing_read():

    store = MemoryStore(store_data.copy())
    arr = open_zarr(store)
    assert isinstance(arr, ZarrArray)

    # Read the whole array
    sub = arr[...]
    assert isinstance(sub, ZarrSubArray)
    f = sub.get()
    assert isinstance(f, Future)
    a = sub.get_wait()
    assert isinstance(a, np.ndarray)
    assert a.shape == (40, 28)
    assert int(a.max()) == 133

    # Read one whole chunk
    a = arr[:10, :7].get_wait()
    assert np.all(a == 100)

    # And another
    a = arr[10:20, 14:21].get_wait()
    assert np.all(a == 112)

    # And another
    a = arr[30:, 14:21].get_wait()
    assert np.all(a == 132)

    # Read subchunk
    a = arr[12:18, 15:20].get_wait()
    assert a.shape == (6, 5)
    assert np.all(a == 112)

    # Read beyond boundaries
    a = arr[8:12, 15:20].get_wait()
    assert a.shape == (4, 5)
    assert np.all(a[:2] == 102)
    assert np.all(a[2:] == 112)

    # Again
    a = arr[:10, 19:23].get_wait()
    assert a.shape == (10, 4)
    assert np.all(a[:, :2] == 102)
    assert np.all(a[:, 2:] == 103)


def test_indexing_read_singleton():

    store = MemoryStore(store_data.copy())
    arr = open_zarr(store)
    assert isinstance(arr, ZarrArray)

    # Scalar
    a = arr[0, 0].get_wait()
    assert a.shape == ()
    assert a == 100

    # One dimensional
    a = arr[0, :].get_wait()
    assert a.shape == (28,)
    assert [int(i) for i in a][::7] == [100, 101, 102, 103]


def test_indexing_read_step():

    store = MemoryStore(store_data.copy())
    arr = open_zarr(store)
    assert isinstance(arr, ZarrArray)

    for i in range(7):
        a = arr[0, i::7].get_wait()
        assert a.shape == (4,)
        assert list(a) == [100, 101, 102, 103]

    for i in range(10):
        a = arr[i::10, 0].get_wait()
        assert a.shape == (4,)
        assert list(a) == [100, 110, 120, 130]

    # Test with various step sizes

    a0 = arr[:, 0].get_wait()
    assert a0.shape == (40,)

    for step in range(1, 100):
        a = arr[::step, 0].get_wait()
        ref = a0[::step]
        assert a.shape == ref.shape
        assert np.all(a == ref)

    # Test with various step sizes, in other dim

    a0 = arr[0, :].get_wait()
    assert a0.shape == (28,)

    for step in range(1, 100):
        a = arr[0, ::step].get_wait()
        ref = a0[::step]
        assert a.shape == ref.shape
        assert np.all(a == ref)

    # Extra test to make sure chunks are not even loaded
    sub = arr[::19, :]
    chunk_indices = {x[0].chunk_index for x in sub._chunk_index_infos}
    assert chunk_indices == {0, 1, 3}  # chunk 2 is never accessed


def test_indexing_write():

    store = MemoryStore(store_data.copy())
    arr = open_zarr(store)
    assert isinstance(arr, ZarrArray)

    # Write an exact chunk


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
