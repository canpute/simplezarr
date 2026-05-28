from concurrent.futures import Future

from simplezarr import MemoryStore, open_zarr, ZarrArray, ZarrSubArray
from simplezarr.indexing import normalize_selection

import numpy as np
import pytest


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


def test_normalize_selection():
    shape = (100, 120)

    # Full size

    s = normalize_selection((...), shape)
    assert s == (slice(0, 100, 1), slice(0, 120, 1))

    s = normalize_selection((slice(None), slice(None)), shape)
    assert s == (slice(0, 100, 1), slice(0, 120, 1))

    # Open ended

    s = normalize_selection((slice(20), slice(100, None)), shape)
    assert s == (slice(0, 20, 1), slice(100, 120, 1))

    # Negative

    s = normalize_selection((slice(-20), slice(-100, None)), shape)
    assert s == (slice(0, 80, 1), slice(20, 120, 1))

    s = normalize_selection((slice(-40, -20), slice(-20, -40)), shape)
    assert s == (slice(60, 80, 1), slice(100, 100, 1))

    # Ints

    s = normalize_selection((3, 4), shape)
    assert s == (3, 4)

    s = normalize_selection((-3, -4), shape)
    assert s == (97, 116)

    # Steps

    s = normalize_selection((slice(None, None, 2), slice(10, 80, 3)), shape)
    assert s == (slice(0, 100, 2), slice(10, 80, 3))

    # Fails

    with pytest.raises(IndexError):  # No floats
        normalize_selection((0, 1.2), shape)
    with pytest.raises(IndexError):  # No floats
        normalize_selection((0, slice(1.2)), shape)
    with pytest.raises(IndexError):  # only one ellipsis
        normalize_selection((..., ...), shape)
    with pytest.raises(IndexError):  # ndim mismatch
        normalize_selection((3,), shape)
    with pytest.raises(IndexError):  # ndim mismatch
        normalize_selection((3, 4, 5), shape)
    with pytest.raises(IndexError):  # step cannot be zero
        normalize_selection((0, slice(0, 100, 0)), shape)
    with pytest.raises(IndexError):  # step cannot be neg
        normalize_selection((0, slice(0, 100, -1)), shape)
    with pytest.raises(IndexError):  # int index out of range
        normalize_selection((0, 1000), shape)
    with pytest.raises(IndexError):  # int index out of range
        normalize_selection((0, -1000), shape)


def test_indexing_read():

    store = MemoryStore(store_data.copy())
    arr = open_zarr(store)
    assert isinstance(arr, ZarrArray)

    # Read the whole array
    sub = arr[...]
    assert isinstance(sub, ZarrSubArray)
    assert "a[:,:]" in repr(sub)
    assert sub.array is arr
    assert sub.shape == arr.shape
    f = sub.get()
    assert isinstance(f, Future)
    assert "a[:,:].get()" in repr(f)
    a = sub.get_wait()
    assert isinstance(a, np.ndarray)
    assert a.shape == (40, 28)
    assert int(a.max()) == 133

    # Read one whole chunk
    a = arr[:10, :7].get_wait()
    assert np.all(a == 100)

    # And another
    sub = arr[10:20, 14:21]
    a = sub.get_wait()
    assert sub.shape == (10, 7)
    assert np.all(a == 112)

    # And another
    a = arr[30:, 14:21].get_wait()
    assert np.all(a == 132)

    # Read subchunk
    a = arr[12:18, 15:20].get_wait()
    assert a.shape == (6, 5)
    assert np.all(a == 112)

    # Read beyond boundaries
    sub = arr[8:12, 15:20]
    a = sub.get_wait()
    assert sub.shape == (4, 5)
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
    assert arr[0, 0].shape == ()
    a = arr[0, 0].get_wait()
    assert a.shape == ()
    assert a == 100

    # One dimensional
    assert arr[0, :].shape == (28,)
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
    chunk_indices = {x.chunk_index[0] for x in sub._chunk_index_infos}
    assert chunk_indices == {0, 1, 3}  # chunk 2 is never accessed


def test_indexing_write1():
    # Test writing scalars

    store = MemoryStore(store_data.copy())
    arr = open_zarr(store)
    assert isinstance(arr, ZarrArray)

    # This is not allowed
    with pytest.raises(IndexError):
        arr[10:20, 7:14] = 7

    # This also fails
    sub = arr[10:20, 7:14]
    with pytest.raises(TypeError):
        sub.set(None)

    # Write an exact chunk
    f = arr[10:20, 7:14].set(7)
    assert isinstance(f, Future)
    assert "a[10:20,7:14]" in repr(f)
    f.result()  # wait

    a = arr[10:20, 7:14].get_wait()
    assert a.min() == 7
    assert a.max() == 7

    # Write accross chunks
    arr[26:34, 18:26].set_wait(200)

    a = arr[26:34, 18:26].get_wait()
    assert a.min() == 200
    assert a.max() == 200

    # With steps
    arr[26:34:2, 18:26:3].set_wait(202)

    a = arr[26:34, 18:26].get_wait()
    assert a.min() == 200
    assert a.max() == 202

    # Write row and scalar
    arr[28, 19:25].set_wait(204)
    arr[32, 22].set_wait(208)

    a = arr[24:36, 16:28].get_wait()
    ref = np.array(
        [
            # 16   17   18   19   20   21   22   23   24   25   26   27
            [122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123],  # 24
            [122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123],  # 25
            [122, 122, 202, 200, 200, 202, 200, 200, 202, 200, 123, 123],  # 26
            [122, 122, 200, 200, 200, 200, 200, 200, 200, 200, 123, 123],  # 27
            [122, 122, 202, 204, 204, 204, 204, 204, 204, 200, 123, 123],  # 28
            [122, 122, 200, 200, 200, 200, 200, 200, 200, 200, 123, 123],  # 29
            [132, 132, 202, 200, 200, 202, 200, 200, 202, 200, 133, 133],  # 30
            [132, 132, 200, 200, 200, 200, 200, 200, 200, 200, 133, 133],  # 31
            [132, 132, 202, 200, 200, 202, 208, 200, 202, 200, 133, 133],  # 32
            [132, 132, 200, 200, 200, 200, 200, 200, 200, 200, 133, 133],  # 33
            [132, 132, 132, 132, 132, 133, 133, 133, 133, 133, 133, 133],  # 34
            [132, 132, 132, 132, 132, 133, 133, 133, 133, 133, 133, 133],  # 35
        ],
        dtype=np.uint8,
    )
    assert np.all(a == ref)


def test_indexing_write2():
    # Test writing arrays

    store = MemoryStore(store_data.copy())
    arr = open_zarr(store)
    assert isinstance(arr, ZarrArray)

    # Write one exact chunk
    arr[10:20, 7:14].set_wait(np.zeros((10, 7), np.uint8))
    a = arr[10:20, 7:14].get_wait()
    assert a.max() == 0

    # Define small patch

    ref = np.array(
        [
            [200, 204, 207, 210],
            [201, 205, 208, 211],
            [203, 206, 209, 212],
            [220, 221, 222, 223],
        ],
        dtype=np.uint8,
    )

    # Write patch inside a chunk
    arr[12:16, 8:12].set_wait(ref)
    a = arr[12:16, 8:12].get_wait()
    assert np.all(a == ref)

    # Write accross chunk
    arr[19:23, 19:23].set_wait(ref)
    a = arr[19:23, 19:23].get_wait()
    assert np.all(a == ref)

    # One row ...

    ref0 = np.array(
        [230, 231, 232, 233],
        dtype=np.uint8,
    )
    ref1 = ref0.reshape(4, 1)
    ref2 = ref0.reshape(1, 4)

    # Vertical

    arr[19:23, 10].set_wait(ref0)
    a = arr[19:23, 10].get_wait()
    assert np.all(a == ref0)

    arr[19:23, 11].set_wait(ref1)
    a = arr[19:23, 11].get_wait()
    assert np.all(a == ref0)

    # Horizontal

    arr[7, 19:23].set_wait(ref0)
    a = arr[7, 19:23].get_wait()
    assert np.all(a == ref0)

    arr[8, 19:23].set_wait(ref2)
    a = arr[8, 19:23].get_wait()
    assert np.all(a == ref0)

    # Fail
    with pytest.raises(IndexError):
        arr[8, 19:23].set_wait(ref1)
    with pytest.raises(IndexError):
        arr[19:23, 11].set_wait(ref2)

    # A fail triggered during chunk writing

    a = np.array(["a", "b", "c", "d"])
    f = arr[19:23, 11].set(a)
    with pytest.raises(ValueError):
        f.result()
    with pytest.raises(ValueError):
        arr[19:23, 11].set_wait(a)


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
