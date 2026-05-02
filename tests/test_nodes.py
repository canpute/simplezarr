from simplezarr.stores import MemoryStore
from simplezarr.nodes import open_zarr, ZarrNode, ZarrGroup, ZarrArray

import numpy as np
import pytest


# Create an in-memory zarr file

store_data = {
    "zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "spam": "ham",
                "eggs": 42
            }
        }
        """.encode(),
    "sub/zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "group"
        }
        """.encode(),
    "sub/array1/zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [8, 6, 4],
            "data_type": "uint16",
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": [8, 6, 4]
                }
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {
                    "separator": "-"
                }
            },
            "codecs": [{
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            }],
            "fill_value": "0"
        }
        """.encode(),
    "sub/array2/zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [100, 100],
            "dimension_names": ["y", "x"],
            "data_type": "float32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": [50, 50]
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
            "fill_value": "0.0",
            "attributes": {
                "foo": true,
                "bar": "apples"
            }
        }
        """.encode(),
    "sub/array1/c/0-0-0": np.full((8, 6, 4), 100, np.uint16).tobytes(),
    "sub/array2/c/0/0": np.full((50, 50), 1.0, np.float32).tobytes(),
    "sub/array2/c/0/1": np.full((50, 50), 2.0, np.float32).tobytes(),
    "sub/array2/c/1/0": np.full((50, 50), 3.0, np.float32).tobytes(),
    "sub/array2/c/1/1": np.full((50, 50), 4.0, np.float32).tobytes(),
}


store = MemoryStore(store_data)
g = open_zarr(store)


def test_zarr_group():
    g = open_zarr(store)

    assert isinstance(g, ZarrGroup)

    # Structure info
    assert len(g.children) == 1
    assert repr(g).count("<Zarr") == 2  # nesting 1
    assert g.get_structure().count("<Zarr") == 4
    g.print_structure()

    # Metadata
    assert isinstance(g.metadata, dict)
    assert isinstance(g.attributes, dict)
    assert g.attributes["spam"] == "ham"
    assert g.attributes["eggs"] == 42
    g.print_metadata()

    # Get sub
    sub = g["sub"]
    assert isinstance(sub, ZarrGroup)

    # Structure info
    assert len(sub.children) == 2
    assert repr(sub).count("<Zarr") == 3

    # Metadata
    assert isinstance(sub.attributes, dict)
    assert not sub.attributes

    # Children
    assert sub["array1"] is g["sub/array1"]
    assert sub["array2"] is g["sub/array2"]

    # Some props ...
    assert g.store is store
    assert g.name == ""
    assert g.path == ""

    assert sub.store is store
    assert sub.name == "sub"
    assert sub.path == "sub"


def test_zarr_array1():
    g = open_zarr(store)
    a1 = g["sub/array1"]

    assert isinstance(a1, ZarrArray)
    assert a1.ndim == 3
    assert a1.shape == (8, 6, 4)
    assert a1.size == 8 * 6 * 4
    assert a1.chunk_shape == (8, 6, 4)
    assert a1.chunk_grid_shape == (1, 1, 1)
    assert a1.chunk_size == 8 * 6 * 4

    chunk1 = a1.get_chunk((0, 0, 0))
    assert isinstance(chunk1, np.ndarray)
    assert chunk1.shape == (8, 6, 4)
    assert chunk1.dtype == np.uint16
    assert np.all(chunk1 == 100)

    assert repr(a1).startswith("<ZarrArray ")
    assert "uint16" in repr(a1)
    assert "8x6x4" in repr(a1)

    # Some props ...
    assert a1.store is store
    assert a1.name == "array1"
    assert a1.path == "sub/array1"


def test_zarr_array2():
    g = open_zarr(store)
    a2 = g["sub/array2"]

    assert isinstance(a2, ZarrArray)
    assert a2.shape == (100, 100)
    assert a2.size == 10000
    assert a2.chunk_shape == (50, 50)
    assert a2.chunk_grid_shape == (2, 2)
    assert a2.chunk_size == 50 * 50

    chunk1 = a2.get_chunk((0, 0))
    assert isinstance(chunk1, np.ndarray)
    assert chunk1.shape == (50, 50)
    assert chunk1.dtype == np.float32
    assert np.all(chunk1 == 1.0)

    chunk2 = a2.get_chunk((0, 1))
    assert isinstance(chunk2, np.ndarray)
    assert chunk2.shape == (50, 50)
    assert chunk2.dtype == np.float32
    assert np.all(chunk2 == 2.0)

    chunk3 = a2.get_chunk((1, 0))
    assert isinstance(chunk3, np.ndarray)
    assert chunk3.shape == (50, 50)
    assert chunk3.dtype == np.float32
    assert np.all(chunk3 == 3.0)

    chunk4 = a2.get_chunk((1, 1))
    assert isinstance(chunk4, np.ndarray)
    assert chunk4.shape == (50, 50)
    assert chunk4.dtype == np.float32
    assert np.all(chunk4 == 4.0)

    # Out of range
    chunk5 = a2.get_chunk((9, 0))
    assert isinstance(chunk5, np.ndarray)
    assert chunk5.shape == (50, 50)
    assert chunk5.dtype == np.float32
    assert np.all(chunk5 == 0.0)  # fill value


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
