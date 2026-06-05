import json

from simplezarr.stores import MemoryStore
from simplezarr.nodes import (
    open_zarr,
    ZarrNode,
    ZarrGroup,
    ZarrArray,
    resolve_fill_value,
)

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
            "fill_value": 0,
            "codecs": [{
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            }]
        }
        """.encode(),
    "sub/array2/zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [100, 100],
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
            "fill_value": "0.0",
            "codecs": [{
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            }],
            "attributes": {
                "foo": true,
                "bar": "apples"
            },
            "dimension_names": ["y", "x"]
        }
        """.encode(),
    "sub/array1/c/0-0-0": np.full((8, 6, 4), 100, np.uint16).tobytes(),
    "sub/array2/c/0/0": np.full((50, 50), 1.0, np.float32).tobytes(),
    "sub/array2/c/0/1": np.full((50, 50), 2.0, np.float32).tobytes(),
    "sub/array2/c/1/0": np.full((50, 50), 3.0, np.float32).tobytes(),
    "sub/array2/c/1/1": np.full((50, 50), 4.0, np.float32).tobytes(),
    "foo/bar": b"extra dirs and files are ignored",
}


store = MemoryStore(store_data)
g = open_zarr(store)


def test_zarr_group():
    g = open_zarr(store)

    assert isinstance(g, ZarrNode)
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

    # This fails
    with pytest.raises(TypeError):
        g[0]
    with pytest.raises(KeyError):
        g["not-a-sub"]

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

    assert isinstance(g, ZarrNode)
    assert isinstance(a1, ZarrArray)
    assert a1.ndim == 3
    assert a1.shape == (8, 6, 4)
    assert a1.size == 8 * 6 * 4
    assert a1.chunk_shape == (8, 6, 4)
    assert a1.chunk_grid_shape == (1, 1, 1)
    assert a1.chunk_size == 8 * 6 * 4
    assert a1.nbytes == 8 * 6 * 4 * 2

    chunk1 = a1.get_chunk_now((0, 0, 0))
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

    chunk1 = a2.get_chunk_now((0, 0))
    assert isinstance(chunk1, np.ndarray)
    assert chunk1.shape == (50, 50)
    assert chunk1.dtype == np.float32
    assert np.all(chunk1 == 1.0)

    chunk2 = a2.get_chunk_now((0, 1))
    assert isinstance(chunk2, np.ndarray)
    assert chunk2.shape == (50, 50)
    assert chunk2.dtype == np.float32
    assert np.all(chunk2 == 2.0)

    chunk3 = a2.get_chunk_now((1, 0))
    assert isinstance(chunk3, np.ndarray)
    assert chunk3.shape == (50, 50)
    assert chunk3.dtype == np.float32
    assert np.all(chunk3 == 3.0)

    chunk4 = a2.get_chunk_now((1, 1))
    assert isinstance(chunk4, np.ndarray)
    assert chunk4.shape == (50, 50)
    assert chunk4.dtype == np.float32
    assert np.all(chunk4 == 4.0)

    # Out of range
    chunk5 = a2.get_chunk_now((9, 0))
    assert isinstance(chunk5, np.ndarray)
    assert chunk5.shape == (50, 50)
    assert chunk5.dtype == np.float32
    assert np.all(chunk5 == 0.0)  # fill value


def test_zarr_getting_and_setting_chunks_basic():
    s = MemoryStore()

    arr = ZarrArray.create(s, "", (100, 100), np.float32, chunk_shape=(10, 10))

    # The array is initially 'empty'

    a = arr.get_chunk_now((0, 0))
    assert np.all(a == 0)

    a = arr.get_chunk_now((0, 0), none_if_missing=True)
    assert a is None

    # Set the first chunk

    arr.set_chunk_now((0, 0), np.ones((10, 10), np.float32))

    a = arr.get_chunk_now((0, 0))
    assert np.all(a == 1)

    # Set the next two chunks, with and without empty-check

    arr.set_chunk_now((0, 1), np.zeros((10, 10), np.float32))
    arr.set_chunk_now((0, 2), np.zeros((10, 10), np.float32), check_empty=False)

    a = arr.get_chunk_now((0, 1))
    assert np.all(a == 0)

    a = arr.get_chunk_now((0, 2))
    assert np.all(a == 0)

    a = arr.get_chunk_now((0, 1), True)
    assert a is None

    a = arr.get_chunk_now((0, 2), True)
    assert a is not None  # the all-zero chunk IS actually stored


def test_zarr_getting_and_setting_chunks_fails():
    s = MemoryStore()

    arr = ZarrArray.create(s, "", (100, 100), np.float32, chunk_shape=(10, 10))

    a = np.ones((10, 10), np.float32)

    # This works

    arr.set_chunk_now((0, 0), a)

    a2 = arr.get_chunk_now((0, 0))
    assert np.all(a2 == 1)

    # Let's check some invalid ways to set a chunk

    with pytest.raises(TypeError):
        arr.set_chunk_now(0, a)
    with pytest.raises(TypeError):
        arr.set_chunk_now([0, 0], a)

    with pytest.raises(IndexError):
        arr.set_chunk_now((0,), a)
    with pytest.raises(IndexError):
        arr.set_chunk_now((0, 0, 0), a)

    with pytest.raises(IndexError):
        arr.set_chunk_now((0.0, 0), a)
    with pytest.raises(IndexError):
        arr.set_chunk_now((-1, 0), a)

    with pytest.raises(TypeError):
        arr.set_chunk_now((0, 0), b"not an array")
    with pytest.raises(ValueError):
        arr.set_chunk_now((0, 0), a.astype(np.float64))
    with pytest.raises(ValueError):
        arr.set_chunk_now((0, 0), a.astype(np.int32))
    with pytest.raises(ValueError):
        arr.set_chunk_now((0, 0), a[1:, 1:])
    with pytest.raises(ValueError):
        arr.set_chunk_now((0, 0), a.reshape(100, 1))

    # Let's check some invalid ways to get a chunk

    with pytest.raises(TypeError):
        arr.get_chunk_now(0)
    with pytest.raises(TypeError):
        arr.get_chunk_now([0, 0])

    with pytest.raises(IndexError):
        arr.get_chunk_now((0,))
    with pytest.raises(IndexError):
        arr.get_chunk_now((0, 0, 0))

    with pytest.raises(IndexError):
        arr.get_chunk_now((0.0, 0))
    with pytest.raises(IndexError):
        arr.get_chunk_now((-1, 0))


def test_zarr_getting_and_setting_chunks_parallel():
    s = MemoryStore()

    arr = ZarrArray.create(s, "", (100, 100), np.float32, chunk_shape=(10, 10))

    # Write
    promises = []
    values = []
    for y in range(10):
        for x in range(10):
            val = y * 10 + x
            a = np.full((10, 10), val, np.float32)
            p = arr.set_chunk_soon((y, x), a)
            promises.append(p)
            values.append(val)

    # Wait
    import concurrent

    concurrent.futures.wait(promises)
    for p in promises:
        p.result()

    # Read
    promises = []
    for y in range(10):
        for x in range(10):
            p = arr.get_chunk_soon((y, x))
            promises.append(p)

    # Wait and check
    for p, val in zip(promises, values, strict=True):
        a = p.result()
        assert np.all(a == val)


def test_zarr_replicate_hardcoded_store():
    # Use .create() to replicate the hardcoded store at the top of the module

    s = MemoryStore()

    ZarrGroup.create(s, "", attributes={"spam": "ham", "eggs": 42})
    ZarrGroup.create(s, "sub")

    codecs1 = [{"name": "bytes", "configuration": {"endian": "little"}}]
    a1 = ZarrArray.create(
        s, "sub/array1", (8, 6, 4), "uint16", chunk_path_separator="-", codecs=codecs1
    )

    codecs2 = codecs1
    a2 = ZarrArray.create(
        s,
        "sub/array2",
        (100, 100),
        np.float32,
        chunk_path_separator=None,
        chunk_shape=(50, 50),
        fill_value="0.0",  # can be str
        codecs=codecs2,
        dimension_names=["y", "x"],
        attributes={"foo": True, "bar": "apples"},
    )

    a1[...].set_now(100)
    a2[:50, :50].set_now(1)
    a2[:50, 50:].set_now(2)
    a2[50:, :50].set_now(3)
    a2[50:, 50:].set_now(4)

    # ... now compare with the reference at the top of this file

    ref_data = store_data
    this_data = s.dict

    for key, ref_val in ref_data.items():
        if key == "foo/bar":
            continue
        assert key in this_data
        val = this_data[key]

        if key.endswith(".json"):
            ref_val = json.loads(ref_val.decode())
            ref_val = json.dumps(ref_val, indent=4).encode()

            if val != ref_val:
                lines1 = val.split(b"\n")
                lines2 = ref_val.split(b"\n")
                print(f"--- compare {key} ----")
                for i in range(max(len(lines1), len(lines2))):
                    line1 = lines1[i] if i < len(lines1) else "<eof>"
                    line2 = lines2[i] if i < len(lines2) else "<eof>"
                    if line1 == line2:
                        print(f" {i:>3}", line1)
                    else:
                        print(f"X{i:>3}", line1)
                        print("    ", line2)

        assert val == ref_val


def test_zarr_create_array():
    # Create array with ramdom data, store, read back, verify
    s = MemoryStore()

    a1 = np.random.uniform(size=(50, 50, 50))
    arr1 = ZarrArray.create(s, "", a1.shape, a1.dtype, chunk_shape=(8, 8, 8))
    arr1[...].set_now(a1)

    arr2 = open_zarr(s)
    a2 = arr2[...].get_now()

    assert a1.shape == a2.shape
    assert a1.dtype == a2.dtype
    assert np.all(a1 == a2)


def test_zarr_array_codecs():
    s = MemoryStore()

    byte_codec = {"name": "bytes", "configuration": {"endian": "little"}}
    zstd_codec = {"name": "zstd", "configuration": {"level": 7, "checksum": True}}

    a = np.random.uniform(0, 2**16, size=(1000, 1000)).astype(np.uint16)
    a = np.zeros((1000, 1000), np.uint16)
    a[100:-100, 100:-100] = 100
    a[200:-200, 200:-200] = 1000

    arr = ZarrArray.create(
        s, "", a.shape, a.dtype, chunk_shape=(8, 8), codecs=[byte_codec]
    )
    arr[...].set_now(a)
    assert arr._codecs == [byte_codec]
    nbytes1 = s.nbytes

    arr = ZarrArray.create(s, "", a.shape, a.dtype, chunk_shape=(8, 8), codecs=None)
    arr[...].set_now(a)
    assert arr._codecs == [byte_codec, zstd_codec]
    nbytes2 = s.nbytes

    assert nbytes1 > nbytes2

    with pytest.raises(ValueError):
        ZarrArray.create(s, "", a.shape, a.dtype, chunk_shape=(8, 8), codecs=[])
    with pytest.raises(ValueError):
        ZarrArray.create(s, "", a.shape, a.dtype, chunk_shape=(8, 8), codecs=[42])
    with pytest.raises(ValueError):
        ZarrArray.create(
            s, "", a.shape, a.dtype, chunk_shape=(8, 8), codecs=[{"not_name": "x"}]
        )


def test_resolve_fill_value():
    # Bool

    real, json = resolve_fill_value(None, "bool")
    assert isinstance(real, bool)
    assert not real
    assert not json

    real, json = resolve_fill_value(True, "bool")
    assert isinstance(real, bool)
    assert real
    assert json

    real, json = resolve_fill_value("NO", "bool")
    assert isinstance(real, bool)
    assert not real
    assert json == "NO"

    real, json = resolve_fill_value("yes", "bool")
    assert isinstance(real, bool)
    assert real
    assert json == "yes"

    with pytest.raises(ValueError):
        resolve_fill_value(3, "bool")
    with pytest.raises(ValueError):
        resolve_fill_value("yesplease", "bool")

    # int

    real, json = resolve_fill_value(None, "uint8")
    assert isinstance(real, int)
    assert real == 0
    assert json == 0

    real, json = resolve_fill_value(7, "uint8")
    assert isinstance(real, int)
    assert real == 7
    assert json == 7

    real, json = resolve_fill_value("7", "uint8")
    assert isinstance(real, int)
    assert real == 7
    assert json == "7"

    with pytest.raises(ValueError):
        resolve_fill_value("7.1", "uint8")
    with pytest.raises(ValueError):
        resolve_fill_value((1,), "uint8")

    # float

    real, json = resolve_fill_value(None, "float32")
    assert isinstance(real, float)
    assert real == 0.0
    assert json == 0.0

    real, json = resolve_fill_value(7, "float32")
    assert isinstance(real, float)
    assert real == 7.0
    assert json == 7.0

    real, json = resolve_fill_value("7", "float32")
    assert isinstance(real, float)
    assert real == 7.0
    assert json == "7"

    with pytest.raises(ValueError):
        resolve_fill_value("spam", "float32")
    with pytest.raises(ValueError):
        resolve_fill_value((1,), "float32")

    # complex

    real, json = resolve_fill_value(None, "complex64")
    assert isinstance(real, complex)
    assert real == complex(0, 0)
    assert json == "0j"

    real, json = resolve_fill_value(complex(1, 2), "complex64")
    assert isinstance(real, complex)
    assert real == complex(1, 2)
    assert json == "(1+2j)"

    real, json = resolve_fill_value((3, 4), "complex64")
    assert isinstance(real, complex)
    assert real == complex(3, 4)
    assert json == "(3+4j)"

    real, json = resolve_fill_value("3+4j", "complex64")
    assert isinstance(real, complex)
    assert real == complex(3, 4)
    assert json == "3+4j"

    with pytest.raises(ValueError):
        resolve_fill_value("spam", "complex64")
    with pytest.raises(ValueError):
        resolve_fill_value(3, "complex64")
    with pytest.raises(ValueError):
        resolve_fill_value((1, "spam"), "complex64")

    # raw

    real, json = resolve_fill_value(None, "r2")
    assert isinstance(real, bytes)
    assert real == b"\x00\x00"
    assert json == [0, 0]

    real, json = resolve_fill_value(b"77", "r2")
    assert isinstance(real, bytes)
    assert real == b"77"
    assert json == [55, 55]

    with pytest.raises(ValueError):
        resolve_fill_value(b"123", "r2")
    with pytest.raises(ValueError):
        resolve_fill_value("123", "r2")
    with pytest.raises(ValueError):
        resolve_fill_value(123, "r2")


def test_zarr_fill_values():
    # A bit like before, but just to check the round-trip

    s = MemoryStore()

    value_json = lambda: json.loads(s.dict["zarr.json"].decode())["fill_value"]

    # bool

    arr = ZarrArray.create(s, "", (10,), "bool", fill_value=None)
    assert not arr._fill_value
    assert not value_json()

    arr = ZarrArray.create(s, "", (10,), "bool", fill_value=True)
    assert arr._fill_value
    assert value_json()

    # int

    arr = ZarrArray.create(s, "", (10,), "uint8", fill_value=None)
    assert arr._fill_value == 0
    assert value_json() == 0

    arr = ZarrArray.create(s, "", (10,), "uint8", fill_value=3)
    assert arr._fill_value == 3
    assert value_json() == 3

    # float

    arr = ZarrArray.create(s, "", (10,), "float32", fill_value=None)
    assert arr._fill_value == 0
    assert value_json() == 0

    arr = ZarrArray.create(s, "", (10,), "float32", fill_value=3)
    assert arr._fill_value == 3.0
    assert value_json() == 3.0


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
