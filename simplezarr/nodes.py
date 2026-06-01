"""
Zarr files are made up of a tree of nodes. Each node is either a ``ZarrGroup`` or a ``ZarrArray``. The arrays are the leaf nodes.
"""

from __future__ import annotations  # Using class names for types without Ruff F821

import json
import math
from concurrent.futures import Future

import numpy as np

from .misc import executor
from .stores import BaseStore, ReadableStore, WritableStore, ListableStore
from .codecs import create_ndarray_type, encode_array, decode_bytes
from .indexing import ZarrArraySlice, ChunkGridIndexer


__all__ = [
    "open_zarr",
    "ZarrNode",
    "ZarrGroup",
    "ZarrArray",
]


def open_zarr(store: ReadableStore) -> ZarrNode:
    """Open a zarr file using the given store."""
    return ZarrNode._from_path(store, "")


def join(*path_parts):
    return "/".join(path_parts).lstrip("/")


class ZarrNode:
    """The base class for ``ZarrGroup`` and ``ZarrArray``.

    A zarr file is made up of nodes, where arrays are the lead nodes.
    Each node is represented by a 'directory' and a corresponding 'zarr.json'
    that contains information about the node.
    """

    def __init__(
        self,
        store: ReadableStore | ListableStore | WritableStore,
        path: str,
        metadata: dict | None = None,
    ):
        # Check path
        if not isinstance(path, str):  # no-cover
            raise TypeError(f"{self.__class__.__name__} path must be str, got {path!r}")
        path = path.lstrip("/")
        if path.endswith("/"):  # no-cover
            raise ValueError(
                f"{self.__class__.__name__} path must not end with '/' unless root, got {path!r}"
            )

        self._store = store
        self._path = path
        self._name = self._path.rsplit("/", 1)[-1]

        assert isinstance(metadata, dict)
        self._metadata = metadata
        self._parse_metadata()

        self._init_node()

    def __repr__(self):
        return self._one_line_repr()

    @classmethod
    def _from_path(cls, store: BaseStore, path: str) -> ZarrNode:
        json_text = store.get(join(path, "zarr.json")).decode()
        metadata = json.loads(json_text)

        if metadata["zarr_format"] != 3:  # no-cover
            raise RuntimeError("Assuming Zarr version 3")

        node_type = metadata["node_type"]
        if node_type == "group":
            return ZarrGroup(store, path, metadata=metadata)
        elif node_type == "array":
            return ZarrArray(store, path, metadata=metadata)
        else:  # no-cover
            raise RuntimeError(f"Unexpected node type {node_type!r}")

    @property
    def store(self) -> BaseStore:
        """The store for this node."""
        return self._store

    @property
    def name(self) -> str:
        """The name of this node."""
        return self._name

    @property
    def path(self) -> str:
        """The full path of this node in the store."""
        return self._path

    @property
    def metadata(self) -> dict:
        """The metadata as a dictionary."""
        return self._metadata

    def print_metadata(self):
        """Print a readable representation of the metadata."""
        print(json.dumps(self._metadata, indent=4))

    def _one_line_repr(self):  # no-cover
        return f"<{self.__class__.__name__} '{self._path}' at {hex(id(self))}>"

    def _parse_metadata(self):
        raise NotImplementedError()

    def _init_node(self):
        raise NotImplementedError()


class ZarrGroup(ZarrNode):
    """The class that represents a group in a Zarr file.

    The ``repr()`` of a group shows its children. One can navigate
    the Zarr file by indexing:

        zarr_group['path/to/node']

    """

    def __repr__(self):
        return self.get_structure(max_depth=1)

    def _one_line_repr(self):
        return f"<{self.__class__.__name__} '{self._path}' with {len(self._children)} children at {hex(id(self))}>"

    @property
    def children(self) -> tuple[ZarrNode]:
        """The child nodes of this group. These can be groups or arrays."""
        return tuple(self._children.values())

    @property
    def attributes(self) -> dict:
        """The attributes of this group. I.e. ``metadata["attributes"]``"""
        return self._attributes

    def print_structure(self, max_depth: int = 999):
        """Print the structure of the Zarr file from this group and below."""
        print(self.get_structure(max_depth=max_depth))

    def get_structure(self, max_depth: int = 999, indent: int = 0) -> str:
        """Get the structure of this group as a human-readble string."""
        max_depth = int(max_depth)
        indent = int(indent)

        indent_str = " " * indent
        r = indent_str + self._one_line_repr()
        if self._children and max_depth > 0:
            for child in self.children:
                r += "\n"
                if isinstance(child, ZarrGroup):
                    r += child.get_structure(max_depth - 1, indent + 4)
                else:
                    r += " " * (indent + 4) + child._one_line_repr()
        return r

    def __getitem__(self, path):
        if not isinstance(path, str):
            raise TypeError("ZarrGroup indexing must be done with a str path.")

        name, _, remaining_path = path.rstrip("/").partition("/")

        try:
            ob = self._children[name]
        except KeyError:
            raise KeyError(
                f"ZarrGroup '{self._path}' does not have a child named {name!r}."
            ) from None

        if remaining_path:
            return ob[remaining_path]
        else:
            return ob

    def _parse_metadata(self):
        meta = self._metadata

        # Parse mandatory fields
        assert meta["node_type"] == "group"

        # Parse optional fields
        self._attributes = meta.get("attributes", {})

    def _init_node(self):
        # Assume ListableStore

        # todo: use consolidated metadata
        # todo: use list_dir only lazily

        n = len(self._path)
        items = self._store.list_dir(self._path + "/")
        dir_names = [item[n:].strip("/") for item in items if item.endswith("/")]

        self._children = {}
        for name in dir_names:
            try:
                node = ZarrNode._from_path(self._store, f"{self._path}/{name}")
            except IOError:
                continue
            else:
                self._children[name] = node


class ZarrArray(ZarrNode):
    """The class that represents a Zarr array.

    These arrays don't contain any bytes themselves, but are used as proxies
    to load data from the store, and provide these as numpy arrays.
    """

    def _one_line_repr(self):
        shape_str = "x".join(str(i) for i in self.shape)
        return f"<{self.__class__.__name__} '{self._path}' {shape_str} {self.dtype} at {hex(id(self))}>"

    @property
    def dtype(self) -> str:
        """The datatype of the array.

        Possible values include 'bool', 'int8', 'int16', 'int32', 'int64',
        'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64',
        'complex64', 'complex128', 'rx' (with x a multiple of 8).
        """
        return self._dtype

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array (includes spatial, time, and channel dimensions)."""
        return len(self._shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array (ndim elements)."""
        return self._shape

    @property
    def size(self) -> int:
        """The size of the array, expressed in number of elements."""
        return int(np.prod(self._shape))

    @property
    def nbytes(self) -> int:
        """The size of the array in bytes (uncompressed)."""
        return int(self.size * self._dtype_bits / 8)

    @property
    def chunk_grid_shape(self) -> tuple[int, ...]:
        """The shape of the chunk grid (ndim elements)."""
        return self._chunk_grid_shape

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """The shape of each chunk (ndim elements)."""
        return self._chunk_shape

    @property
    def chunk_size(self) -> int:
        """The size of each chunk, in number of elements."""
        return int(np.prod(self._chunk_shape))

    @property
    def chunk_nbytes(self) -> int:
        """The size of each chunk in (uncompressed) bytes."""
        return int(self.chunk_size * self._dtype_bits / 8)

    @property
    def chunks(self):
        return ChunkGridIndexer(self)

    def __getitem__(self, selection) -> ZarrArraySlice:
        return ZarrArraySlice(self, selection)

    def __setitem__(self, *args):  # co-cover
        raise IndexError(
            "ZarrArray does not support index assignment (``a[..] = foo``), instead use ``a[..].set_now(foo)`` or ``a[..].set_soon(foo)``."
        )

    def get_chunk_now(self, index) -> np.ndarray:
        """Read a chunk from the store.

        This function is synchronous; you may want to use ``get_chunk_soon()``
        to do the loading and decompression in a separate thread.

        Converts the index to the path for that chunk, load the bytes
        from the store, and decode them into a numpy array. This
        function is blocking (no threading or async).
        """
        # TODO: kwarg to return None when the chunk does not exist

        # Check index
        if not isinstance(index, tuple):
            raise TypeError(
                f"ZarrArray.get_chunk_now() needs a tuple index, got {index!r}"
            )
        if len(index) != len(self._shape):
            raise IndexError(
                f"ZarrArray.get_chunk_now() needs {len(self._shape)} indices."
            )
        if not all(isinstance(i, int) for i in index):
            raise ValueError("ZarrArray.get_chunk_now() needs integer indices.")

        # Load data. This could take a while if it's a remote/slow store
        path = "c/" + self._chunk_separator.join(f"{x}" for x in index)
        if self._path:
            path = self._path + "/" + path
        try:
            encoded_bytes = self._store.get(path)
        except IOError:
            return np.full(self._chunk_shape, self._fill_value, self._dtype)

        # Return decoded
        array_type = create_ndarray_type(self._chunk_shape, self._dtype)
        return decode_bytes(memoryview(encoded_bytes), self._codecs, array_type)

    def get_chunk_soon(self, index) -> Future[np.ndarray]:
        """Read a chunk and return a ``concurrent.futures.Future``.

        Calls ``get_chunk_now()`` in a separate thread (using a ``ThreadPoolExecutor``).
        One can wait for the result, and also combine multiple reads in parallel.

        This has little to do with async programming and asyncio, although the future-object
        can be converted to an awaitable using ``asyncio.wrap_future()``.

        Example to wait for the data::

            f = zarr_array.get_chunk_soon(...)
            data = f.result()

        Combine multiple reads in parallel::

            f1 = zarr_array.get_chunk_soon(...)
            f2 = zarr_array.get_chunk_soon(...)
            f3 = zarr_array.get_chunk_soon(...)

            data1, data2, data3 = [f.result() for f in [f1, f2, f3]]

        Asynchronously await the data::

            f = zarr_array.get_chunk_soon(...)
            data = await asyncio.wrap_future(f)

        Async and parallel reads::

            f1 = zarr_array.get_chunk_soon(...)
            f2 = zarr_array.get_chunk_soon(...)
            f3 = zarr_array.get_chunk_soon(...)

            asyncio_futures = [asyncio.wrap_future(f) for f in [f1, f2, f3]]
            data1, data2, data3 = await asyncio.gather(*asyncio_futures)

        """
        return executor.submit(self.get_chunk_now, index)

    def set_chunk_now(self, index, data, check_empty=True) -> None:
        """Write a chunk to the store.

        Converts the index to the path for that chunk. Encodes the array
        to bytes, and save these to the store. This function is blocking
        (no threading or async).
        """

        # Check index
        if not isinstance(index, tuple):
            raise TypeError(
                f"ZarrArray.set_chunk_now() needs a tuple index, got {index!r}"
            )
        if len(index) != len(self._shape):
            raise IndexError(
                f"ZarrArray.set_chunk_now() needs {len(self._shape)} indices."
            )
        if not all(isinstance(i, int) for i in index):
            raise ValueError("ZarrArray.set_chunk_now() needs integer indices.")

        # Check data
        if not isinstance(data, np.ndarray):
            raise TypeError("A chunk should be a numpy array")
        if not (data.shape == self._chunk_shape and data.dtype == self._dtype):
            raise ValueError(
                f"Chunk must have shape {self._chunk_shape!r} and dtype {self._dtype!r}, but got {data.shape!r} and {data.dtype!r}"
            )

        # Write (or erase) the chunk
        path = "c/" + self._chunk_separator.join(f"{x}" for x in index)
        if self._path:
            path = self._path + "/" + path
        if check_empty and np.all(data == self._fill_value):
            try:
                self._store.erase(path)
            except IOError:
                pass
        else:
            encoded_bytes = encode_array(data, self._codecs)
            self._store.set(path, encoded_bytes)

    def set_chunk_soon(self, index, data) -> Future[None]:
        """Write a chunk and return a ``concurrent.futures.Future``.

        Calls ``set_chunk_now()`` in a separate thread (using a ``ThreadPoolExecutor``).
        One can wait for the result, and also combine multiple writes in parallel.

        This has little to do with async programming and asyncio, although the future-object
        can be converted to an awaitable using ``asyncio.wrap_future()``.

        Example to write and forget::

            f = zarr_array.set_chunk_soon(...)

        Combine multiple writes in parallel, and wait for them to finish::

            f1 = zarr_array.set_chunk_soon(...)
            f2 = zarr_array.set_chunk_soon(...)
            f3 = zarr_array.set_chunk_soon(...)

            [f.result() for f in [f1, f2, f3]]

        Asynchronously await the data::

            f = zarr_array.set_chunk_soon(...)
            await asyncio.wrap_future(f)

        Async and parallel reads::

            f1 = zarr_array.set_chunk_soon(...)
            f2 = zarr_array.set_chunk_soon(...)
            f3 = zarr_array.set_chunk_soon(...)

            asyncio_futures = [asyncio.wrap_future(f) for f in [f1, f2, f3]]
            await asyncio.gather(*asyncio_futures)

        """
        return executor.submit(self.set_chunk_now, index, data)

    def _parse_metadata(self):
        meta = self._metadata

        # Parse mandatory fields

        assert meta["node_type"] == "array"

        self._shape = tuple(int(i) for i in meta["shape"])
        self._dtype = dtype = meta["data_type"]

        i = len(dtype)
        while i > 0 and dtype[i - 1].isdigit():
            i -= 1
        self._dtype_bits = int(dtype[i:]) if i < len(dtype) else 8

        self._chunk_grid = meta["chunk_grid"]
        assert self._chunk_grid["name"] == "regular"
        self._chunk_shape = tuple(self._chunk_grid["configuration"]["chunk_shape"])

        self._chunk_grid_shape = tuple(
            math.ceil(array_s / chunk_s)
            for array_s, chunk_s in zip(self._shape, self._chunk_shape, strict=True)
        )

        self._chunk_key_encoding = meta["chunk_key_encoding"]
        assert self._chunk_key_encoding["name"] == "default"
        self._chunk_separator = self._chunk_key_encoding["configuration"]["separator"]

        self._fill_value = meta["fill_value"]
        self._codecs = meta["codecs"]
        assert len(self._codecs) >= 1
        assert self._codecs[0]["name"] == "bytes"

        # Parse optional fields

        self._attributes = meta.get("attributes", None)
        self._storage_transformers = meta.get("storage_transformers", None)
        self._dimension_names = meta.get("dimension_names", None)

    def _init_node(self):
        pass
