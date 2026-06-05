from __future__ import annotations  # Using class names for types without Ruff F821

import json
import math
from concurrent.futures import Future

import numpy as np

from .misc import executor, DTYPES, resolve_fill_value
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
    """Open a zarr file using the given store.

    Zarr files are made up of a tree of nodes. Each node is either a
    ``ZarrGroup`` or a ``ZarrArray``. The arrays are the leaf nodes. When
    opening a Zarr file, it may be a single array, or a group containing
    multiple arrays.
    """
    return ZarrNode._from_path(store, "")


def join(*path_parts):
    return "/".join(path_parts).lstrip("/")


class ZarrNode:
    """The base class for ``ZarrGroup`` and ``ZarrArray``.

    A Zarr file is made up of nodes, where arrays are the leaf nodes.
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
        self._attributes = {}
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
        """The store where the data related to this node is stored."""
        return self._store

    @property
    def name(self) -> str:
        """The name of this node, i.e. the part of the path after the last slash."""
        return self._name

    @property
    def path(self) -> str:
        """The full path of this node in the store."""
        return self._path

    @property
    def metadata(self) -> dict:
        """The full metadata for this node, as a Python dictionary."""
        return self._metadata

    @property
    def attributes(self) -> dict:
        """The attributes of this node. I.e. ``metadata["attributes"]``"""
        return self._attributes

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
    the Zarr file by indexing::

        sub_node = zarr_group['path/to/node']

    """

    @classmethod
    def create(cls, store: WritableStore, path: str, *, attributes: dict | None = None):
        """Create a new ZarrGroup in the given store.

        Arguments:
            store : WritableStore
                The store to write the group to.
            path : str
                The path of the zarr-group in the store.
            attributes : dict
                Additional metadata.
        """
        # Checks
        if not isinstance(path, str):  # no-cover
            raise TypeError(f"ZarrGroup path must be str, got {path!r}")
        if not (attributes is None or isinstance(attributes, dict)):  # no-cover
            raise TypeError(
                f"ZarrGroup attributes must be None or dict, got {attributes!r}"
            )

        # Build metadata
        metadata = {
            "zarr_format": 3,
            "node_type": "group",
        }
        if attributes is not None:
            metadata["attributes"] = attributes

        # Write
        json_text = json.dumps(metadata, indent=4)
        store.set(join(path, "zarr.json"), json_text.encode())

        return cls(store, path, metadata)

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

    def __repr__(self):
        return self.get_structure(max_depth=1)

    def _one_line_repr(self):
        return f"<{self.__class__.__name__} '{self._path}' with {len(self._children)} children at {hex(id(self))}>"

    @property
    def children(self) -> tuple[ZarrNode]:
        """The child nodes of this group. These can be groups or arrays."""
        return tuple(self._children.values())

    def print_structure(self, max_depth: int = 999):
        """Print the structure of the Zarr file from this group and below."""
        print(self.get_structure(max_depth=max_depth))

    def get_structure(self, max_depth: int = 999, indent: int = 0) -> str:
        """Get the structure of this group as a human-readable string."""
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


class ZarrArray(ZarrNode):
    """The class that represents a Zarr array.

    These arrays don't contain any bytes themselves, but are used as proxies
    that provide metadata about the array. Indexing into a ``ZarrArray`` gives a
    ``ZarrArraySlice`` which can be used to get and set the data as numpy
    arrays.
    """

    @classmethod
    def create(
        cls,
        store: WritableStore,
        path: str,
        shape: tuple[int, ...],
        dtype: str,
        *,
        fill_value: object = None,
        chunk_shape: tuple[int, ...] | None = None,
        chunk_path_separator: str = "/",
        codecs: list[dict] | None = None,
        dimension_names: list[str] | None = None,
        attributes: dict | None = None,
    ):
        """Create a new ZarrArray in the given store.

        Arguments:
            store : WritableStore
                The store to write the array (json and chunks) to.
            path : str
                The path of the zarr-array in the store.
            shape : tuple[int, ...]
                The shape of the array.
            dtype : str
                The data-type of the array.
            fill_value
                The value to fill in when no chunk is present. The type must match the dtype.
                In most cases None can be provided to mean "zero".
            chunk_shape : tuple[int]
                The shape of the chunks, must match the dimensions of ``shape``.
                If not given or None, there is one chunk with the same shape as the array itself.
            chunk_path_separator : str
                The path-separator to use for chunks. Default '/'. Part of the metadata's ``chunk_key_encoding```.
            codecs : list[dict] | None
                A list of codecs to encode the array data. If None, uses ZSTD compression with level 7,
                using ``{"name": "bytes", "configuration": {"endian": "little"}}``
                and ``{"name": "zstd", "configuration": {"level": 7, "checksum": True}}``.
            dimension_names : tuple[str]
                The names of the dimensions. Optional. The number of names must match the legth of the ``shape``.
            attributes : dict
                Additional metadata.
        """

        # Check path
        if not isinstance(path, str):  # no-cover
            raise TypeError(f"ZarrArray path must be str, got {path!r}")

        # Check dtype
        if isinstance(dtype, type) and issubclass(dtype, np.number):
            dtype = dtype.__name__
        elif isinstance(dtype, np.dtype):
            dtype = dtype.name
        elif not isinstance(dtype, str):  # no-cover
            raise TypeError(f"ZarrArray dtype must be str, got {dtype!r}")
        if not (
            dtype in DTYPES or (dtype.startswith("r") and dtype[1:].isnumeric())
        ):  # no-cover
            # Currently ignoring possible dtypes of extensions
            raise ValueError(f"ZarrArray dtype must be one of {DTYPES}, got {dtype!r}")

        # Check shape
        ndim = len(shape)
        shape = tuple(int(i) for i in shape)
        if ndim < 1:  # no-cover
            raise ValueError(
                f"ZarrArray dimensions must be at least 1D, got shape {shape!r}"
            )
        if any(i <= 0 for i in shape):  # no-cover
            raise ValueError(
                f"ZarrArray dimensions cannot be zero or less, got shape {shape!r}"
            )

        # Check and resolve chunk_grid
        if chunk_shape is None:
            chunk_shape = shape
        if len(chunk_shape) != ndim:  # no-cover
            raise ValueError(
                f"ZarrArray chunk_shape does not match the shape ndim ({ndim}), got {chunk_shape!r}"
            )
        if any(i <= 0 for i in chunk_shape):  # no-cover
            raise ValueError(
                f"ZarrArray chunk_shape cannot have zero or less dimensions, got {chunk_shape!r}"
            )
        chunk_grid = {"name": "regular", "configuration": {"chunk_shape": chunk_shape}}

        # Check and resolve fill_value
        fill_value = resolve_fill_value(fill_value, dtype)[1]

        # Check and create chunk_key_encoding
        if chunk_path_separator is None:
            chunk_path_separator = "/"
        if not isinstance(chunk_path_separator, str):  # no-cover
            raise TypeError(
                f"ZarrArray chunk_path_separator must be str got {chunk_path_separator!r}"
            )
        chunk_key_encoding = {
            "name": "default",
            "configuration": {"separator": chunk_path_separator},
        }

        # Check and create codecs
        if codecs is None:
            codecs = [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 7, "checksum": True}},
            ]
        elif not (
            isinstance(codecs, (tuple, list))
            and len(codecs) >= 1
            and all(isinstance(d, dict) and "name" in d for d in codecs)
        ):
            raise ValueError(
                f"ZarrArray codecs must be a list of dicts with at least a field 'name', got {codecs!r}"
            )

        # Build metadata. The order of the keys matches their order in the spec.
        metadata = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": dtype,
            "chunk_grid": chunk_grid,
            "chunk_key_encoding": chunk_key_encoding,
            "fill_value": fill_value,
            "codecs": codecs,
        }

        if attributes is not None:
            if not isinstance(attributes, dict):  # no-cover
                raise TypeError(
                    f"ZarrGroup attributes must be None or dict, got {attributes!r}"
                )
            metadata["attributes"] = attributes

        storage_transformers = None  # Zarr spec 3.1 does not specify any
        if storage_transformers is not None:  # no-cover
            metadata["storage_transformers"] = storage_transformers

        if dimension_names is not None:
            dimension_names = tuple(str(s) for s in dimension_names)
            if len(dimension_names) != ndim:  # no-cover
                raise ValueError(
                    f"ZarrArray dimension_names must match ndim {ndim}, got {dimension_names!r}"
                )
            metadata["dimension_names"] = dimension_names

        # Write
        json_text = json.dumps(metadata, indent=4)
        store.set(join(path, "zarr.json"), json_text.encode())

        return cls(store, path, metadata)

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
        self._chunk_path_separator = self._chunk_key_encoding["configuration"][
            "separator"
        ]

        self._fill_value = resolve_fill_value(meta["fill_value"], self._dtype)[0]

        self._codecs = meta["codecs"]
        assert len(self._codecs) >= 1
        assert self._codecs[0]["name"] == "bytes"

        # Parse optional fields

        self._attributes = meta.get("attributes", None)
        self._storage_transformers = meta.get("storage_transformers", None)
        self._dimension_names = meta.get("dimension_names", None)

    def _init_node(self):
        pass

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
        """Select a contiguous set of chunks.

        Similar to ``__getitem__``, but the indices are coordinates in the chunk grid.

        Example::

            arr.chunks[0, 0]  # Select array for the first chunk
            arr.chunks[0, :]  # Select array for first row of chunks
        """
        return ChunkGridIndexer(self)

    def __getitem__(self, selection) -> ZarrArraySlice:
        """Select a slice from the array.

        The returned ``ZarrArraySlice`` can be used to get and set the actual data.

        Examples::

            # The lines below assume ndim=2 for sake of simplicity
            arr[...]  # select the whole array
            arr[0, :]  # select one row
            arr[:10, 100:800:5]  # Slice, optionally with steps
            arr[10, 10]  # select a scalar
        """
        return ZarrArraySlice(self, selection)

    def __setitem__(self, *args):  # co-cover
        raise IndexError(
            "ZarrArray does not support index assignment (``a[..] = foo``), instead use ``a[..].set_now(foo)`` or ``a[..].set_soon(foo)``."
        )

    def get_chunk_now(self, index, none_if_missing=False) -> None | np.ndarray:
        """Read a chunk from the store.

        This function is synchronous; you may want to use ``get_chunk_soon()``
        to do the loading and decompression in a separate thread.

        Converts the index to the path for that chunk, load the bytes
        from the store, and decode them into a numpy array. This
        function is blocking (no threading or async).

        If the chunk does not exist, an array populated with the fill-value is returned,
        unless ``none_if_missing`` is True, in which case None is returned.
        """

        # Check index
        if not isinstance(index, tuple):
            raise TypeError(
                f"ZarrArray.get_chunk_now() needs a tuple index, got {index!r}"
            )
        if len(index) != len(self._shape):
            raise IndexError(
                f"ZarrArray.get_chunk_now() needs {len(self._shape)} indices."
            )
        if not all(isinstance(i, int) and i >= 0 for i in index):
            raise IndexError(
                "ZarrArray.get_chunk_now() needs positive integer indices."
            )

        # Load data. This could take a while if it's a remote/slow store
        path = "c/" + self._chunk_path_separator.join(f"{x}" for x in index)
        if self._path:
            path = self._path + "/" + path
        try:
            encoded_bytes = self._store.get(path)
        except IOError:
            encoded_bytes = None

        # Return decoded
        if encoded_bytes is not None:
            array_type = create_ndarray_type(self._chunk_shape, self._dtype)
            return decode_bytes(memoryview(encoded_bytes), self._codecs, array_type)
        elif none_if_missing:
            return None
        else:
            return np.full(self._chunk_shape, self._fill_value, self._dtype)

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
        if not all(isinstance(i, int) and i >= 0 for i in index):
            raise IndexError(
                "ZarrArray.set_chunk_now() needs positive integer indices."
            )

        # Check data
        if not isinstance(data, np.ndarray):
            raise TypeError("A chunk should be a numpy array")
        if not (data.shape == self._chunk_shape and data.dtype == self._dtype):
            raise ValueError(
                f"Chunk must have shape {self._chunk_shape!r} and dtype {self._dtype!r}, but got {data.shape!r} and {data.dtype!r}"
            )

        # Write (or erase) the chunk
        path = "c/" + self._chunk_path_separator.join(f"{x}" for x in index)
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
