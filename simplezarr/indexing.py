"""
Implementation for Zarr array indexing.
"""

from __future__ import annotations  # Using class names for types without Ruff F821

from math import ceil
from dataclasses import dataclass
from concurrent.futures import Future
from typing import TYPE_CHECKING

import numpy as np

from .misc import executor, ZarrFuture

if TYPE_CHECKING:
    from .nodes import ZarrArray


class ChunkGridIndexer:
    """Helper class to select a slice in the array using the chunk grid.

    Usage::

        # Select the 5th chunk vertically, and the 4th up to the 8th chunk horizontally.
        sub = zarr_array.chunks[5, 4:8]

        # Now get or set to get the numpy array
        a = sub.get_now()

        # Short form
        a = zarr_array.chunks[5, 4:8].get_now()
    """

    def __init__(self, array: ZarrArray):
        self._array = array

    def __getitem__(self, selection) -> ZarrSubArray:
        chunk_selection = normalize_selection(selection, self._array.chunk_grid_shape)
        ndim = self._array.ndim
        chunk_shape = self._array.chunk_shape

        array_selection = []
        for axis in range(ndim):
            chunk_index = chunk_selection[axis]
            if isinstance(chunk_index, int):
                array_index = slice(
                    chunk_index * chunk_shape[axis],
                    (chunk_index + 1) * chunk_shape[axis],
                    1,
                )
            else:
                if chunk_index.step > 1:
                    raise IndexError(
                        "When indexing in the chunk grid, slices with steps are not allowed. Use multiple calls to zarr_array.get_chunk_soon() instead."
                    )
                array_index = slice(
                    chunk_index.start * chunk_shape[axis],
                    chunk_index.stop * chunk_shape[axis],
                    1,
                )
            array_selection.append(array_index)

        return ZarrSubArray(self._array, tuple(array_selection))


class ZarrSubArray:
    """A Zarr sub-array that can be used to get and set data."""

    def __init__(self, array: ZarrArray, selection: tuple):
        self._array = array
        shape = array.shape

        normalized_selection = normalize_selection(selection, shape)
        self._index_repr = get_selection_repr(normalized_selection, shape)
        self._chunk_index_infos = get_chunk_index_info_from_zarr_array_slice(
            normalized_selection, shape, array.chunk_shape
        )

        # Determine sub array shape
        shape1 = []
        shape2 = []
        for index in normalized_selection:
            if isinstance(index, int):
                shape2.append(1)
            else:  # isinstance(index, slice)
                i = ceil((index.stop - index.start) / index.step)
                shape1.append(i)
                shape2.append(i)
        self._shape1 = tuple(shape1)  # shape with collapsed dims
        self._shape2 = tuple(shape2)  # uncollapsed shape

    def __repr__(self):
        return f"<{self.__class__.__name__} a[{self._index_repr}] at {hex(id(self))}>"

    @property
    def array(self) -> ZarrArray:
        """The ZarrArray that this is a slice of."""
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the sub array. Some dimensions can be collapsed, the array can even represent a scalar."""
        return self._shape1

    def get_soon(self) -> Future[np.ndarray]:
        """Get the data for this sub-array as a numpy array.

        Returns a Future so the caller can wait for it in an appropriate way (e.g. wait for multiple gets in parallel).
        """
        chunk_index_infos = self._chunk_index_infos

        array1 = np.empty(self._shape1, self._array.dtype)
        array2 = array1.reshape(self._shape2)

        aggregate_future = ZarrFuture(f"a[{self._index_repr}].get_soon()")
        aggregator = Aggregator(aggregate_future, array1)

        for chunk_index_info in chunk_index_infos:
            aggregator.add(chunk_index_info.chunk_index)

        for chunk_index_info in chunk_index_infos:
            _future = executor.submit(
                read_chunk,
                self._array,
                aggregator,
                array2,
                chunk_index_info.array_slices,
                chunk_index_info.chunk_index,
                chunk_index_info.chunk_slices,
            )

        return aggregate_future

    def get_now(self) -> np.ndarray:
        """Get the data for this sub-array as a numpy array.

        Blocks while waiting for the data to arrive. If the requested data
        consists of multiple chunks, these chunks are loaded in parallel.
        """
        return self.get_soon().result()

    def set_soon(self, value: float | np.ndarray) -> Future:
        """Set the data for this sub-array using a numpy array.

        Returns a Future so the caller can wait in an appropriate for the write to finish.
        You could "fire and forget", but then you don't see any errors when the write fails.
        If you are in an async framework, you can async wait for it so you do see the error.
        """
        if not isinstance(value, (int, float, np.ndarray)):
            raise TypeError(
                f"{self.__class__.__name__}.set_soon() accepts only a numpy array."
            )
        if isinstance(value, np.ndarray):
            if value.shape == self._shape1:
                value = value.reshape(self._shape2)
            elif value.shape == self._shape2:
                pass  # ok
            else:
                raise IndexError(
                    f"{self.__class__.__name__}.set_soon() array has shape {value.shape} but expected {self._shape1} or {self._shape2}."
                )
        chunk_index_infos = self._chunk_index_infos
        aggregate_future = ZarrFuture(f"a[{self._index_repr}].set_soon()")
        aggregator = Aggregator(aggregate_future, None)

        for chunk_index_info in chunk_index_infos:
            aggregator.add(chunk_index_info.chunk_index)

        for chunk_index_info in chunk_index_infos:
            _future = executor.submit(
                write_chunk,
                self._array,
                aggregator,
                value,
                chunk_index_info.array_slices,
                chunk_index_info.chunk_index,
                chunk_index_info.chunk_slices,
            )

        return aggregate_future

    def set_now(self, value: np.ndarray) -> None:
        """Set the data for this sub-array using a numpy array.

        Blocks while waiting for the write to finish. If the written data covers
        multiple chunks, these chunks are written in parallel.
        """
        return self.set_soon(value).result()


class Aggregator:
    """Helper to detect the finishing of multiple futures to finish the aggregate future."""

    def __init__(self, future: Future, result: object):
        self._future = future
        self._result = result
        self._chunk_indices = set()

    def add(self, chunk_index: tuple[int, ...]):
        self._chunk_indices.add(chunk_index)

    def finish(self, chunk_index: tuple[int, ...]):
        self._chunk_indices.discard(chunk_index)
        if not self._chunk_indices and self._future is not None:
            self._future.set_result(self._result)
            self._future = None
            self._result = None

    def set_exception(self, err: Exception):
        self._future.set_exception(err)
        self._future = None
        self._result = None


def read_chunk(zarr_array, aggregator, array, array_slices, chunk_index, chunk_slices):
    """Function to run in the exectutor to read a chunk."""
    try:
        data = zarr_array.get_chunk_now(chunk_index)
        array[*array_slices] = data[*chunk_slices]
    except Exception as err:  # no-cover
        aggregator.set_exception(err)
    else:
        aggregator.finish(chunk_index)


def write_chunk(
    zarr_array, aggregator, array_or_scalar, array_slices, chunk_index, chunk_slices
):
    """Function to run in the executor to write a chunk."""
    try:
        is_full_chunk = (
            all(s.start == 0 for s in chunk_slices)
            and tuple(s.stop for s in chunk_slices) == zarr_array.chunk_shape
        )
        if isinstance(array_or_scalar, (float, int)):
            sub_data = array_or_scalar
            is_full_chunk = False
        else:
            sub_data = array_or_scalar[*array_slices].astype(
                zarr_array.dtype, copy=False
            )
        if is_full_chunk:
            data = sub_data
        else:
            data = zarr_array.get_chunk_now(chunk_index).copy()
            data[*chunk_slices] = sub_data
        zarr_array.set_chunk_now(chunk_index, data, check_empty=True)
    except Exception as err:
        aggregator.set_exception(err)
    else:
        aggregator.finish(chunk_index)


@dataclass(slots=True)
class ChunkIndexInfo:
    """Represents a partial chunk; a chunk which is sliced, with corresponding indices in a target array."""

    chunk_index: tuple[int, ...]  #: the index of the chunk in the chunk grid
    chunk_slices: tuple[slice, ...]  #: the n-dimensional slice to address in the chunk
    array_slices: tuple[slice, ...]  #: the n-dimensional slice to address in the array


def normalize_selection(selection: tuple, shape: tuple[int, ...]) -> tuple:
    """Check types and dimensions, resolve ellipsis and slices."""

    ndim = len(shape)

    # Make selection a list (mutable)
    if not isinstance(selection, tuple):
        selection = [selection]
    else:
        selection = list(selection)

    # Resolve Ellipsis
    has_ellipsis = selection.count(Ellipsis)
    if has_ellipsis:
        if has_ellipsis > 1:
            raise IndexError(
                "Only one Ellipsis (...) allowed in indexing a Zarr array."
            )
        pos = selection.index(Ellipsis)
        extra = [slice(None)] * (ndim - len(selection) + 1)
        selection = [*selection[:pos], *extra, *selection[pos + 1 :]]

    # Check selection
    if len(selection) != ndim:
        raise IndexError(
            f"ZarrArray chunk indexing needs {ndim} indices, use ellipsis if necessary."
        )

    # More checks and resolve slices for None and negative values
    for axis in range(ndim):
        index = selection[axis]
        if isinstance(index, int):
            if index < 0:
                index = shape[axis] + index
                selection[axis] = index
        elif isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step
            if start is None:
                start = 0
            if stop is None:
                stop = shape[axis]
            if step is None:
                step = 1
            if not (
                isinstance(start, int)
                and isinstance(stop, int)
                and isinstance(step, int)
            ):
                raise IndexError("Index slice must consist only of ints.")
            if start < 0:
                start = max(0, shape[axis] + start)
            else:
                start = min(shape[axis], start)
            if stop < 0:
                stop = shape[axis] + stop
            else:
                stop = min(shape[axis], stop)
            stop = max(start, stop)
            if step <= 0:
                raise IndexError(f"Index slice step must one or higher, got {step}")
            selection[axis] = slice(start, stop, step)
        else:
            raise IndexError(
                "ZarrArray chunk indexing needs the index of each dim to be int or slice"
            )

    # Bounds check
    for axis in range(ndim):
        index = selection[axis]
        if isinstance(index, int):
            if index < 0 or index > shape[axis]:
                raise IndexError(
                    f"Index out of bounds: {get_selection_repr(selection, shape)}"
                )

    return tuple(selection)


def get_selection_repr(normalized_selection: tuple, shape: tuple[int, ...]) -> str:
    index_repr = []
    for axis in range(len(normalized_selection)):
        index = normalized_selection[axis]
        if isinstance(index, int):
            index_repr.append(str(index))
        elif isinstance(index, slice):
            x = f"{index.start if index.start > 0 else ''}:"
            if index.stop < shape[axis]:
                x += f"{index.stop}"
            if index.step > 1:
                x += f":{index.step}"
            index_repr.append(x)
    return ",".join(index_repr)


def get_chunk_index_info_from_zarr_array_slice(
    normalized_selection: tuple, shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> list[ChunkIndexInfo]:
    """Get per-chunk indexing info, based on array slices."""

    ndim = len(shape)
    assert ndim == len(chunk_shape)

    chunk_index_info_list_per_axis = {i: [] for i in range(ndim)}
    chunk_index_info_list_per_axis[-1] = [ChunkIndexInfo((), (), ())]

    def add_chunk_index_info(axis, chunk_int, chunk_slice, array_slice):
        old_chunk_index_info_list = chunk_index_info_list_per_axis[axis - 1]
        new_chunk_index_info_list = chunk_index_info_list_per_axis[axis]
        for old_chunk_index_info in old_chunk_index_info_list:
            new_chunk_index_info = ChunkIndexInfo(
                chunk_index=(*old_chunk_index_info.chunk_index, chunk_int),
                chunk_slices=(*old_chunk_index_info.chunk_slices, chunk_slice),
                array_slices=(*old_chunk_index_info.array_slices, array_slice),
            )
            new_chunk_index_info_list.append(new_chunk_index_info)

    for axis in range(ndim):
        index = normalized_selection[axis]
        if isinstance(index, int):
            chunk_int = index // chunk_shape[axis]
            chunk_sub = index - chunk_int * chunk_shape[axis]
            add_chunk_index_info(
                axis, chunk_int, slice(chunk_sub, chunk_sub + 1, None), slice(0, 1)
            )
        elif isinstance(index, slice):
            # Prep calculations
            first_chunk_int = index.start // chunk_shape[axis]
            last_chunk_int = (index.stop - 1) // chunk_shape[axis]
            array_offset = 0
            # Iterate over (possible) chunks
            for chunk_int in range(first_chunk_int, last_chunk_int + 1):
                # Establish chunk_sub1 and chunk_sub2
                zarray_index_for_chunk = chunk_int * chunk_shape[axis]
                chunk_sub1 = 0
                chunk_sub2 = chunk_shape[axis]
                if chunk_int == first_chunk_int:
                    chunk_sub1 = index.start - zarray_index_for_chunk
                elif index.step > 1:
                    zarray_offset = zarray_index_for_chunk - index.start
                    zarray_offset = ceil(zarray_offset / index.step) * index.step
                    zarrray_index = index.start + zarray_offset
                    chunk_sub1 = zarrray_index - zarray_index_for_chunk
                if chunk_int == last_chunk_int:
                    chunk_sub2 = index.stop - zarray_index_for_chunk
                if chunk_sub1 >= chunk_sub2:
                    continue  # this chunk is skipped due to step size being larger than chunk size
                # Calculate indices for the target array
                array_sub1 = array_offset
                array_sub2 = array_offset + ceil((chunk_sub2 - chunk_sub1) / index.step)
                array_offset = array_sub2
                add_chunk_index_info(
                    axis,
                    chunk_int,
                    slice(chunk_sub1, chunk_sub2, index.step),
                    slice(array_sub1, array_sub2),
                )

    chunk_index_info_list = chunk_index_info_list_per_axis[ndim - 1]
    return chunk_index_info_list
