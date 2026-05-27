from math import ceil
from dataclasses import dataclass
from concurrent.futures import Future, wait as concurrent_gather
from typing import TYPE_CHECKING

import numpy as np

from .misc import executor


if TYPE_CHECKING:
    from .nodes import ZarrArray


@dataclass(slots=True)
class AxisInfo:
    """Represents a partial chunk; a chunk which is sliced."""

    chunk_index: int  #: the index of the chunk in the chunk grid
    chunk_sub1: int  #: the start index of the slice from the chunk
    chunk_sub2: int  #: the end index of the slice from the chunk
    array_sub1: int  #: the start index in the array
    array_sub2: int  #: the end index in the array
    step: int | None  #: None means collapsed dim


# class ChunkIndexInfo:
#     axis_info: tuple[AxisInfo]
#     future: object


class BaseIndexer:
    def __init__(self, zarr_array, *, return_future):
        self._array = zarr_array
        self._return_future = bool(return_future)
        self._shape = zarr_array._shape
        self._grid_shape = zarr_array._chunk_grid_shape

    def _normalize_selection(self, selection, shape):
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
        has_slices = False
        for axis in range(ndim):
            index = selection[axis]
            if isinstance(index, int):
                pass
            elif isinstance(index, slice):
                if index.start is None:
                    index = slice(0, index.stop, index.step)
                elif index.start < 0:
                    index = slice(shape[axis] + index.start, index.stop, index.step)
                if index.stop is None:
                    index = slice(index.start, shape[axis], index.step)
                elif index.stop < 0:
                    index = slice(index.start, shape[axis] + index.stop, index.step)
                if index.step is None:
                    index = slice(index.start, index.stop, 1)
                selection[axis] = index
            else:
                raise IndexError(
                    "ZarrArray chunk indexing needs the index of each dim to be int or slice"
                )

        return tuple(selection), has_slices

    def _get_chunk_ids_from_chunk_slice(self, selection):
        grid_shape = self._grid_shape
        selection, has_slices = self._normalize_selection(selection, grid_shape)

        if not has_slices:
            chunk_ids = [selection]

        else:

            def resolve(axis, partial_index, partial_selection):
                index, remaining_selection = partial_selection[0], partial_selection[1:]
                if isinstance(index, int):
                    if index < 0:
                        index = grid_shape[axis] + index
                    new_index = (*partial_index, index)
                    if remaining_selection:
                        resolve(axis + 1, new_index, remaining_selection)
                    else:
                        chunk_ids.append(new_index)
                elif isinstance(index, slice):
                    for i in range(*index.indices(grid_shape[axis])):
                        new_index = (*partial_index, i)
                        if remaining_selection:
                            resolve(axis + 1, new_index, remaining_selection)
                        else:
                            chunk_ids.append(new_index)

            chunk_ids = []
            resolve(0, (), selection)

        return chunk_ids

    def _get_chunk_index_info_from_zarr_array_slice(self, normalized_selection):
        shape = self._shape
        chunk_shape = self._array._chunk_shape

        # We produce a list of chunk_sub_indexes
        # Every chunk_sub_index is: (axis_info, ...)
        # Every axis_ino is: (chunk_index_for_axis, i_start, i_end)

        # TODO: refactor into a for-loop, to avoid recalculating the same info multiple times

        def resolve(axis, axis_info_tuple, partial_selection):
            # Get index for this axis. This index is for the zarr-array, and needs to be mapped to the chunk grid.
            index, remaining_selection = partial_selection[0], partial_selection[1:]
            if isinstance(index, int):
                if index < 0:
                    index = shape[axis] + index
                chunk_index = index // chunk_shape[axis]
                chunk_sub1 = index - chunk_index * chunk_shape[axis]
                chunk_sub2 = chunk_sub1 + 1
                array_sub1 = 0
                array_sub2 = 1
                axis_info = AxisInfo(
                    chunk_index,
                    chunk_sub1,
                    chunk_sub2,
                    array_sub1,
                    array_sub2,
                    step=None,
                )
                new_axis_info_tuple = (*axis_info_tuple, axis_info)
                if remaining_selection:
                    resolve(axis + 1, new_axis_info_tuple, remaining_selection)
                else:
                    chunk_index_infos.append(new_axis_info_tuple)
            elif isinstance(index, slice):
                # Prep calculations
                first_chunk_index = index.start // chunk_shape[axis]
                last_chunk_index = (index.stop - 1) // chunk_shape[axis]
                array_offset = 0
                # Iterate over (possible) chunks
                for chunk_index in range(first_chunk_index, last_chunk_index + 1):
                    # Establish chunk_sub1 and chunk_sub2
                    zarray_index_for_chunk = chunk_index * chunk_shape[axis]
                    chunk_sub1 = 0
                    chunk_sub2 = chunk_shape[axis]
                    if chunk_index == first_chunk_index:
                        chunk_sub1 = index.start - zarray_index_for_chunk
                    elif index.step > 1:
                        zarray_offset = zarray_index_for_chunk - index.start
                        zarray_offset = ceil(zarray_offset / index.step) * index.step
                        zarrray_index = index.start + zarray_offset
                        chunk_sub1 = zarrray_index - zarray_index_for_chunk
                    if chunk_index == last_chunk_index:
                        chunk_sub2 = index.stop - zarray_index_for_chunk
                    if chunk_sub1 >= chunk_sub2:
                        continue  # this chunk is skipped due to step size being larger than chunk size
                    # Calculate indices for the target array
                    array_sub1 = array_offset
                    array_sub2 = array_offset + ceil(
                        (chunk_sub2 - chunk_sub1) / index.step
                    )
                    array_offset = array_sub2
                    axis_info = AxisInfo(
                        chunk_index,
                        chunk_sub1,
                        chunk_sub2,
                        array_sub1,
                        array_sub2,
                        step=index.step,
                    )
                    new_axis_info_tuple = (*axis_info_tuple, axis_info)
                    if remaining_selection:
                        resolve(axis + 1, new_axis_info_tuple, remaining_selection)
                    else:
                        chunk_index_infos.append(new_axis_info_tuple)

        chunk_index_infos = []
        resolve(0, (), normalized_selection)

        return chunk_index_infos


class BaseChunkIndexer(BaseIndexer):
    pass


class IndexConverter(BaseIndexer):
    def __getitem__(self, selection):
        shape = self._shape
        normalized_selection, _has_slices = self._normalize_selection(selection, shape)

        # chunk_index_infos = self._get_chunk_index_info_from_zarr_array_slice(
        #     normalized_selection
        # )
        return normalized_selection


class DataLoader(BaseIndexer):
    def __getitem__(self, selection):
        shape = self._shape

        normalized_selection, _has_slices = self._normalize_selection(selection, shape)

        chunk_index_infos = self._get_chunk_index_info_from_zarr_array_slice(
            normalized_selection
        )

        return ZarrSubArray(self._array, normalized_selection, chunk_index_infos)


class ZarrSubArray:
    def __init__(self, array, normalized_selection, chunk_index_infos):
        self._array = array
        self._chunk_index_infos = chunk_index_infos

        # Determine sub array shape
        shape1 = []
        shape2 = []
        shape3 = []
        for index in normalized_selection:
            if isinstance(index, int):
                shape2.append(1)
                shape3.append(0)
            else:  # isinstance(index, slice)
                i = ceil((index.stop - index.start) / index.step)
                shape1.append(i)
                shape2.append(i)
                shape3.append(i)
        self._shape1 = tuple(shape1)  # shape with collapsed dims
        self._shape2 = tuple(shape2)  # uncollapsed shape
        self._shape3 = tuple(shape3)  # same but zero for collapsed axis

    @property
    def array(self) -> ZarrArray:
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the sub array."""
        return self._shape1

    def get(self):
        chunk_index_infos = self._chunk_index_infos

        # Create array
        array1 = np.empty(self._shape1, self._array.dtype)
        array2 = array1.reshape(self._shape2)

        chunk_index_infos = self._chunk_index_infos
        aggregate_future = Future()
        aggregator = Aggregator(aggregate_future, array1)

        for chunk_index_info in chunk_index_infos:
            chunk_index = tuple(i.chunk_index for i in chunk_index_info)
            aggregator.add(chunk_index)

        for chunk_index_info in chunk_index_infos:
            chunk_index = tuple(i.chunk_index for i in chunk_index_info)
            array_slices = tuple(
                slice(i.array_sub1, i.array_sub2) for i in chunk_index_info
            )
            chunk_slices = tuple(
                slice(i.chunk_sub1, i.chunk_sub2, i.step) for i in chunk_index_info
            )
            _future = executor.submit(
                read_chunk,
                self._array,
                aggregator,
                array2,
                array_slices,
                chunk_index,
                chunk_slices,
            )

        return aggregate_future

    def get_wait(self):
        return self.get().result()

    def set(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"{self.__class__.__name__}.set() accepts only a numpy array."
            )
        if value.shape == self._shape1:
            pass  # ok
        elif value.shape == self._shape2:
            value = value.reshape(self._shape1)
        else:
            raise ValueError(
                f"{self.__class__.__name__}.set() array has shape {value.shape} but expected {self._shape1} or {self._shape2}."
            )

        chunk_index_infos = self._chunk_index_infos
        aggregate_future = Future()
        aggregator = Aggregator(aggregate_future, None)

        for chunk_index_info in chunk_index_infos:
            chunk_index = tuple(i.chunk_index for i in chunk_index_info)
            aggregator.add(chunk_index)

        for chunk_index_info in chunk_index_infos:
            chunk_index = tuple(i.chunk_index for i in chunk_index_info)
            array_slices = tuple(
                slice(i.array_sub1, i.array_sub2) for i in chunk_index_info
            )
            chunk_slices = tuple(
                slice(i.chunk_sub1, i.chunk_sub2, i.step) for i in chunk_index_info
            )
            _future = executor.submit(
                write_chunk,
                self._array,
                aggregator,
                value,
                array_slices,
                chunk_index,
                chunk_slices,
            )

        return aggregate_future

    def set_wait(self, value):
        return self.set(value).result()


class Aggregator:
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
    try:
        data = zarr_array.get_chunk(chunk_index)
        array[*array_slices] = data[*chunk_slices]
    except Exception as err:
        aggregator.set_exception(err)
    else:
        aggregator.finish(chunk_index)


def write_chunk(zarr_array, aggregator, array, array_slices, chunk_index, chunk_slices):
    try:
        is_full_chunk = (
            all(s.start == 0 for s in chunk_slices)
            and (s.stop for s in chunk_slices) == zarr_array._chunk_shape
        )
        sub_data = array[*array_slices].astype(zarr_array._dtype, copy=False)
        if is_full_chunk:
            data = sub_data
        else:
            data = zarr_array.get_chunk(chunk_index)
            data[*chunk_slices] = sub_data
        zarr_array.set_chunk(chunk_index, data, check_empty=True)
    except Exception as err:
        aggregator.set_exception(err)
    else:
        aggregator.finish(chunk_index)


class ChunkLoader(BaseChunkIndexer):
    def __getitem__(self, selection):
        chunk_ids = self._get_chunk_ids_from_chunk_slice(selection)
        futures = []
        for chunk_id in chunk_ids:
            futures.append(self._array.get_chunk_future(*chunk_id))
        return concurrent_gather(*futures)
