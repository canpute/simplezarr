"""
Support for writing multiscale images, a.k.a. ome-zarr pyramids.

Supports huge datasets by building the pyramid up in pieces. Has fast throughput by adopting a depth-first approach and makes use of parallel writes.

Currently it seems to be about 4-10 times faster than the ome-zarr Python library, depending on various factors like chunk size.

Examples:

.. code-block:: python

    from simplezarr.utils.write_multiscale import write_ome_zarr_pyramid

    g = write_ome_zarr_pyramid(
        store,
        "example.ome.zarr",
        numpy_array_or_zarr_array,
        "zyx",  # or "zy" or "czyx" etc.
        space_unit="micrometer",
        chunk_shape=(32, 32, 32),
    )
"""

import itertools
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

import simplezarr
from simplezarr.utils.npzarrarray import NumpyZarrArray
from simplezarr.utils.units import SPACE_UNITS, TIME_UNITS

_DEFAULT_MAX_MEMORY = 1024**3  # == 2**30 == 1GiB


@dataclass
class PyramidData:
    """Small dataclass that defines the data for one multiscale pyramid.

    Arguments:
        src_array : ZarrArray or np.ndarray
            The source array. An array of 2, 3, 4 or 5 dimensions (t, c, z, y, x).
            A numpy array is wrapped in a read-only zero-copy view (NumpyZarrArray).
        name : str
            The name for the multi-scale pyramid. Default empty string.
        dtype : str
            The data type of the new array. If not given, uses the src_array's dtype.
            When casting to an integer dtype, values are rounded and clipped to
            the dtype's range.
        fill_value
            The value to fill in when no chunk is present. If not given, uses the src_array's fill_value.
        codecs : list[dict] | None
            Same as ``ZarrArray.create()``. If not given, uses the src_array's codecs.
    """

    src_array: simplezarr.ZarrArray | np.ndarray
    name: str = ""
    dtype: str | None = None
    fill_value: object = None
    codecs: list[dict] | None = None

    def __post_init__(self):
        src_array = self.src_array

        if isinstance(src_array, np.ndarray):
            # Wrap it, so that no copy of the data is made
            self.src_array = src_array = NumpyZarrArray(src_array)
        if not isinstance(src_array, (simplezarr.ZarrArray, NumpyZarrArray)):
            raise TypeError(
                f"write_ome_zarr() Expected an array, got {src_array.__class__.__name__} {src_array}"
            )
        if not (src_array.ndim >= 2 and src_array.ndim <= 5):
            raise ValueError(
                f"write_ome_zarr() Expected an array with 2, 3, 4 or 5 dimensions, got shape {src_array.shape}"
            )

        if not isinstance(self.name, str):
            raise TypeError(f"write_ome_zarr() name must be str, got {self.name!r}")
        if self.dtype is None:
            self.dtype = src_array.dtype
        if self.fill_value is None:
            self.fill_value = src_array._fill_value
        if self.codecs is None:
            self.codecs = src_array._codecs


def write_ome_zarr_pyramid(
    store: simplezarr.BaseStore,
    path: str,
    axes: tuple[str],
    pyramids: list[PyramidData],
    *,
    nlevels: int = 0,
    space_unit: str = "",
    base_scale: float | tuple[float, ...] = 1.0,
    chunk_shape: int | tuple[int] = 64,
    chunk_path_separator: str = ".",
    pyramid_path_prefix: str = "pyramid",
    max_memory: int = _DEFAULT_MAX_MEMORY,
    steps: list | None = None,
):
    """
    Write an ome-zarr image pyramid to the given store.

    Arguments:
        store : simplezarr.BaseStore
            The simplezarr store object to store the Zarr pyramid in.
        path : str
            The path in the store to write the group that represents the pyramid.
        axes : str
            The names of the dimensions, e.g. "xyz" or "cyx", with "t" for time,
            "c" for channels, and "zyx" for the spatial dimensions. Must match the array's ndim.
        pyramids : PyramidData or list[PyramidData]
            The definition for each multiscale pyramid in the Zarr files. Historically it's common for OME-Zarr files
            to have a single pyramid, but the spec allows multiple pyramids, which can be convenient for better
            separation of e.g. codecs. If multiple pyramds are written, it's up to the reader to interpret
            what the different pyramids mean. In this implementation, the shape, scaling and chunk size are forced to be the same
            across multiple pyramids.
        nlevels : int
            The number of scale levels to create. If zero, determine automatically
            so that the highest level is <= chunk shape.
        space_unit : str
            The spatial unit, default empty string for not-set.
        base_scale : float or tuple[float]
            The scale of the original data (at level 0), default 1.0.
        chunk_shape : int or tuple[int]
            The chunk_shape for all levels in the new array. Default 64.
        chunk_path_separator : str
            The path-separator to use for chunks. Default '.'.
        pyramid_path_prefix : str
            The path prefix that is used when multiple pyramids are written. Default 'pyramid'.
        max_memory : int
            The approximate maximum amount of memory (in bytes) to use for intermediate
            data in the "depthfirst" algorithm. The default is 1GiB.
            Set to 0 to force using the simple algorithm.
        steps : list | None
            For diagnosis: if a list is given, a dict is appended to it (in-place)
            for each step that the algorithm takes, specifying the bottom level,
            the number of levels created in that step, and whether the intermediate
            top level was kept in memory.
    """

    # Check pyramids
    if isinstance(pyramids, PyramidData):
        pyramids = [pyramids]
    if not (
        isinstance(pyramids, list)
        and len(pyramids) >= 1
        and all(isinstance(p, PyramidData) for p in pyramids)
    ):
        raise TypeError(
            f"Pyramids must be a PyramidData or a nonempty list of PyramidData instance, got {pyramids.__class__}"
        )

    # Make sure the dimensionality matches
    ndim = pyramids[0].src_array.ndim
    array_shape = pyramids[0].src_array.shape
    for p in pyramids:
        if ndim != p.src_array.ndim:
            raise ValueError(
                "Expecting all source arrays to have the same dimensionality."
            )
        if array_shape != p.src_array.shape:
            raise ValueError("Expecting all source arrays to have the same shape.")

    if not isinstance(path, str):
        raise TypeError(f"write_ome_zarr() path must be str, got {path!r}")
    if not isinstance(axes, str):
        raise TypeError(f"write_ome_zarr() axes must be tr, got {axes!r}")
    if len(axes) != ndim:
        raise ValueError(
            f"write_ome_zarr() axes length must match array dimensions, got {axes!r}"
        )
    if axes not in ["yx", "zyx", "cyx", "czyx", "tyx", "tzyx", "tcyx", "tczyx"]:
        raise ValueError(f"write_ome_zarr() unexpected axes {axes!r}.")

    space_unit = str(space_unit)

    if isinstance(base_scale, float):
        base_scale = tuple(1.0 if a in "tc" else base_scale for a in axes)
    elif not (
        isinstance(base_scale, tuple)
        and all(isinstance(x, float) for x in base_scale)
        and len(base_scale) == ndim
    ):
        raise TypeError(
            f"write_ome_zarr() base_scale must be a tuple of {ndim} floats, got {base_scale!r}"
        )
    nlevels = max(0, int(nlevels))

    # Build axes list
    axes_list = []
    for a in axes:
        typ = {"t": "time", "c": "channel"}.get(a, "space")
        unit = {"t": "", "c": ""}.get(a, space_unit)
        d = {"name": a, "type": typ}
        if unit:
            d["unit"] = unit
            if typ == "space" and unit not in SPACE_UNITS:
                raise ValueError(
                    f"write_ome_zarr() unit {unit!r} is not a valid space unit."
                )
            if typ == "time" and unit not in TIME_UNITS:  # no-cover
                raise ValueError(
                    f"write_ome_zarr() unit {unit!r} is not a valid time unit."
                )
            if typ == "channel":  # no-cover
                raise ValueError(f"write_ome_zarr() unit {unit!r} for a channel??")
        axes_list.append(d)

    zyx_ndim = sum(1 for a in axes if a in "xyz")
    _tc_ndim = ndim - zyx_ndim

    # Chunk shape
    if isinstance(chunk_shape, int):
        zyx_chunk_shape = (chunk_shape,) * zyx_ndim
        chunk_shape = (*array_shape[:-zyx_ndim], *zyx_chunk_shape)
    elif isinstance(chunk_shape, tuple) and all(
        isinstance(x, int) for x in chunk_shape
    ):
        if len(chunk_shape) != ndim:
            raise ValueError("chunk_shape does not match ndim")
        zyx_chunk_shape = chunk_shape[-zyx_ndim:]
    else:
        raise TypeError(f"Invalid chunk_shape {chunk_shape}")

    # Path separator
    if not isinstance(chunk_path_separator, str):
        raise ValueError("chunk_path_separator must be str.")

    # Select downsample method
    method = "local_mean"  # our API enum, also ends up as 'type' in metadata
    if method not in {"local_mean"}:
        raise ValueError(f"write_ome_zarr() unexpected method {method!r}")
    downsample_metadata = {
        "description": f"Pyramid build with simplezarr.utils.pyramid using the {method!r} method"
    }

    # Build dataset list, we append to this list during processing below
    dataset_lists = [[] for _ in pyramids]
    pyramid_paths = []

    # Based on ome-zarr 0.5
    attributes = {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "name": pyramids[i].name,
                    "axes": axes_list,
                    "datasets": dataset_lists[i],
                    "type": method,
                    "metadata": downsample_metadata,
                }
                for i in range(len(pyramids))
            ],
        }
    }

    # Prepare arrays

    path_prefix = "level"

    src_zyx_shape = tuple(array_shape[i] for i in range(ndim) if axes[i] in "zyx")
    src_zyx_scale = tuple(base_scale[i] for i in range(ndim) if axes[i] in "zyx")

    tc_shape = tuple(array_shape[i] for i in range(ndim) if axes[i] in "tc")
    tc_scale = tuple(base_scale[i] for i in range(ndim) if axes[i] in "tc")
    tc_translation = tuple(0 for _ in tc_scale)

    for pyramid_index in range(len(pyramids)):
        new_arrays = []
        zyx_shape = src_zyx_shape
        zyx_scale = src_zyx_scale
        prev_zyx_shape = zyx_shape

        pyramid_path = None
        if len(pyramids) > 1:
            pyramid_path = f"{pyramid_path_prefix}{pyramid_index}"
            pyramid_paths.append(pyramid_path)

        # Note the implicit level cap, but it's huge, even for massive data
        for level in range(100):
            if level > 0:
                prev_zyx_shape = zyx_shape
                zyx_shape = tuple(s // 2 for s in zyx_shape)
                zyx_scale = tuple(s * 2 for s in zyx_scale)
            # Decide when to stop
            if nlevels == 0:
                # Stop when the last level fits in a single chunk,
                # or when we cannot halve the shape any further.
                if level >= 1 and (
                    all(
                        prev_zyx_shape[i] <= zyx_chunk_shape[i] for i in range(zyx_ndim)
                    )
                    or any(s == 0 for s in zyx_shape)
                ):
                    break
            elif level >= nlevels:
                break
            array_path = f"{path_prefix}{level}"
            if pyramid_path:
                array_path = pyramid_path + "/" + array_path
            new_array = simplezarr.ZarrArray.create(
                store,
                f"{path}/{array_path}",
                (*tc_shape, *zyx_shape),
                pyramids[pyramid_index].dtype,
                fill_value=pyramids[pyramid_index].fill_value,
                chunk_shape=chunk_shape,
                chunk_path_separator=chunk_path_separator,
                codecs=pyramids[pyramid_index].codecs,
                dimension_names=tuple(axes),
            )
            new_arrays.append(new_array)
            # Add to metadata list
            zyx_translation = tuple(
                zyx_scale[i] / 2 - src_zyx_scale[i] / 2 for i in range(zyx_ndim)
            )
            d = {
                "path": array_path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": (*tc_scale, *zyx_scale)},
                    {
                        "type": "translation",
                        "translation": (*tc_translation, *zyx_translation),
                    },
                ],
            }
            dataset_lists[pyramid_index].append(d)

        max_memory = int(max_memory)
        if max_memory <= 0:
            if steps is not None:
                steps.append(
                    {"level": 0, "nlevels": len(new_arrays) - 1, "kept_in_memory": True}
                )
            _write_simple_pyramid(
                pyramids[pyramid_index].src_array, new_arrays, zyx_ndim, method
            )
        else:
            _write_large_pyramid(
                pyramids[pyramid_index].src_array,
                new_arrays,
                zyx_ndim,
                method,
                chunk_shape,
                max_memory,
                steps,
            )

    # Write a group for each pyramid subdirectory.
    # Note that this currently are plain groups, we could make them ome-zarr groups, but maybe not because it would be an imcomplete file?
    for pyramid_path in pyramid_paths:
        simplezarr.ZarrGroup.create(store, f"{path}/{pyramid_path}")

    # Write toplevel ZarrGroup for this pyramid
    zarr_group = simplezarr.ZarrGroup.create(store, path, attributes=attributes)

    return zarr_group


def _write_simple_pyramid(src_array, new_arrays, zyx_ndim, method):
    """Straightforward algorithm to create a pyramid."""

    tc_shape = src_array.shape[:-zyx_ndim]
    inner_indices = _get_inner_indices(tc_shape)

    # The pyramid arrays may have a different dtype than the source array
    src_data = _cast_to_dtype(src_array[...].get_now(), new_arrays[0].dtype)
    data = src_data

    futures = []
    for level, new_array in enumerate(new_arrays):
        if level > 0:
            data = _downsample_to_shape(
                data, new_array.shape, zyx_ndim, method, inner_indices
            )

        # Write
        fut = new_array[...].set_soon(data)
        futures.append(fut)

    # Wait for all writing to finish
    for fut in futures:
        fut.result()


def _write_large_pyramid(
    src_array, new_arrays, zyx_ndim, method, chunk_shape, max_memory, steps=None
):
    """Depth-first algorithm to create a pyramid, so it can handle data of any size without running out of memory.

    The pyramid is built bottom-up in steps. Each step selects a set of chunks
    at the current bottom level, and downsamples it, level by level, until the
    selection is exactly a single chunk: the intermediate top of that step.
    The number of selected chunks is a power of 2 in each spatial dimension,
    and is calculated from the memory constraint. Because the selection is
    chunk-aligned at every level, all writes are whole-chunk writes, which can
    be issued asynchronously.

    At the intermediate top of a step there are two options:
    * If the full intermediate level fits in memory, it is kept in memory
      while the bottom level is processed, and the remaining levels are
      written using the same approach as _write_simple_pyramid.
    * Otherwise the intermediate level is only written to disk, and the next
      step loads those chunks from disk again. This is slower, but uses less
      memory.

    If the given `steps` list is not None, a dict is appended to it for each
    step, specifying the bottom level, the number of levels created, and
    whether the intermediate top was kept in memory. This is for diagnostics/testing.
    """

    nlevels = len(new_arrays)
    tc_shape = src_array.shape[:-zyx_ndim]
    tc_slices = tuple(slice(None) for _ in tc_shape)
    inner_indices = _get_inner_indices(tc_shape)
    tc_size = math.prod(tc_shape)
    itemsize = np.dtype(new_arrays[0].dtype).itemsize
    zyx_chunk_shape = chunk_shape[-zyx_ndim:]

    # Calculate how many chunks we can select at the bottom, given the memory
    # constraint. The selection is a power of 2 chunks in each spatial
    # dimension, so that the intermediate top is exactly a single chunk.
    levels_per_step = _get_levels_per_step(
        max_memory, tc_size, zyx_chunk_shape, itemsize, nlevels - 1
    )

    # Writes are issued asynchronously, but pending data counts toward the
    # memory usage, so we flush the oldest pending write when over budget.
    pending = deque()  # (future, nbytes)
    pending_nbytes = 0
    write_budget = 0

    def submit_write(zarr_array, index, data):
        nonlocal pending_nbytes
        fut = zarr_array[index].set_soon(data)
        nbytes = data.nbytes
        pending.append((fut, nbytes))
        pending_nbytes += nbytes
        while pending and pending_nbytes > write_budget:
            fut2, nbytes2 = pending.popleft()
            pending_nbytes -= nbytes2
            fut2.result()

    level = 0
    while True:
        # The number of levels that this step creates
        k = min(levels_per_step, nlevels - 1 - level)

        # Decide whether we can keep the intermediate top of this step in
        # memory, so that we can do the remaining levels with the simple
        # approach. If not, the next step loads the chunks from disk again.
        step_nbytes = _step_nbytes(k, tc_size, zyx_chunk_shape, itemsize)
        keep_in_memory = False
        intermediate = None
        if level + k < nlevels - 1:
            inter_nbytes = _array_nbytes(new_arrays[level + k])
            tail_nbytes = sum(
                _array_nbytes(new_arrays[j]) for j in range(level + k, nlevels)
            )
            keep_in_memory = max(step_nbytes + inter_nbytes, tail_nbytes) <= max_memory
            if keep_in_memory:
                intermediate = np.zeros(
                    new_arrays[level + k].shape,
                    np.dtype(new_arrays[level + k].dtype),
                )
                write_budget = max_memory - step_nbytes - inter_nbytes
            else:
                write_budget = max_memory - step_nbytes
        else:
            write_budget = max_memory - step_nbytes

        if steps is not None:
            steps.append(
                {"level": level, "nlevels": k, "kept_in_memory": keep_in_memory}
            )

        # Select chunks at the current bottom level, and process each selection
        phase_src = src_array if level == 0 else new_arrays[level]
        zyx_shape = phase_src.shape[-zyx_ndim:]
        sel_steps = tuple((2**k) * c for c in zyx_chunk_shape)
        starts_per_dim = [range(0, zyx_shape[i], sel_steps[i]) for i in range(zyx_ndim)]
        for starts in itertools.product(*starts_per_dim):
            sel_slices = tuple(
                slice(starts[i], min(starts[i] + sel_steps[i], zyx_shape[i]), 1)
                for i in range(zyx_ndim)
            )
            data = phase_src[*tc_slices, *sel_slices].get_now()
            if level == 0:
                # The source data may have a different dtype than the pyramid
                data = _cast_to_dtype(data, new_arrays[0].dtype)
                submit_write(new_arrays[0], (*tc_slices, *sel_slices), data)
            slices = sel_slices
            for j in range(1, k + 1):
                new_slices = tuple(slice(s.start // 2, s.stop // 2, 1) for s in slices)
                new_zyx_shape = tuple(s // 2 for s in data.shape[-zyx_ndim:])
                new_data = _downsample_to_shape(
                    data,
                    (*tc_shape, *new_zyx_shape),
                    zyx_ndim,
                    method,
                    inner_indices,
                )
                submit_write(new_arrays[level + j], (*tc_slices, *new_slices), new_data)
                if j == k and intermediate is not None:
                    intermediate[*tc_slices, *new_slices] = new_data
                data = new_data
                slices = new_slices

        # Wait for this step's writes to finish
        while pending:
            fut, nbytes = pending.popleft()
            pending_nbytes -= nbytes
            fut.result()

        level += k

        if intermediate is not None:
            # Do the remaining levels from the in-memory intermediate level,
            # using the same approach as _write_simple_pyramid
            if steps is not None:
                steps.append(
                    {
                        "level": level,
                        "nlevels": nlevels - 1 - level,
                        "kept_in_memory": True,
                    }
                )
            data = intermediate
            futures = []
            for j in range(level + 1, nlevels):
                data = _downsample_to_shape(
                    data, new_arrays[j].shape, zyx_ndim, method, inner_indices
                )
                futures.append(new_arrays[j][...].set_soon(data))
            for fut in futures:
                fut.result()
            level = nlevels - 1

        if level >= nlevels - 1:
            break


def _get_levels_per_step(max_memory, tc_size, zyx_chunk_shape, itemsize, max_levels):
    """Calculate the number of chunks to select at the bottom, given the memory constraint.

    The selection is a power of 2 chunks in each spatial dimension, so that
    the intermediate top is exactly a single chunk. At least one level is
    selected per step, so that we always make progress.
    """
    levels_per_step = 1
    while (
        levels_per_step < max_levels
        and _step_nbytes(levels_per_step + 1, tc_size, zyx_chunk_shape, itemsize)
        <= max_memory
    ):
        levels_per_step += 1
    return levels_per_step


def _step_nbytes(levels_per_step, tc_size, zyx_chunk_shape, itemsize):
    """The approximate amount of memory needed to process one selection of 2**levels_per_step chunks
    (per spatial dimension), including the intermediate downsampled selections.
    """
    zyx_ndim = len(zyx_chunk_shape)
    chunk_pixels = math.prod(zyx_chunk_shape)
    total_pixels = sum(
        chunk_pixels * 2 ** (i * zyx_ndim) for i in range(levels_per_step + 1)
    )
    return itemsize * tc_size * total_pixels


def _array_nbytes(zarr_array):
    return math.prod(zarr_array.shape) * np.dtype(zarr_array.dtype).itemsize


def _downsample_to_shape(data, new_shape, zyx_ndim, method, inner_indices):
    """Downsample (local mean) the last zyx_ndim dimensions of the given array to the given shape."""
    new_data = np.zeros(new_shape, data.dtype)
    new_zyx_shape = new_data.shape[-zyx_ndim:]
    subslices2 = tuple(slice(0, s, 1) for s in new_zyx_shape)
    subslices1 = tuple(slice(0, s.stop * 2, 1) for s in subslices2)
    for index in inner_indices:  # for each time/channel combi
        new_data[*index, *subslices2] = _downsample(data[*index, *subslices1], method)
    return new_data


def _cast_to_dtype(data, dtype):
    """Cast the given data to the given dtype.

    When casting to an integer dtype, values are rounded (instead of
    truncated) and clipped to the dtype's range (instead of wrapping around).
    """
    dtype = np.dtype(dtype)
    if data.dtype == dtype:
        return data
    if np.issubdtype(dtype, np.integer):
        b = np.rint(data) if np.issubdtype(data.dtype, np.floating) else data
        info = np.iinfo(dtype)
        return np.clip(b, info.min, info.max).astype(dtype)
    return data.astype(dtype)


def _get_inner_indices(tc_shape):
    inner_indices = [()]
    for n in tc_shape:
        new_inner_indices = []
        for j in inner_indices:
            new_inner_indices += [(*j, i) for i in range(n)]
        inner_indices = new_inner_indices
    return inner_indices


def _downsample(a, method):
    assert method == "local_mean"  # only one method implemented right now
    if a.ndim == 2:
        b = a[::2, ::2].astype(float)
        b += a[1::2, ::2]
        b += a[::2, 1::2]
        b += a[1::2, 1::2]
        b /= 4
    else:  # ndim == 3:
        b = a[::2, ::2, ::2].astype(float)
        b += a[1::2, ::2, ::2]
        b += a[::2, 1::2, ::2]
        b += a[1::2, 1::2, ::2]
        b += a[::2, ::2, 1::2]
        b += a[1::2, ::2, 1::2]
        b += a[::2, 1::2, 1::2]
        b += a[1::2, 1::2, 1::2]
        b /= 8

    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        np.rint(b, out=b)
        np.clip(b, info.min, info.max, out=b)

    return b.astype(a.dtype)
