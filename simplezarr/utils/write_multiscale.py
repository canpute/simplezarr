"""
Support for writing multiscale images, a.ka. ome-zarr pyramids.

Supports huge datasets by building the pyramid up in pieces. But adopts a depth-first approach and makes use of parallel writes to realize fast throughput.

Currently it seems to be about 4 times faster than ome-zarr, but we have not applied all tricks yet.

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

import numpy as np
import simplezarr
from simplezarr.utils.units import SPACE_UNITS, TIME_UNITS


def write_ome_zarr_pyramid(
    store: simplezarr.BaseStore,
    path: str,
    src_array: simplezarr.ZarrArray | np.ndarray,
    axes: tuple[str],
    *,
    name: str = "",
    space_unit: str = "",
    base_scale: float | tuple[float, ...] = 1.0,
    nlevels: int = 0,
    # args that inherit from src array, but can be overriden
    chunk_shape: tuple[int] | None = None,
    dtype: str | None = None,
    fill_value: object = None,
    chunk_path_separator: str | None = None,
    codecs: list[dict] | None = None,
):
    """
    Write an ome-zarr image pyramid to the given store.

    Arguments:
        store : simplezarr.BaseStore
            The simplezarr store object to store the Zarr pyramid in.
        path : str
            The path in the store for the group that represents the pyramid.
        src_array : ZarrArray or np.ndarray
            The source array. An array of 2, 3, 4 or 5 dimensions (t, c, z, y, x).
        axes : str
            The names of the dimensions, e.g. "xyz" or "cyx", with "t" for time,
            "c" for channels, and "zyx" for the spatial dimensions.
        name : str
            The name for the multi-scale image. Default empty string.
        space_unit : str
            The spatial unit. Default empty string which means not set.
        base_scale : float or tuple[float]
            The scale of the original data (scale 0). Default 1.
        nlevels : int
            The number of scale levels to create. If zero, determine automatically
            so that the highest level is <= chunk shape.
        chunk_shape : int or tuple[int]
            The chunk_shape for all levels in the new array.
        dtype : str
            The data type of the new array. If not given, uses the array's dtype.
        fill_value
            The value to fill in when no chunk is present. If not given, uses the array's fill_value.
        chunk_path_separator : str
            The path-separator to use for chunks. If not given, uses the array's chunk_path_separator.
        codecs : list[dict] | None
            Same as ``ZarrArray.create()``. If not given, uses the array's chunk_path_separator.
    """

    # TODO: base_scale should use src_array's scale by default

    # Allow numpy array input -> convert to an in-memory ZarrArray
    numpy_array = None
    if isinstance(src_array, np.ndarray):
        numpy_array = src_array
        s = simplezarr.MemoryStore()
        src_array = simplezarr.ZarrArray.create(
            s, "", numpy_array.shape, numpy_array.dtype
        )
        src_array[...].set_now(numpy_array)
    if not isinstance(src_array, simplezarr.ZarrArray):
        raise TypeError(
            f"write_ome_zarr() Expected an array, got {src_array.__class__.__name__} {src_array}"
        )
    if   not (src_array.ndim >= 2 and src_array.ndim <= 5):
        raise ValueError(
            f"write_ome_zarr() Expected an array with 2, 3, 4 or 5 dimensions, got shape {src_array.shape}"
        )

    ndim = src_array.ndim

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
    if not isinstance(name, str):
        raise TypeError(f"write_ome_zarr() name must be str, got {name!r}")

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
    for name in axes:
        typ = {"t": "time", "c": "channel"}.get(name, "space")
        unit = {"t": "", "c": ""}.get(name, space_unit)
        d = {"name": name, "type": typ}
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
    tc_ndim = ndim - zyx_ndim

    # Overloadable from source array
    if chunk_shape is None:
        chunk_shape = src_array.chunk_shape
        zyx_chunk_shape = chunk_shape[-zyx_ndim:]
    elif isinstance(chunk_shape, int):
        zyx_chunk_shape = (chunk_shape,) * chunk_shape
        chunk_shape = (*src_array.shape[:-zyx_ndim], *zyx_chunk_shape)
    else:
        assert len(chunk_shape) == ndim
        assert all(isinstance(i, int) for i in chunk_shape)
        zyx_chunk_shape = chunk_shape[-zyx_ndim:]
    if dtype is None:
        dtype = src_array.dtype
    if fill_value is None:
        fill_value = src_array._fill_value
    if chunk_path_separator is None:
        chunk_path_separator = src_array._chunk_path_separator
    if codecs is None:
        codecs = src_array._codecs

    # Select downsample method
    method = "local_mean"  # our API enum, also ends up as 'type' in metadata
    if method not in {"local_mean"}:
        raise ValueError(f"write_ome_zarr() unexpected method {method!r}")
    dowmsample_metadata = {
        "description": f"Pyramid build with simplezarr.utils.pyramid using the {method!r} method"
    }

    # Build dataset list, we append to this list during processing below
    dataset_list = []

    # Based on ome-zarr 0.5
    attributes = {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "name": name,
                    "axes": axes_list,
                    "datasets": dataset_list,
                    "type": method,
                    "metadata": dowmsample_metadata,
                }
            ],
        }
    }

    # Prepare arrays

    path_prefix = "level"
    src_shape = src_array.shape

    src_zyx_shape = tuple(src_shape[i] for i in range(ndim) if axes[i] in "zyx")
    src_zyx_scale = tuple(base_scale[i] for i in range(ndim) if axes[i] in "zyx")

    tc_shape = tuple(src_shape[i] for i in range(ndim) if axes[i] in "tc")
    tc_scale = tuple(base_scale[i] for i in range(ndim) if axes[i] in "tc")
    tc_translation = tuple(0 for _ in tc_scale)

    new_arrays = []
    zyx_shape = src_zyx_shape
    zyx_scale = src_zyx_scale

    for level in range(100):
        if level > 0:
            zyx_shape = tuple(s // 2 for s in zyx_shape)
            zyx_scale = tuple(s * 2 for s in zyx_scale)
        # Decide when to stop
        if nlevels == 0:
            if level >= 1 and any(
                zyx_shape[i] < zyx_chunk_shape[i] for i in range(zyx_ndim)
            ):
                break
        elif level >= nlevels:
            break
        array_path = f"{path_prefix}{level}"
        new_array = simplezarr.ZarrArray.create(
            store,
            f"{path}/{array_path}",
            (*tc_shape, *zyx_shape),
            dtype,
            fill_value=fill_value,
            chunk_shape=chunk_shape,
            chunk_path_separator=chunk_path_separator,
            codecs=codecs,
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
                # {"type": "translation", "translation": (*tc_translation, *zyx_translation)}
            ],
        }
        dataset_list.append(d)

    # Write toplevel ZarrGroup for this pyramid
    zarr_group = simplezarr.ZarrGroup.create(store, path, attributes=attributes)

    # _write_simple_pyramid(src_array, new_arrays, zyx_ndim, method)
    _write_large_pyramid(src_array, new_arrays, zyx_ndim, method, chunk_shape)

    return zarr_group


def _write_simple_pyramid(src_array, new_arrays, zyx_ndim, method):
    tc_shape = src_array.shape[:-zyx_ndim]
    inner_indices = _get_inner_indices(tc_shape)

    src_zyx_shape = src_array.shape[-zyx_ndim:]
    src_slices = tuple(slice(0, s // 2 * 2, 1) for s in src_zyx_shape)
    src_data = src_array[...].get_now()

    for level, new_array in enumerate(new_arrays):
        if level == 0:
            new_data = src_data
        else:
            new_data = np.zeros(new_array.shape, new_array.dtype)
            new_zyx_shape = new_data.shape[-zyx_ndim:]
            subslices2 = tuple(slice(0, s // 2 * 2, 1) for s in new_zyx_shape)
            subslices1 = tuple(slice(0, s.stop * 2, 1) for s in subslices2)
            for index in inner_indices:  # for each time/channel combi
                new_data[*index, *subslices2] = _downsample(
                    data[*index, *subslices1], method
                )

        # Write
        fut = new_array[...].set_soon(new_data)
        # TODO:we could not-wait for the writing, and wait for futures at the end
        fut.result()

        # Next
        data = new_data


def _write_large_pyramid(src_array, new_arrays, zyx_ndim, method, chunk_shape):
    tc_shape = src_array.shape[:-zyx_ndim]
    inner_indices = _get_inner_indices(tc_shape)
    tc_slices = [slice(None) for _ in tc_shape]

    src_zyx_shape = src_array.shape[-zyx_ndim:]
    src_slices = tuple(slice(0, s // 2 * 2, 1) for s in src_zyx_shape)

    zyx_chunk_shape = chunk_shape[-zyx_ndim:]

    # Determine pyramid-building steps
    levels_per_step = 3
    chunk_multiply_factor = 2**levels_per_step

    pyramid_base_size = tuple(chunk_multiply_factor * i for i in zyx_chunk_shape)

    zyx_slices_list = []
    if zyx_ndim == 2:
        for dy in range(0, src_zyx_shape[-2], pyramid_base_size[-2]):
            for dx in range(0, src_zyx_shape[-1], pyramid_base_size[-1]):
                slices = (
                    slice(dy, dy + pyramid_base_size[-2], 1),
                    slice(dx, dx + pyramid_base_size[-1], 1),
                )
                zyx_slices_list.append(slices)
    else:
        assert zyx_ndim == 3
        for dz in range(0, src_zyx_shape[-3], pyramid_base_size[-3]):
            for dy in range(0, src_zyx_shape[-2], pyramid_base_size[-2]):
                for dx in range(0, src_zyx_shape[-1], pyramid_base_size[-1]):
                    slices = (
                        slice(dz, dz + pyramid_base_size[-3], 1),
                        slice(dy, dy + pyramid_base_size[-2], 1),
                        slice(dx, dx + pyramid_base_size[-1], 1),
                    )
                    zyx_slices_list.append(slices)

    print(f"Source array's shape is {src_array.shape}, spatially {src_zyx_shape}")
    print(f"Identified {len(zyx_slices_list)} initial pyramid building steps")

    for step_index, zyx_slices in enumerate(zyx_slices_list):
        print(f"Step {step_index}/{len(zyx_slices_list)}")

        # The zyx_slices are the slice in the full array, we reduce it in the loop below
        src_data = src_array[*tc_slices, *zyx_slices].get_now()
        src_slices = tuple(
            slice(zyx_slices[i].start, min(zyx_slices[i].stop, src_array.shape[i]), 1)
            for i in range(zyx_ndim)
        )

        for level, new_array in enumerate(new_arrays):
            # Downscale

            if level == 0:
                new_data = src_data
                new_slices = src_slices
            else:
                new_slices = tuple(
                    slice(s.start // 2, s.stop // 4 * 2, 1) for s in slices
                )
                new_zyx_shape = tuple(data.shape[i] // 4 * 2 for i in range(zyx_ndim))
                new_shape = (*tc_shape, *new_zyx_shape)
                new_data = np.zeros(new_shape, data.dtype)
                subslices2 = tuple(slice(0, s // 2 * 2, 1) for s in new_zyx_shape)
                subslices1 = tuple(slice(0, s.stop * 2, 1) for s in subslices2)
                for index in inner_indices:  # for each time/channel combi
                    new_data[*index, *subslices2] = _downsample(
                        data[*index, *subslices1], method
                    )

            # Write
            fut = new_array[*tc_slices, *new_slices].set_soon(new_data)
            # TODO:we could not-wait for the writing, and wait for futures at the end
            fut.result()

            # Next
            data = new_data
            slices = new_slices


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
