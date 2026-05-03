"""Support for multiscale images (most notably ome-zarr)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import simplezarr
from simplezarr.utils.units import SPACE_UNITS, TIME_UNITS


@dataclass
class MultiscaleInfo:
    """Represents a single multiscale image."""

    name: str
    unit: str
    unit_factor: float  # TODO: USE app-specific scale factor
    scales: list[ScaleInfo]


@dataclass
class ScaleInfo:
    """Information that represents a single scale in a multiscale image."""

    level: int
    array: simplezarr.ZarrArray
    path: str
    ref_scale: float
    spatial_shape: tuple[int, ...]
    spatial_scale: tuple[float, ...]
    spatial_offset: tuple[float, ...]
    spatial_chunk_shape: tuple[float, ...]
    nchannels: int
    ntimes: int
    numel: int


def create_scale_infos_from_zarr_node(
    zarr_node: simplezarr.ZarrGroup,
) -> list[MultiscaleInfo]:
    """Given a zarr node, produce a list of MultiscaleInfo objects (one per multiscale dict in the metadata).

    This information is geared for use by the ``simplezarr.chunkpool.ChunkPool``.
    """
    # The ome-zarr spec: https://ngff.openmicroscopy.org/specifications/

    zarr_info = zarr_node.metadata
    if "ome" in zarr_info["attributes"]:
        return create_scale_infos_from_ome_zarr_group(zarr_node)
    elif isinstance(zarr_node, simplezarr.ZarrArray):
        # TODO: test this code path
        return create_scale_infos_from_zarr_array(zarr_node)
    else:
        raise RuntimeError(f"Cannot get scale infos from {zarr_node!r}")


def create_scale_infos_from_zarr_array(
    zarr_array: simplezarr.ZarrArray,
) -> list[MultiscaleInfo]:
    """Mimic a multiscale image even though this is just a single image."""

    # There's no metadata to know what are spatial dims vs channels vs time, so just guess
    ndim = len(zarr_array.shape)
    if ndim == 1:
        space_dims = 1
        channel_dim = None
        time_dim = None
    elif ndim == 2:
        space_dims = 2
        channel_dim = None
        time_dim = None
    elif ndim == 3:
        space_dims = 3
        channel_dim = None
        time_dim = None
    elif ndim == 4:
        space_dims = 3
        channel_dim = 0
        time_dim = None
    elif ndim == 5:
        space_dims = 3
        channel_dim = 1
        time_dim = 0
    else:
        raise RuntimeError("Unsupported dimensions")

    full_scale = [1] * ndim
    full_translation = [0] * ndim
    spatial_scale = tuple(full_scale[-space_dims:])
    ref_scale = sum(spatial_scale) / len(spatial_scale)

    scale_info = ScaleInfo(
        level=0,
        array=zarr_array,
        path=zarr_array.path,
        spatial_offset=tuple(full_translation[-space_dims:]),
        spatial_scale=tuple(full_scale[-space_dims:]),
        spatial_shape=tuple(zarr_array.shape[-space_dims:]),
        spatial_chunk_shape=tuple(zarr_array.chunk_shape[-space_dims:]),
        ref_scale=ref_scale,
        nchannels=zarr_array.shape[channel_dim] if channel_dim is not None else 0,
        ntimes=zarr_array.shape[time_dim] if time_dim is not None else 0,
        numel=0,
    )
    # TODO: set numel. Or remove that field??

    return [MultiscaleInfo("", [scale_info])]


def create_scale_infos_from_ome_zarr_group(
    zarr_group: simplezarr.ZarrGroup,
) -> list[MultiscaleInfo]:
    zarr_info = zarr_group.metadata
    ome_info = zarr_info["attributes"]["ome"]
    ome_version = ome_info["version"]  # this code kind of assumes 0.5

    # “multiscales” contains a list of dictionaries where each entry describes a multiscale image.
    assert "multiscales" in ome_info

    multiscale_images = []

    for multiscale_dict in ome_info["multiscales"]:
        name = multiscale_dict.get("name", "")  # SHOULD field
        downscale_type = multiscale_dict.get("type", "")  # SHOULD field
        downscale_metadata = multiscale_dict.get("metadata", {})  # SHOULD field

        # Process axes metadata

        axes_info = multiscale_dict["axes"]  # MUST field

        # Check axes names
        axes_names = tuple(x["name"] for x in axes_info)  # MUST field
        axes_types = tuple(x["type"] for x in axes_info)  # SHOULD field
        if not all(an in "tczyx" for an in axes_names):
            raise RuntimeError(f"Zarr data has unexpected axes: {axes_names}")
        space_dims = axes_types.count("space")
        assert all(at == "space" for at in axes_types[-space_dims:])
        assert all(at != "space" for at in axes_types[0:-space_dims])
        time_dim = channel_dim = None
        if axes_types[0] == "time":
            time_dim = 0
        if axes_types[0] == "channel":
            channel_dim = 0
        elif axes_types[1] == "channel":
            channel_dim = 1

        # TODO: store axes_names somewhere

        # Get and check unit
        unit = axes_info[-1].get("unit", "").lower()  # SHOULD field
        if unit not in SPACE_UNITS:
            raise RuntimeError(f"Zarr data has unexpected space unit: {unit!r}")

        # Process coordinateTransformations

        global_transformations = multiscale_dict.get(
            "coordinateTransformations", None
        )  # MAY field

        global_transforms = {}
        if global_transformations is not None:
            global_transforms = {
                t["type"]: t[t["type"]] for t in global_transformations
            }

        # Process datasets metadata

        datasets = multiscale_dict["datasets"]  # MUST field

        # Collect scale info for each resolution
        # MUST be ordered from largest (i.e. highest resolution) to smallest.
        scale_infos = []
        for level, dataset_dict in enumerate(datasets):
            path = dataset_dict["path"]  # MUST field
            transforms = {
                t["type"]: t[t["type"]]
                for t in dataset_dict["coordinateTransformations"]  # MUST field
            }
            zarr_array = zarr_group[path]
            # scale_dict = zarr_info["consolidated_metadata"]["metadata"][dataset_dict["path"]]
            # shape = scale_dict["shape"]
            full_scale = transforms.get("scale", [1] * len(axes_types))
            full_translation = transforms.get("translation", [0] * len(axes_types))
            for global_type, global_value in global_transforms.items():
                if global_type == "scale":
                    full_scale = [
                        s1 * s2 for s1, s2 in zip(full_scale, global_value, strict=True)
                    ]
                    full_translation = [
                        t1 * s2
                        for t1, s2 in zip(full_translation, global_value, strict=True)
                    ]
                elif global_type == "translation":
                    full_translation = [
                        t1 + t2
                        for t1, t2 in zip(full_translation, global_value, strict=True)
                    ]

            spatial_scale = tuple(full_scale[-space_dims:])
            ref_scale = sum(spatial_scale) / len(spatial_scale)

            si = ScaleInfo(
                level=level,
                array=zarr_array,
                path=path,
                spatial_offset=tuple(full_translation[-space_dims:]),
                spatial_scale=tuple(full_scale[-space_dims:]),
                spatial_shape=tuple(zarr_array.shape[-space_dims:]),
                spatial_chunk_shape=tuple(zarr_array.chunk_shape[-space_dims:]),
                ref_scale=ref_scale,
                nchannels=zarr_array.shape[channel_dim]
                if channel_dim is not None
                else 0,
                ntimes=zarr_array.shape[time_dim] if time_dim is not None else 0,
                numel=0,
            )
            si.numel = int(np.prod(si.spatial_shape)) * si.nchannels  # TODO: fix
            scale_infos.append(si)

        # Sort, by highest res first
        # The “path's MUST be ordered from largest (i.e. highest resolution) to smallest, but we sort anyway
        scale_infos.sort(key=lambda si: si.ref_scale)
        multiscale_images.append(
            MultiscaleInfo(name, unit, SPACE_UNITS[unit], scale_infos)
        )

    return multiscale_images
