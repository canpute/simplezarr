"""
Support for multiscale images, most notably ome-zarr.

The purpose of this utility is to examine the metadata of a Zarr file and produce
typed structures to easily process the data further.

OME-Zarr, a.k.a. next-generation file format (NGFF) builds on Zarr version 3, to
define hierarchical datasets. This module implements the "multiscales" metadata,
ignoring the transitional "bioformats2raw.layout" and "omero" metadata. In its
current form, the "labels", "plate" and "well" metadata are also ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import simplezarr
from simplezarr.utils.units import SPACE_UNITS
from simplezarr.misc import logger


__all__ = ["MultiscaleInfo", "ScaleInfo"]

MSG_PREFIX = "simplezarr.utils.multiscale"


@dataclass
class MultiscaleInfo:
    """Represents a single multiscale image."""

    name: str  #: the name of this multiscale image
    axes_names: tuple[str, ...]  #: the names of the axes/dimensions of the array
    unit: str  #: the unit for the spatial dimension
    unit_factor: float  #: the factor to map the unit to meters
    scales: list[ScaleInfo]  #: more info per scale

    @classmethod
    def from_zarr_node(cls, zarr_node: simplezarr.ZarrGroup) -> list[MultiscaleInfo]:
        """Given a zarr node, produce a list of MultiscaleInfo objects (one per multiscale dict in the metadata).

        This information is geared for use by the ``simplezarr.chunkpool.ChunkPool``.
        """
        # The ome-zarr spec: https://ngff.openmicroscopy.org/specifications/

        attributes = zarr_node.metadata.get("attributes", {})

        if "ome" in attributes:
            return create_scale_infos_from_ome_zarr_group(zarr_node)
        elif isinstance(zarr_node, simplezarr.ZarrArray):
            return create_scale_infos_from_zarr_array(zarr_node)
        else:  # no-cover
            raise TypeError(f"Cannot get scale infos from {zarr_node!r}")


@dataclass
class ScaleInfo:
    """Information that represents a single scale in a multiscale image."""

    array: simplezarr.ZarrArray  #: the ZarrArray object
    level: int  #: the integer level in the multiscale stack, 0 being the highest-resolution
    mean_scale: float  #: the reference scale for this scale layer in world units (the average of the spatial scales)
    spatial_shape: tuple[int, ...]  #: The shape of the spatial dimensions
    spatial_scale: tuple[float, ...]  #: the scale factor for the spatial dimensions
    spatial_offset: tuple[float, ...]  #: the offset for the spatial dimensions
    spatial_chunk_shape: tuple[float, ...]  # The chunk shape for the spatial dimensions
    nchannels: int  #: the number of channels for this image
    ntimes: int  #: the number of time-frames for this image


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
    else:  # no-cover
        raise TypeError("Unsupported dimensions")

    full_scale = [1] * ndim
    full_translation = [0] * ndim
    spatial_scale = tuple(full_scale[-space_dims:])
    mean_scale = sum(spatial_scale) / len(spatial_scale)

    scale_info = ScaleInfo(
        array=zarr_array,
        level=0,
        mean_scale=mean_scale,
        spatial_offset=tuple(full_translation[-space_dims:]),
        spatial_scale=tuple(full_scale[-space_dims:]),
        spatial_shape=tuple(zarr_array.shape[-space_dims:]),
        spatial_chunk_shape=tuple(zarr_array.chunk_shape[-space_dims:]),
        nchannels=zarr_array.shape[channel_dim] if channel_dim is not None else 0,
        ntimes=zarr_array.shape[time_dim] if time_dim is not None else 0,
    )

    return [
        MultiscaleInfo(
            name="",
            axes_names=tuple("" for _ in range(ndim)),
            unit="",
            unit_factor=1,  # safer than zero or nan
            scales=[scale_info],
        )
    ]


def create_scale_infos_from_ome_zarr_group(
    zarr_group: simplezarr.ZarrGroup,
) -> list[MultiscaleInfo]:
    zarr_info = zarr_group.metadata
    ome_info = zarr_info["attributes"]["ome"]
    ome_version = tuple(int(x) for x in str(ome_info["version"]).split("."))
    assert ome_version >= (0, 5)  # this code assumes 0.5

    # “multiscales” contains a list of dictionaries where each entry describes a multiscale image.
    assert "multiscales" in ome_info

    multiscale_images = []

    for multiscale_dict in ome_info["multiscales"]:
        name = multiscale_dict.get("name", "")  # SHOULD field
        _downscale_type = multiscale_dict.get("type", "")  # SHOULD field
        _downscale_metadata = multiscale_dict.get("metadata", {})  # SHOULD field

        # Process axes metadata

        axes_info = multiscale_dict["axes"]  # MUST field

        # Check axes names
        axes_names = tuple(x["name"] for x in axes_info)  # MUST field
        axes_types = tuple(x["type"] for x in axes_info)  # SHOULD field
        if not all(an in "tczyx" for an in axes_names):  # no-cover
            raise TypeError(f"Zarr data has unexpected axes: {axes_names}")
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

        # Get and check unit - # SHOULD field
        units = [d.get("unit", "").lower() for d in axes_info[-space_dims:]]
        units = [unit for unit in units if unit]
        unit = "" if not units else units[-1]
        if unit not in SPACE_UNITS:  # no-cover
            raise TypeError(f"{MSG_PREFIX}: unexpected space unit: {unit!r}")
        elif not unit:  # no-cover
            logger.warning(f"{MSG_PREFIX}: spatial dimensions don't not have a unit.")
        elif len(set(units)) > 1:  # no-cover
            logger.warning(
                f"{MSG_PREFIX}: spatial dimensions define different units, using the last ('{unit}')"
            )

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
            mean_scale = sum(spatial_scale) / len(spatial_scale)

            si = ScaleInfo(
                array=zarr_array,
                level=level,
                mean_scale=mean_scale,
                spatial_offset=tuple(full_translation[-space_dims:]),
                spatial_scale=tuple(full_scale[-space_dims:]),
                spatial_shape=tuple(zarr_array.shape[-space_dims:]),
                spatial_chunk_shape=tuple(zarr_array.chunk_shape[-space_dims:]),
                nchannels=zarr_array.shape[channel_dim]
                if channel_dim is not None
                else 0,
                ntimes=zarr_array.shape[time_dim] if time_dim is not None else 0,
            )
            scale_infos.append(si)

        # Sort, by highest res first
        # The “path's MUST be ordered from largest (i.e. highest resolution) to smallest, but we sort anyway
        scale_infos.sort(key=lambda si: si.mean_scale)
        multiscale_images.append(
            MultiscaleInfo(
                name=name,
                axes_names=tuple(axes_names),
                unit=unit,
                unit_factor=SPACE_UNITS[unit] if unit else 1,
                scales=scale_infos,
            )
        )

    return multiscale_images
