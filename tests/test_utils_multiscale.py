from simplezarr import MemoryStore, open_zarr, ZarrArray
from simplezarr.utils.multiscale import (
    create_scale_infos_from_zarr_node,
    MultiscaleInfo,
    ScaleInfo,
)


store_data1 = {
    "zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [100, 100],
            "data_type": "uint16",
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": [100, 100]
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
    # "c/0-0": np.full((100, 100), 100, np.uint16).tobytes(),  # -> data is never read
}

store_data2 = {
    "zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": {
                    "version": "0.5",
                    "multiscales": [
                        {
                            "name": "a multiscale image",
                            "axes": [
                                {"name": "c", "type": "channel"},
                                {"name": "y", "type": "space", "unit": "micrometer"},
                                {"name": "x", "type": "space", "unit": "micrometer"}
                            ],
                            "datasets": [
                                {
                                    "path": "scale1",
                                    "coordinateTransformations": [
                                        {"type": "scale", "scale": [1, 1, 1]},
                                        {"type": "translation", "translation": [0, 0, 0]}
                                    ]
                                },
                                {
                                    "path": "scale2",
                                    "coordinateTransformations": [
                                        {"type": "scale", "scale": [2, 2, 2]},
                                        {"type": "translation", "translation": [0, 0, 0]}
                                    ]
                                },
                                {
                                    "path": "scale3",
                                    "coordinateTransformations": [
                                        {"type": "scale", "scale": [4, 4, 4]},
                                        {"type": "translation", "translation": [0, 0, 0]}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        }
        """.encode(),
    "scale1/zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [2, 100, 100],
            "data_type": "uint16",
            "chunk_grid": {
                "name": "regular",
                "configuration": { "chunk_shape": [2, 100, 100] }
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "-"}
            },
            "codecs": [{
                "name": "bytes",
                "configuration": {"endian": "little"}
            }],
            "fill_value": "0"
        }
        """.encode(),
    "scale2/zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [2, 50, 50],
            "data_type": "uint16",
            "chunk_grid": {
                "name": "regular",
                "configuration": { "chunk_shape": [2, 50, 50] }
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "-"}
            },
            "codecs": [{
                "name": "bytes",
                "configuration": {"endian": "little"}
            }],
            "fill_value": "0"
        }
        """.encode(),
    "scale3/zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [2, 25, 25],
            "data_type": "uint16",
            "chunk_grid": {
                "name": "regular",
                "configuration": { "chunk_shape": [2, 25, 25] }
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "-"}
            },
            "codecs": [{
                "name": "bytes",
                "configuration": {"endian": "little"}
            }],
            "fill_value": "0"
        }
        """.encode(),
    # No actual array data; data is never read in this test / by the multiscale logic
}


def test_create_scale_infos_from_zarr_node1():
    # Test getting scale info from a single array (compat mode)

    temp_store_data = store_data1.copy()

    store = MemoryStore(temp_store_data)
    g = open_zarr(store)

    infos = create_scale_infos_from_zarr_node(g)
    assert len(infos) == 1
    info = infos[0]
    assert isinstance(info, MultiscaleInfo)

    assert info.name == ""
    assert info.unit == ""
    assert info.unit_factor == 1

    assert len(info.scales) == 1
    si = info.scales[0]
    assert isinstance(si, ScaleInfo)

    assert si.level == 0
    assert si.mean_scale == 1
    assert isinstance(si.array, ZarrArray)
    assert si.spatial_shape == (100, 100)
    assert si.nchannels == 0
    assert si.ntimes == 0

    # Auto-channel detection, 3D array assumes 3D data
    g._shape = (100, 100, 100)
    g._chunk_grid_shape = 1, 1, 1

    infos = create_scale_infos_from_zarr_node(g)
    si = infos[0].scales[0]

    assert si.spatial_shape == (100, 100, 100)
    assert si.nchannels == 0
    assert si.ntimes == 0

    # Auto-channel detection, 4D array assumes 3D data + channels
    g._shape = (5, 100, 100, 100)
    g._chunk_grid_shape = 1, 1, 1, 1

    infos = create_scale_infos_from_zarr_node(g)
    si = infos[0].scales[0]

    assert si.spatial_shape == (100, 100, 100)
    assert si.nchannels == 5
    assert si.ntimes == 0

    # Auto-channel detection, 5D array assumes 3D data + channels + time
    g._shape = (5, 1, 100, 100, 100)
    g._chunk_grid_shape = 1, 1, 1, 1, 1

    infos = create_scale_infos_from_zarr_node(g)
    si = infos[0].scales[0]

    assert si.spatial_shape == (100, 100, 100)
    assert si.nchannels == 1
    assert si.ntimes == 5


def test_create_scale_infos_from_zarr_node2():
    # Test getting scale info from a multi-scale OME-Zarr file

    temp_store_data = store_data2.copy()

    store = MemoryStore(temp_store_data)
    g = open_zarr(store)

    infos = create_scale_infos_from_zarr_node(g)
    assert len(infos) == 1
    info = infos[0]
    assert isinstance(info, MultiscaleInfo)

    assert info.name == "a multiscale image"
    assert info.unit == "micrometer"
    assert info.unit_factor == 1e-6

    assert len(info.scales) == 3

    # scale 0

    si = info.scales[0]
    assert isinstance(si, ScaleInfo)

    assert si.level == 0
    assert si.mean_scale == 1
    assert isinstance(si.array, ZarrArray)
    assert si.spatial_shape == (100, 100)
    assert si.nchannels == 2
    assert si.ntimes == 0

    # scale 1

    si = info.scales[1]
    assert isinstance(si, ScaleInfo)

    assert si.level == 1
    assert si.mean_scale == 2
    assert isinstance(si.array, ZarrArray)
    assert si.spatial_shape == (50, 50)
    assert si.nchannels == 2
    assert si.ntimes == 0

    # scale 3

    si = info.scales[2]
    assert isinstance(si, ScaleInfo)

    assert si.level == 2
    assert si.mean_scale == 4
    assert isinstance(si.array, ZarrArray)
    assert si.spatial_shape == (25, 25)
    assert si.nchannels == 2
    assert si.ntimes == 0


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
