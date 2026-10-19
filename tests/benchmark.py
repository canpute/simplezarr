"""
Read speed benchmark.

Note that the times depend a bit on the order in which they are run.
Regardless, the trend is consistent: simplezarr is 10-20% faster than
zarr-python, and paralel reads yields an approximate 5x speedup (local
filesystem).
"""

import shutil
import time
from concurrent.futures import wait as wait_futures
from pathlib import Path

import numcodecs
import numpy as np
import zarr

import simplezarr


def read_chunks_without_lib():
    for c in range(grid_shape[0]):
        for z in range(grid_shape[1]):
            # print(f"z: {z}")
            for y in range(grid_shape[2]):
                for x in range(grid_shape[3]):
                    # Mimic array1.blocks[c, z, y, x]
                    fname = str(filename) + f"/image/c/{c}/{z}/{y}/{x}"
                    try:
                        bytes1 = open(fname, "rb").read()
                        bytes2 = numcodecs.zstd.decompress(bytes1)
                        assert len(bytes2) > 100000
                    except FileNotFoundError:
                        pass


def read_chunks_with_simplezarr():
    arr = simplezarr.open_zarr(simplezarr.LocalStore(filename), "image")

    for c in range(grid_shape[0]):
        for z in range(grid_shape[1]):
            # print(f"z: {z}")
            for y in range(grid_shape[2]):
                for x in range(grid_shape[3]):
                    chunk = arr.get_chunk_now((c, z, y, x))
                    assert chunk.shape == arr.chunk_shape


def read_chunks_with_simplezarr_parallel():
    arr = simplezarr.open_zarr(simplezarr.LocalStore(filename), "image")
    step = 3

    for c in range(grid_shape[0]):
        for z in range(0, grid_shape[1], step):
            # print(f"z: {z}")
            for y in range(0, grid_shape[2], step):
                for x in range(0, grid_shape[3], step):
                    futures = []
                    for zz in range(step):
                        for yy in range(step):
                            for xx in range(step):
                                if (
                                    z + zz < grid_shape[1]
                                    and y + yy < grid_shape[2]
                                    and x + xx < grid_shape[3]
                                ):
                                    f = arr.get_chunk_soon((c, z + zz, y + yy, x + xx))
                                    futures.append(f)
                    wait_futures(futures)
                    for f in futures:
                        chunk = f.result()
                        assert chunk.shape == arr.chunk_shape


def read_chunks_zarrpy():
    arr = zarr.open_array(store=filename, path="image", mode="r")

    for c in range(grid_shape[0]):
        for z in range(grid_shape[1]):
            # print(f"z: {z}")
            for y in range(grid_shape[2]):
                for x in range(grid_shape[3]):
                    chunk = arr.blocks[c, z, y, x]
                    assert isinstance(chunk, np.ndarray)
                    # assert chunk.shape == a.chunk_shape


def write_pyramid_with_simplezarr():
    from simplezarr.utils.write_multiscale import write_ome_zarr_pyramid

    store = simplezarr.LocalStore(filename)
    src_arr = simplezarr.open_zarr(store, "image")

    _g = write_ome_zarr_pyramid(
        store,
        "pyramid.ome.zarr",
        src_arr,
        "czyx",
        space_unit="micrometer",
        chunk_shape=(2, 32, 32, 32),
        chunk_path_separator=".",
        max_memory=2**30,
    )


def write_pyramid_with_ome_zarr_lib():
    from ome_zarr.writer import write_image

    scale_factors = [
        {"z": 2, "y": 2, "x": 2},
        {"z": 4, "y": 4, "x": 4},
        {"z": 8, "y": 8, "x": 8},
        {"z": 16, "y": 16, "x": 16},
    ]

    src_arr = zarr.open_array(store=filename, path="image", mode="r")

    write_image(
        src_arr,
        str(store.path / "pyramid2"),
        scale_factors=scale_factors,
        axes="czyx",
        method="local_mean",
        storage_options={"chunks": 32},
    )


# Create a Zarr file

filename = Path(__file__).absolute().parent / "benchmark_data"
shutil.rmtree(filename)
filename.mkdir(exist_ok=True)

store = simplezarr.LocalStore(filename)

arr = simplezarr.ZarrArray.create(
    store, "image", (2, 1000, 1000, 100), "uint16", chunk_shape=(2, 128, 128, 128)
)
grid_shape = arr.chunk_grid_shape
arr[...].set_now(7)

del arr


# Run functions ...
print("Done prepping, let's run!")
for func in [
    read_chunks_without_lib,
    read_chunks_without_lib,
    read_chunks_zarrpy,
    read_chunks_zarrpy,
    read_chunks_with_simplezarr,
    read_chunks_with_simplezarr,
    read_chunks_with_simplezarr_parallel,
    read_chunks_with_simplezarr_parallel,
    write_pyramid_with_simplezarr,
    write_pyramid_with_simplezarr,
    write_pyramid_with_ome_zarr_lib,  # once, bc it checks for overwrites
]:
    t0 = time.perf_counter()
    func()
    t1 = time.perf_counter()
    time.sleep(0.1)
    print(f"{func.__name__}: {t1 - t0:0.3f} s")
