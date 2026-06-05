"""
Read speed benchmark.

Note that the times depend a bit on the order in which they are run.
Regardless, the trend is consistent: simplezarr is 10-20% faster than
zarr-python, and paralel reads yields an approximate 5x speedup (local
filesystem).
"""

import time
import shutil
from pathlib import Path
from concurrent.futures import wait as wait_futures

import zarr
import numcodecs
import simplezarr
import numpy as np


def read_chunks_without_lib():
    for c in range(grid_shape[0]):
        for z in range(grid_shape[1]):
            # print(f"z: {z}")
            for y in range(grid_shape[2]):
                for x in range(grid_shape[3]):
                    # Mimic array1.blocks[c, z, y, x]
                    fname = str(filename) + f"/c/{c}/{z}/{y}/{x}"
                    try:
                        bytes1 = open(fname, "rb").read()
                        bytes2 = numcodecs.zstd.decompress(bytes1)
                        assert len(bytes2) > 100000
                    except FileNotFoundError:
                        pass


def read_chunks_with_simplezarr():
    arr = simplezarr.open_zarr(simplezarr.LocalStore(filename))

    for c in range(grid_shape[0]):
        for z in range(grid_shape[1]):
            # print(f"z: {z}")
            for y in range(grid_shape[2]):
                for x in range(grid_shape[3]):
                    chunk = arr.get_chunk_now((c, z, y, x))
                    assert chunk.shape == arr.chunk_shape


def read_chunks_with_simplezarr_parallel():
    arr = simplezarr.open_zarr(simplezarr.LocalStore(filename))
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
                                f = arr.get_chunk_soon((c, z + zz, y + yy, x + xx))
                                futures.append(f)
                    wait_futures(futures)
                    for f in futures:
                        chunk = f.result()
                        assert chunk.shape == arr.chunk_shape


def read_chunks_zarrpy():
    arr = zarr.open_array(store=filename, mode="r")

    for c in range(grid_shape[0]):
        for z in range(grid_shape[1]):
            # print(f"z: {z}")
            for y in range(grid_shape[2]):
                for x in range(grid_shape[3]):
                    chunk = arr.blocks[c, z, y, x]
                    assert isinstance(chunk, np.ndarray)
                    # assert chunk.shape == a.chunk_shape


# Create a Zarr file

filename = Path(__file__).absolute().parent / "benchmark_data"
shutil.rmtree(filename)
filename.mkdir(exist_ok=True)

store = simplezarr.LocalStore(filename)

arr = simplezarr.ZarrArray.create(
    store, "", (2, 1000, 1000, 1000), "uint16", chunk_shape=(2, 128, 128, 128)
)
grid_shape = arr.chunk_grid_shape
arr[...].set_now(7)

del arr


# Run functions ...

for func in [
    read_chunks_without_lib,
    read_chunks_without_lib,
    read_chunks_zarrpy,
    read_chunks_zarrpy,
    read_chunks_with_simplezarr,
    read_chunks_with_simplezarr,
    read_chunks_with_simplezarr_parallel,
    read_chunks_with_simplezarr_parallel,
]:
    t0 = time.perf_counter()
    func()
    t1 = time.perf_counter()
    time.sleep(0.1)
    print(f"{func.__name__}: {t1 - t0:0.3f} s")
