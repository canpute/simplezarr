import os
import time
import asyncio

from simplezarr import MemoryStore, SlowStore, open_zarr
from simplezarr.utils.chunkpool import (
    ChunkPool,
    ChunkLocation,
    ChunkManager,
    create_chunk_pools_from_zarr_node,
)

import numpy as np
import pytest


# Create an in-memory zarr file

store_data = {
    "zarr.json": """
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [100, 100],
            "dimension_names": ["y", "x"],
            "data_type": "float32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": [50, 50]
                }
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {
                    "separator": "/"
                }
            },
            "codecs": [{
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            }],
            "fill_value": "0.0",
            "attributes": {
                "foo": true,
                "bar": "apples"
            }
        }
        """.encode(),
    "c/0/0": np.full((50, 50), 1.0, np.float32).tobytes(),
    "c/0/1": np.full((50, 50), 2.0, np.float32).tobytes(),
    "c/1/0": np.full((50, 50), 3.0, np.float32).tobytes(),
    "c/1/1": np.full((50, 50), 4.0, np.float32).tobytes(),
}


store = SlowStore(MemoryStore(store_data), base_delay=0.25)


def test_chunk_pool_simple():

    g = open_zarr(store)

    # Create a pool from the fake data
    pools = create_chunk_pools_from_zarr_node(g)
    assert len(pools) == 1
    pool = pools[0]
    assert isinstance(pool, ChunkPool)
    assert len(pool.multiscale_info.scales) == 1

    # Create some chunks
    chunk1a = pool.get_chunk(0, (0, 0))
    chunk1b = pool.get_chunk(0, (0, 0))
    chunk2 = pool.get_chunk(0, (0, 1))
    chunk3 = pool.get_chunk(0, (1, 0))

    assert chunk1a is chunk1b
    assert chunk1a is not chunk2
    assert chunk1a is not chunk3

    # Check some basics
    assert isinstance(chunk1a, ChunkLocation)
    assert chunk1a.scale_info.spatial_chunk_shape == (50, 50)
    assert chunk1a.level == 0
    assert chunk1a.index == (0, 0)
    assert chunk2.index == (0, 1)
    assert chunk3.index == (1, 0)

    # Need to wait for data, using SlowStore to test this reliably
    with pytest.raises(RuntimeError):
        chunk1a.data  # noqa

    # Wait for one particular chunk to load
    chunk1a.wait()

    # Wait for all requested chunks to load
    pool.wait_for_chunks_to_load()

    # Can now load
    assert isinstance(chunk1a.data, np.ndarray)
    assert chunk1a.data.max() == 1
    assert chunk2.data.max() == 2
    assert chunk3.data.max() == 3

    # Check state
    assert len(list(pool.iter_chunks())) == 3

    # Drop a chunk
    pool.drop_chunk(0, (0, 0))
    assert len(list(pool.iter_chunks())) == 2

    # Destroy the pool, which auto-drops all remaining chunks
    # Note that pool.destroy() is called when the object is deleted by gc
    pool.destroy()
    assert len(list(pool.iter_chunks())) == 0


def test_chunk_pool_multiuser():

    # Create a pool
    g = open_zarr(store)
    pools = create_chunk_pools_from_zarr_node(g)
    pool = pools[0]

    # Create some chunks
    chunk1a = pool.get_chunk(0, (0, 0), "r1a")
    chunk1b = pool.get_chunk(0, (0, 0), "r1a")  # same ref
    chunk1c = pool.get_chunk(0, (0, 0), "r1c")
    chunk2 = pool.get_chunk(0, (0, 1), "r2")
    chunk3 = pool.get_chunk(0, (1, 0), "r3")

    assert chunk1a is chunk1b
    assert chunk1a is chunk1c
    assert chunk1a is not chunk2
    assert chunk1a is not chunk3

    # Wait for all requested chunks to load
    pool.wait_for_chunks_to_load()

    # Can now load
    assert isinstance(chunk1a.data, np.ndarray)
    assert chunk1a.data.max() == 1
    assert chunk2.data.max() == 2
    assert chunk3.data.max() == 3

    # Check the multi-use strategy
    assert len(list(pool.iter_chunks())) == 3
    assert len(chunk1a.refs) == 2
    assert len(chunk2.refs) == 1
    assert len(chunk3.refs) == 1

    # Drop a chunk
    pool.drop_chunk(0, (0, 0), "r1a")

    assert len(list(pool.iter_chunks())) == 3
    assert len(chunk1a.refs) == 1
    assert len(chunk2.refs) == 1
    assert len(chunk3.refs) == 1

    # Drop same chunk with last ref
    pool.drop_chunk(0, (0, 0), "r1c")

    assert len(list(pool.iter_chunks())) == 2
    assert len(chunk1a.refs) == 0
    assert len(chunk2.refs) == 1
    assert len(chunk3.refs) == 1

    # Drop chunk that only had one ref to begin with
    pool.drop_chunk(0, (0, 1), "r2")

    assert len(list(pool.iter_chunks())) == 1
    assert len(chunk1a.refs) == 0
    assert len(chunk2.refs) == 0
    assert len(chunk3.refs) == 1

    # Destroy the pool, which auto-drops all remaining chunks
    # Note that pool.destroy() is called when the object is deleted by gc
    pool.destroy()
    assert len(list(pool.iter_chunks())) == 0


def test_chunk_pool_handlers():

    # Get pool
    pools = create_chunk_pools_from_zarr_node(open_zarr(store))
    pool = pools[0]

    events = []

    def on_load1(chunk_location):
        events.append(f"load {chunk_location.index} 1")

    def on_load2(chunk_location):
        events.append(f"load {chunk_location.index} 2")

    def on_drop(chunk_location):
        events.append(f"drop {chunk_location.index}")

    def on_destroy(chunk_location):
        events.append(f"destroy {chunk_location.index}")

    # Request some chunks
    _chunk1a = pool.get_chunk(
        0,
        (0, 0),
        "r1a",
        load_handler=on_load1,
        drop_handler=on_drop,
        destroy_handler=on_destroy,
    )
    _chunk1b = pool.get_chunk(
        0,
        (0, 0),
        "r1b",
        load_handler=on_load1,
        drop_handler=on_drop,
        destroy_handler=on_destroy,
    )

    _chunk2 = pool.get_chunk(
        0,
        (0, 1),
        "r2",
        load_handler=on_load1,
        drop_handler=on_drop,
        destroy_handler=on_destroy,
    )
    pool.get_chunk(0, (0, 1), "r2", load_handler=on_load2)

    # Load them
    pool.wait_for_chunks_to_load()

    assert events == [
        "load (0, 0) 1",
        "load (0, 0) 1",
        "load (0, 1) 1",
        "load (0, 1) 2",
    ]
    events.clear()

    # Drop them

    pool.drop_chunk(0, (0, 0), "r1a")
    assert events == ["drop (0, 0)"]

    pool.drop_chunk(0, (0, 0), "r1a")  # nothing
    assert events == ["drop (0, 0)"]

    pool.drop_chunk(0, (0, 0), "r1b")
    assert events == ["drop (0, 0)", "drop (0, 0)", "destroy (0, 0)", "destroy (0, 0)"]
    events.clear()

    # Destroy drops last one
    pool.destroy()
    assert events == ["drop (0, 1)", "destroy (0, 1)"]


def test_chunk_pool_handlers_async_load():
    # Test the ChunkPool.enable_async_load_handlers util

    sleep_time = 0.5

    pools = create_chunk_pools_from_zarr_node(open_zarr(store))
    pool = pools[0]

    events = []

    def on_load(chunk_location):
        events.append(f"load {chunk_location.index}")

    # No integration

    _chunk = pool.get_chunk(0, (0, 0), "r1", load_handler=on_load)
    assert events == []
    time.sleep(sleep_time)
    assert events == []

    # Reset
    pool.destroy()
    events.clear()
    pool.enable_async_load_handlers("none")

    # Raw integration (this test is thread-safe, kinda, because we sleep)

    pool.enable_async_load_handlers(lambda f, *args: f(*args))
    _chunk = pool.get_chunk(0, (0, 0), "r1", load_handler=on_load)
    assert events == []
    time.sleep(sleep_time)
    assert events == ["load (0, 0)"]

    # Reset
    pool.destroy()
    events.clear()
    pool.enable_async_load_handlers("none")

    # Skip the rest if running in an asyncio interactive IDE
    if asyncio._get_running_loop() and not os.getenv("CI"):
        return

    # Asyncio, but not enabled

    async def main1():
        _chunk = pool.get_chunk(0, (0, 0), "r1", load_handler=on_load)
        assert events == []
        await asyncio.sleep(sleep_time)

    assert events == []
    asyncio.run(main1())
    assert events == []

    # Reset
    pool.destroy()
    events.clear()
    pool.enable_async_load_handlers("none")

    # Asyncio, iwth async enables

    async def main2():
        pool.enable_async_load_handlers("asyncio")
        await main1()

    assert events == []
    asyncio.run(main2())
    assert events == ["load (0, 0)"]

    # Reset
    pool.destroy()
    events.clear()
    pool.enable_async_load_handlers("none")

    # Can only enable when in a running loop

    with pytest.raises(RuntimeError):
        pool.enable_async_load_handlers("asyncio")


def test_chunk_manager():

    pools = create_chunk_pools_from_zarr_node(open_zarr(store))
    pool = pools[0]

    events = []

    class MyManager(ChunkManager):
        def on_load(self, chunk_location):
            events.append(f"load {chunk_location.index}")

        def on_drop(self, chunk_location):
            events.append(f"drop {chunk_location.index}")

        def on_destroy(self, chunk_location):
            events.append(f"destroy {chunk_location.index}")

    manager1 = MyManager(pool)
    manager2 = MyManager(pool)

    chunk1 = manager1.get_chunk(0, (0, 0))
    chunk2 = manager2.get_chunk(0, (0, 0))

    assert chunk1 is chunk2
    assert len(chunk1.refs) == 2

    pool.wait_for_chunks_to_load()
    assert events == ["load (0, 0)", "load (0, 0)"]
    events.clear()

    # Drop one
    manager1.drop_chunk(0, (0, 0))
    assert events == ["drop (0, 0)"]
    events.clear()

    # Drop other
    manager2.drop_chunk(0, (0, 0))
    assert events == ["drop (0, 0)", "destroy (0, 0)", "destroy (0, 0)"]
    events.clear()

    assert len(list(pool.iter_chunks())) == 0


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
