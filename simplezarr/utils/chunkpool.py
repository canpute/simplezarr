from __future__ import annotations


import asyncio
from itertools import count as Counter  # noqa: N812
from concurrent.futures import wait as concurrent_wait
from typing import Generator

import numpy as np
import simplezarr
from simplezarr.utils.multiscale import (
    create_scale_infos_from_zarr_node,
    MultiscaleInfo,
    ScaleInfo,
)
from simplezarr.utils.logs import log_exception

ref_counter = Counter()


def create_chunk_pools_from_zarr_node(
    zarr_node: simplezarr.ZarrNode,
) -> list[ChunkPool]:
    multiscale_infos = create_scale_infos_from_zarr_node(zarr_node)
    pools = []
    for multiscale_info in multiscale_infos:
        pool = ChunkPool(multiscale_info)
        pools.append(pool)

    return pools


class ChunkPool:
    """A pool of chunks, aware of multiscale (ome-zarr), and multi-channel."""

    def __init__(self, multiscale_info: MultiscaleInfo):
        self._multiscale_info = multiscale_info

        # TODO: allow chunks to exist for longer, based on a max memory pool

        self._chunks = []  # level -> chunk_index -> ChunkSpot
        for _ in range(len(self._multiscale_info.scales)):
            self._chunks.append({})

    def __del__(self):
        return self.destroy()

    @property
    def multiscale_info(self) -> MultiscaleInfo:
        """Get the object that represent the information on the multiscale image."""
        return self._multiscale_info

    def destroy(self):
        """Clear all chunks."""
        raise NotImplementedError()

    def get_chunk(
        self, level: int, chunk_index: tuple[int, ...], ref: str
    ) -> ChunkSpot:
        """Get a ChunkSpot object."""

        chunk_spot = self._chunks[level].get(chunk_index, None)
        if chunk_spot is None:
            chunk_info = self._multiscale_info.scales[level]
            chunk_spot = ChunkSpot(chunk_info, chunk_index)
            self._chunks[level][chunk_index] = chunk_spot
        chunk_spot._add_ref(ref)
        return chunk_spot

    def drop_chunk(self, level: int, chunk_index: tuple[int, ...], ref: str) -> None:
        """Tell the pool that you're done using the chunk spot at the given location."""
        chunk_spot = self._chunks[level].get(chunk_index, None)
        if chunk_spot is not None:
            has_refs = chunk_spot._drop_ref(ref)
            if not has_refs:
                chunk_spot.destroy()
                self._chunks[level].pop(chunk_index, None)

    def iter_chunks(self) -> Generator[ChunkSpot]:
        for chunks in self._chunks:
            for chunk in chunks.values():
                yield chunk


class ChunkManager:
    """Object that manages chunks for a certain purpose, using a shared ChunkPool.

    Chunks can be loaded in parallel:

        with manager:
            chunk1 = manager.get_chunk(level, index1)
            chunk2 = manager.get_chunk(level, index2)

        # After the with-statement, the code will wait for the data of all chunks to be loaded

    """

    def __init__(self, pool):
        # Each ChunkManager has a unique ref, so that the pool can keep track how many 'users' each chunk has, and drop the chunk when there are none left
        self._ref = f"{self.__class__.__name__} {next(ref_counter)}"
        self._pool = pool
        self._unloaded_chunk_spots = None

        # We also keep track of what chunks we have
        self._chunks = []  # level -> chunk_index -> ChunkSpot
        for _ in range(len(self._pool.multiscale_info.scales)):
            self._chunks.append({})

    def __enter__(self):
        assert self._unloaded_chunk_spots is None
        self._unloaded_chunk_spots = []
        return self

    def __exit__(self, type, value, traceback):
        unloaded_chunk_spots = self._unloaded_chunk_spots
        self._unloaded_chunk_spots = None
        if unloaded_chunk_spots:
            concurrent_wait([chunk_spot._future for chunk_spot in unloaded_chunk_spots])
            for chunk_spot in unloaded_chunk_spots:
                self._call_on_chunk_load(chunk_spot)
        return None
        # TODO: should the manager drop all chunks here? If so, add manager.load() to sync-load all requested chunks so far.

    @property
    def ref(self) -> str:
        """The unique reference of this chunk manager.

        References are used by the pool to keep track of chunk usage.
        """
        return self._ref

    @property
    def pool(self) -> ChunkPool:
        """The ChunkPool used by this chunk manager.

        Multiple chunk managers can use the same pool.
        """
        return self._pool

    def request_chunk_sync(self, level, chunk_index):
        """Request a chunk in a synchronous manner.

        Must be called by using the manager in a ``switch`` statement. All
        requested chunks within that statement are then loaded in parallel.
        """
        chunk_spot = self._chunks[level].get(chunk_index)
        if chunk_spot is not None:
            return chunk_spot

        assert self._unloaded_chunk_spots is not None
        chunk_spot = self._pool.get_chunk(level, chunk_index, self._ref)
        self._chunks[level][chunk_index] = chunk_spot

        self._unloaded_chunk_spots.append(chunk_spot)
        return chunk_spot

    def request_chunk_async(self, level, chunk_index) -> asyncio.Task:
        """Request a chunk asynchronously. Assumes a running asyncio loop.

        Loads the chunk (if needed) and calls ``_on_chunk_load()``.
        Can be used as fire-and-forget, but also returns a future than can be awaited.
        """
        loop = asyncio.get_running_loop()

        chunk_spot = self._chunks[level].get(chunk_index)
        if chunk_spot is not None:
            return loop.create_task(asyncio.sleep(0))

        chunk_spot = self._pool.get_chunk(level, chunk_index, self._ref)
        self._chunks[level][chunk_index] = chunk_spot

        async def request_chunk_async_internal(chunk_spot):
            await asyncio.wrap_future(chunk_spot._future)
            self._call_on_chunk_load(chunk_spot)

        return loop.create_task(request_chunk_async_internal(chunk_spot))

    def drop(self, chunk_spot: ChunkSpot):
        """Drop the given chunk."""
        self.drop_chunk(chunk_spot.level, chunk_spot.index)

    def drop_chunk(self, level, chunk_index):
        """Drop a chunk so it can be unloaded."""
        self._pool.drop_chunk(level, chunk_index, self._ref)
        chunk_spot = self._chunks[level].pop(chunk_index, None)
        if chunk_spot is not None:
            self._call_on_chunk_drop(chunk_spot)

    def _call_on_chunk_load(self, chunk_spot):
        with log_exception("on_chunk_load"):
            self.on_chunk_load(chunk_spot)

    def on_chunk_load(self, chunk_spot):
        """Overload this in a subclass to automatically perform an action when a chunk is loaded."""
        pass

    def _call_on_chunk_drop(self, chunk_spot):
        with log_exception("on_chunk_drop"):
            self.on_chunk_drop(chunk_spot)

    def on_chunk_drop(self, chunk_spot):
        """Overload this in a subclass to automatically perform an action when a chunk is dropped."""
        pass


class ChunkSpot:
    """An Object that represents a chunk location."""

    def __init__(self, scale_info: ScaleInfo, index: tuple[int, ...]):
        self._scale_info = scale_info
        self._index = index

        self._refs = set()
        self._data = None
        self._future = scale_info.array.get_chunk_future(index)

    def destroy(self):
        self._future = None
        self._data = None

    @property
    def scale_info(self) -> ScaleInfo:
        """The info for the scale that this chunk is part of."""
        return self._scale_info

    @property
    def level(self) -> int:
        return self._scale_info.level

    @property
    def index(self) -> tuple[int, ...]:
        return self._index

    @property
    def data(self) -> np.ndarray:
        """The data (numpy array) for this chunk.

        When this property is accessed before the data is loaded, a RuntimeError is raised.

        """
        if self._data is None:
            if not self._future.done():
                raise RuntimeError(
                    "Cannot access ``chunk_spot.data`` when the data is not yet loaded."
                )
            self._data = self._future.result()
        return self._data

    @property
    def refs(self) -> set[str]:
        """A set of references that currently use this chunk."""
        return set(self._refs)

    def _add_ref(self, ref: str):
        # Called by the ChunkPool
        assert isinstance(ref, str)
        self._refs.add(ref)

    def _drop_ref(self, ref: str) -> bool:
        # Called by the ChunkPool
        assert isinstance(ref, str)
        self._refs.discard(ref)
        return len(self._refs) > 0  # does it still have refs?
