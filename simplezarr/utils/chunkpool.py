"""
The ChunkPool object keeps track of the individual chunks. Freeing the chunks when no longer used.
of chunks, and free chunks when no longer needed.
"""

from __future__ import annotations

import inspect
import asyncio
from itertools import count as Counter  # noqa: N812
from typing import Generator

import numpy as np
import simplezarr
from simplezarr.utils.multiscale import (
    create_scale_infos_from_zarr_node,
    MultiscaleInfo,
    ScaleInfo,
)

ref_counter = Counter()


def create_chunk_pools_from_zarr_node(
    zarr_node: simplezarr.ZarrNode,
) -> list[ChunkPool]:
    """Create a ``ChuckPool`` for every (multiscale) image in the given Zarr node."""
    multiscale_infos = create_scale_infos_from_zarr_node(zarr_node)
    pools = []
    for multiscale_info in multiscale_infos:
        pool = ChunkPool(multiscale_info)
        pools.append(pool)
    return pools


class ChunkPool:
    """An object to get access to individual chunks, with support for caching and parallel loading."""

    def __init__(self, multiscale_info: MultiscaleInfo):
        self._multiscale_info = multiscale_info
        self._chunks = []  # level -> chunk_index -> ChunkSpot
        for _ in range(len(self._multiscale_info.scales)):
            self._chunks.append({})

    def __del__(self):
        return self.destroy()

    @property
    def multiscale_info(self) -> MultiscaleInfo:
        """Get the object that represents the information on the multiscale image."""
        return self._multiscale_info

    def destroy(self):
        """Clear all chunks."""
        # TODO: implement!
        raise NotImplementedError()

    def get_chunk(
        self, level: int, chunk_index: tuple[int, ...], ref: str
    ) -> ChunkSpot:
        """Get a ChunkSpot object.

        The returned object represents the requested chunk. The ``ref`` should
        be a unique string indicating the 'user'. When a chunk is requested, the
        pool caches the chunk until it is dropped (using the same ``ref``). That
        way, other code that uses the same pool, requesting chunks that are
        already loaded, can share the chunks.

        The corresponding data is being loaded but may not be ready yet. One can
        either sync-wait for it, async-wait for multiple chunks in parallel, or
        schedule an async task to happen when the chunk has loaded.

        Individual chunks can be loaded synchronously using::

            chunk_spot.wait()
            data = chunk_spot.data

        After getting multiple chunks, it's easy to load them in parallel::

            loop.wait_for_chunks_to_load()

        This is equivalent to:

            chunk_spots = [...] for chunk_spot in chunk_spots:
            for chunk_spot in chunk_spots:
                chunk_spot.wait()

        In applications using asyncio, you can asynchronously process chunks as
        soon as they are loaded::

            def chunk_handler(chunk_spot):
               ...  # this func can also be async

            chunk_spot.add_async_handler(chunk_handler)

        """
        if not (isinstance(ref, str) and len(ref) > 0):
            raise TypeError("get_chunk() ref must be a nonempty string.")

        chunk_spot = self._chunks[level].get(chunk_index, None)
        if chunk_spot is None:
            chunk_info = self._multiscale_info.scales[level]
            chunk_spot = ChunkSpot(chunk_info, chunk_index)
            self._chunks[level][chunk_index] = chunk_spot
        chunk_spot._refs.add(ref)

        return chunk_spot

    def drop_chunk(self, level: int, chunk_index: tuple[int, ...], ref: str) -> None:
        """Release a chunk by their index.

        It is important to use the same unique ``ref`` as when ``get_chunk()`` was called. That way
        the pool can properly detect when no-one is using a chunk anymore, so it can be marked for deletion.
        """
        chunk_spot = self._chunks[level].get(chunk_index, None)
        if chunk_spot is not None:
            chunk_spot._refs.discard(ref)
            has_refs = len(chunk_spot._refs) > 0
            if not has_refs:
                chunk_spot.destroy()
                self._chunks[level].pop(chunk_index, None)

    def iter_chunks(self) -> Generator[ChunkSpot]:
        """Iterate over all currently loaded chunks."""
        for chunks in self._chunks:
            for chunk in chunks.values():
                yield chunk

    def wait_for_chunks_to_load(self):
        """Wait for all requested chunks to load their data."""
        for chunk_spot in self.iter_chunks:
            chunk_spot.wait()


class ChunkSpot:
    """An Object that represents a chunk location."""

    def __init__(self, scale_info: ScaleInfo, index: tuple[int, ...]):
        self._scale_info = scale_info
        self._index = index

        self._refs = set()  # managed by the ChunkPool
        self._data = None
        self._future = scale_info.array.get_chunk_future(index)

    @property
    def scale_info(self) -> ScaleInfo:
        """The info for the scale that this chunk is part of."""
        return self._scale_info

    @property
    def level(self) -> int:
        """The integer level that this chunk belongs to."""
        return self._scale_info.level

    @property
    def index(self) -> tuple[int, ...]:
        """The index of this chunk."""
        return self._index

    @property
    def future(self):
        """The ``concurrent.futures.Future`` for loading the data."""
        return self._future

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

    def wait(self):
        """Synchronously wait for the chunk's data to load."""
        self._future.result()

    def add_async_handler(self, func_to_process_chunk) -> asyncio.Task:
        """Add a a handler that gets called when the chunk is done loading.

        This creates a new task to the currently running asyncio loop, which waits for the
        chunks data to be loaded, and then calls ``func_to_process_chunk``. That function
        can be either a plain function or a coroutine.

        Returns the new asyncio task.
        """

        async def add_async_handler_wrapper(chunk_spot):
            await asyncio.wrap_future(self._future)
            x = func_to_process_chunk(chunk_spot)
            if inspect.iscoroutine(x):
                await x

        loop = asyncio.get_running_loop()
        return loop.create_task(add_async_handler_wrapper(self))

    def destroy(self):
        """Destroy this chunkspot, clearing all data. This is automatically called by the pool when no-one uses the chunk anymore."""
        self._future = None
        self._data = None
