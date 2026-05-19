"""
The ChunkPool object keeps track of the individual chunks. Freeing the chunks when no longer used.
of chunks, and free chunks when no longer needed.
"""

from __future__ import annotations

import asyncio
from itertools import count as Counter  # noqa: N812
from typing import Generator

import numpy as np
import simplezarr
from simplezarr.utils.logs import log_exception
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

    def __init__(self, multiscale_info: MultiscaleInfo, call_soon_threadsafe=None):
        self._multiscale_info = multiscale_info
        self._call_soon_threadsafe = call_soon_threadsafe
        self._chunks = []  # level -> chunk_index -> ChunkLocation
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
        for chunk_dict in self._chunks:
            chunks = list(chunk_dict.values())
            chunk_dict.clear()
            for chunk in chunks:
                chunk._destroy()

    def get_chunk(
        self,
        level: int,
        chunk_index: tuple[int, ...],
        ref: str,
        *,
        load_handler=None,
        drop_handler=None,
        destroy_handler=None,
    ) -> ChunkLocation:
        """Get a ChunkLocation object.

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

            def load_handler(chunk_spot):
               ...  # this func can also be async

            pool.get_chunk(..., load_handler=load_handler)

        """
        if not (isinstance(ref, str) and len(ref) > 0):
            raise TypeError("get_chunk() ref must be a nonempty string.")

        chunk_spot = self._chunks[level].get(chunk_index, None)
        if chunk_spot is None:
            chunk_info = self._multiscale_info.scales[level]
            chunk_spot = ChunkLocation(chunk_info, chunk_index)
            self._chunks[level][chunk_index] = chunk_spot

        call_soon_threadsafe = self._call_soon_threadsafe
        if call_soon_threadsafe is None and load_handler is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                call_soon_threadsafe = loop.call_soon_threadsafe

        chunk_spot._register(
            ref, call_soon_threadsafe, load_handler, drop_handler, destroy_handler
        )

        return chunk_spot

    def drop_chunk(self, level: int, chunk_index: tuple[int, ...], ref: str) -> None:
        """Release a chunk by their index.

        It is important to use the same unique ``ref`` as when ``get_chunk()`` was called. That way
        the pool can properly detect when no-one is using a chunk anymore, so it can be marked for deletion.
        """
        chunk_spot = self._chunks[level].get(chunk_index, None)
        if chunk_spot is not None:
            chunk_spot._drop(ref)
            has_refs = len(chunk_spot.refs) > 0
            if not has_refs:
                chunk_spot._destroy()
                self._chunks[level].pop(chunk_index, None)

    def iter_chunks(self) -> Generator[ChunkLocation]:
        """Iterate over all currently loaded chunks."""
        for chunks in self._chunks:
            for chunk in chunks.values():
                yield chunk

    def wait_for_chunks_to_load(self):
        """Wait for all requested chunks to load their data."""
        for chunk_spot in self.iter_chunks():
            chunk_spot.wait()


class ChunkLocation:
    """An Object that represents a chunk location."""

    def __init__(self, scale_info: ScaleInfo, index: tuple[int, ...]):
        self._scale_info = scale_info
        self._index = index

        self._refs = set()
        self._data = None
        self._future = scale_info.array.get_chunk_future(index)
        print("future", self._future.running(), self._future.done())
        self._load_handlers = {}
        self._drop_handlers = {}
        self._destroy_handlers = {}

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
        print("future", self._future.running(), self._future.done())
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
        self._process_load_handlers()

    def _register(
        self,
        ref: str,
        call_soon_threadsafe=None,
        on_load=None,
        on_drop=None,
        on_destroy=None,
    ):
        self._refs.add(ref)

        if call_soon_threadsafe is not None:
            self._future.add_done_callback(
                lambda f: call_soon_threadsafe(self._process_load_handlers)
            )

        if on_load is not None:
            self._load_handlers.setdefault(ref, []).append(on_load)
            if self._future.done():
                self._process_load_handlers()

        if on_drop is not None:
            self._drop_handlers.setdefault(ref, []).append(on_drop)

        if on_destroy is not None:
            self._destroy_handlers.setdefault(ref, []).append(on_destroy)

    def _invoke_handlers(self, what, *handlers):
        for func in handlers:
            with log_exception(what):
                func(self)

    def _process_load_handlers(self):
        for ref in list(self._load_handlers.keys()):
            handlers = self._load_handlers.pop(ref, [])
            self._invoke_handlers(
                f"ChunkLocation load callback for ref {ref!r}", *handlers
            )

    def _drop(self, ref: str):
        self._refs.discard(ref)
        self._load_handlers.pop(ref, None)
        handlers = self._drop_handlers.pop(ref, [])
        self._invoke_handlers(f"ChunkLocation drop callback for ref {ref!r}", *handlers)

    def _destroy(self):
        self._refs.clear()
        self._load_handlers = {}
        for ref in list(self._drop_handlers.keys()):
            handlers = self._drop_handlers.pop(ref, [])
            self._invoke_handlers(
                f"ChunkLocation drop callback for ref {ref!r}", *handlers
            )
        for ref in list(self._destroy_handlers.keys()):
            handlers = self._destroy_handlers.pop(ref, [])
            self._invoke_handlers(
                f"ChunkLocation destroy callback for ref {ref!r}", *handlers
            )
        self._future = None
        self._data = None
