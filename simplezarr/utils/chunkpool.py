"""
The ChunkPool object keeps track of the individual chunks. Freeing the chunks when no longer used.
of chunks, and free chunks when no longer needed.
"""

from __future__ import annotations

from typing import Generator, Callable, Literal
from itertools import count as Count

import numpy as np
import simplezarr
from simplezarr.utils.logs import log_exception
from simplezarr.utils.multiscale import (
    create_scale_infos_from_zarr_node,
    MultiscaleInfo,
    ScaleInfo,
)


__all__ = ["ChunkPool", "ChunkManager", "ChunkLocation"]


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
        self._call_soon_threadsafe = None
        self._chunks = []  # level -> chunk_index -> ChunkLocation
        for _ in range(len(self._multiscale_info.scales)):
            self._chunks.append({})

    def __del__(self):
        return self.destroy()

    @property
    def multiscale_info(self) -> MultiscaleInfo:
        """The ``MultiscaleInfo`` object that represents the information on the multiscale image."""
        return self._multiscale_info

    def enable_async_load_handlers(
        self, call_soon_threadsafe: Callable | Literal["asyncio", "none"]
    ):
        """Set the pool up to asynchronously call the load-handlers.

        In normal operation, the load-handlers only get invoked when the
        ChunkLocation objects are waited upon, either by
        ``chunk_location.wait()`` or ``chunk_pool.wait_for_chunks_to_load()``.
        With async enabled, the load-handlers are fired as soon as the data is
        loaded. This behaviour is especially intended for interactive
        applications such as data viewers.

        The async behaviour does not depend on asyncio, but can be used with any
        framework that can provide a ``call_soon_threadsafe()`` function. That
        said, asyncio is the most common use-case, so one can simply do
        ``enable_async_load_handlers("asyncio")``. Use
        ``enable_async_load_handlers("none")`` to turn off again.
        """
        if call_soon_threadsafe == "none":
            self._call_soon_threadsafe = None
        elif call_soon_threadsafe == "asyncio":
            import asyncio  # noqa

            self._call_soon_threadsafe = asyncio.get_running_loop().call_soon_threadsafe
        elif callable(call_soon_threadsafe):
            self._call_soon_threadsafe = call_soon_threadsafe
        else:
            raise TypeError(
                "CheckPool.enable_async_load_handlers(): unexpected call_soon_threadsafe value: {call_soon_threadsafe!r}"
            )

    def destroy(self):
        """Drop and destroy all chunks."""
        for chunk_dict in self._chunks:
            chunks = list(chunk_dict.values())
            chunk_dict.clear()
            for chunk in chunks:
                chunk._destroy()

    def get_chunk(
        self,
        level: int,
        chunk_index: tuple[int, ...],
        ref: str = "pool",
        *,
        load_handler=None,
        drop_handler=None,
        destroy_handler=None,
    ) -> ChunkLocation:
        """Get a ChunkLocation object.

        Parameters
        ----------
        level : int
            The scale level for the requested chunk.
        index : tuple[int, ...]
            The index for the requested chunk.
        ref : str
            The reference to identify the code that requests the chunk. Default 'pool'.
            This is used by the pool to ref-count the chunk usage and destroy
            the chunk when there are no more refs left. Also see the ``ChunkManager``.
        handlers : callable
            Functions that will be called at specific lifetime events of the
            chunk.

        Returns
        -------
        chunk_location : ChunkLocation
            A representation of the requested chunk. The corresponding data is
            being loaded but may not be ready yet. If async loading is enabled,
            the load handler will be called as soon as the data arrives.

        Individual chunks can be loaded synchronously using::

            chunk_spot.wait()
            data = chunk_spot.data

        After getting multiple chunks, it's easy to load them in parallel::

            loop.wait_for_chunks_to_load()

        This is equivalent to:

            chunk_spots = [...]
            for chunk_spot in chunk_spots:
                chunk_spot.wait()
        """
        if not (isinstance(ref, str) and len(ref) > 0):
            raise TypeError("get_chunk() ref must be a nonempty string.")

        chunk_spot = self._chunks[level].get(chunk_index, None)
        if chunk_spot is None:
            chunk_info = self._multiscale_info.scales[level]
            chunk_spot = ChunkLocation(chunk_info, chunk_index)
            self._chunks[level][chunk_index] = chunk_spot

        chunk_spot._register(
            ref, self._call_soon_threadsafe, load_handler, drop_handler, destroy_handler
        )

        return chunk_spot

    def drop_chunk(
        self, level: int, chunk_index: tuple[int, ...], ref: str = "pool"
    ) -> None:
        """Release a chunk by their index.

        Parameters
        ----------
        level : int
            The scale level for the requested chunk.
        index : tuple[int, ...]
            The index for the requested chunk.
        ref : str
            The reference to identify the code that requested the chunk. Default 'pool'.
            This must be the same value as when ``get_chunk()`` was called.

        This drops the chunk, invoking any drop handlers. When the chunk has no
        more refs, the chunk is destroyed. When the code has a single user,
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


manager_counter = Count(1)


class ChunkManager:
    """A simple wrapper for a ``ChunkPool`` that represents one specific 'user' of the pool.

    To use this class, subclass it and implement the ``on_load``, ``on_drop`` and ``on_destroy`` methods.
    """

    def __init__(self, pool):
        if not isinstance(pool, ChunkPool):
            raise TypeError(f"ChunkManager expects a ChunkPool instance, got {pool!r}")
        self._pool = pool
        self._ref = f"{self.__class__.__name__}-{next(manager_counter)}"

    def get_chunk(self, level: int, chunk_index: tuple[int, ...]):
        """Get a ChunkLocation object.

        Parameters
        ----------
        level : int
            The scale level for the requested chunk.
        index : tuple[int, ...]
            The index for the requested chunk.

        Returns
        -------
        chunk_location : ChunkLocation
            A representation of the requested chunk. The corresponding data is
            being loaded but may not be ready yet. If async loading is enabled,
            the ``on_load`` method will be called as soon as the data arrives.

        The managers ``on_load``, ``on_drop``, and ``on_destroy``, are automatically registered as handlers.
        """

        return self._pool.get_chunk(
            level,
            chunk_index,
            self._ref,
            load_handler=self.on_load,
            drop_handler=self.on_drop,
            destroy_handler=self.on_destroy,
        )

    def drop_chunk(self, level: int, chunk_index: tuple[int, ...]) -> None:
        """Release a chunk by their index.

        Parameters
        ----------
        level : int
            The scale level for the requested chunk.
        index : tuple[int, ...]
            The index for the requested chunk.

        This drops the chunk, invoking any drop handlers. When the chunk has no
        more refs, the chunk is destroyed.
        """
        return self._pool.drop_chunk(level, chunk_index, self._ref)

    def on_load(self, chunk_location: ChunkLocation):
        """Method that gets called when a chunk is loaded. Override this in your subclass."""
        pass

    def on_drop(self, chunk_location: ChunkLocation):
        """Method that gets called when a chunk is dropped. Override this in your subclass."""
        pass

    def on_destroy(self, chunk_location: ChunkLocation):
        """Method that gets called when a chunk is destroyed. Override this in your subclass."""
        pass


class ChunkLocation:
    """An object that represents a chunk location."""

    def __init__(self, scale_info: ScaleInfo, index: tuple[int, ...]):
        self._scale_info = scale_info
        self._index = index

        self._refs = set()
        self._data = None
        self._future = scale_info.array.get_chunk_future(index)
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
        if self._future is not None:
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
