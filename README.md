# simplezarr
A simple, elegant, and efficient Zarr implementation

The core of `simplezarr` implements the Zarr 3 spec in straightforward
Python, without extra fuzz. This makes the code easy to follow, and gives
predictable performance. Extra functionality is provided as functions and classes
that are provided in `simplezarr.utils`.

Since `simplezarr` is nice and simple, it's easy to adopt in various
use-cases. It supports parallel io, but does not force the use of asyncio.


## Status

* Stores are implemented, (except no remote stores yet).
* Codecs are implemented (all except for sharding).
* Main API can (asynchronously) read and write chunks.

What is not yet supported:

* Writing Zarr files.
* Indexing (wip).
* Sharding.


## Motivation

Zarr 3 is a great file format for large datasets. It's nice and elegant. The
`simplezarr` lib is what happened when we took the Zarr 3 spec, and implemented
it as directly as possible.

Parallelism is achieved using a thread-pool and `concurrent.futures.Future`
objects. And in once place exactly: the code that reads a chunk (`ZarrArray.get_chunk_soon()`).

We don't force asyncio. In fact, ``simplezarr`` does not even import
asyncio (except in code paths that represent a utility specific to asyncio
users).

## Comparison with zarr-python

Why not use zarr-python? We ran into performance issues, and upon
investigating what happens under the hood, we found it hard to follow the path
that the code takes, especially regarding threading and asyncio. Granted, part
of that complexity is because it must support older Zarr versions as well.
Another reason is that zarr-python does not seem to have a way to read individual blocks
asynchronously (`AsyncArray.get_block_selection()` does not exist), which was a
requirement for our use-case.

### What zarr-python does

* The store loads data using `asyncio.to_thread()`. This runs the io-bound reading of bytes in a separate thread (from the loop's default `ThreadPoolExecutor`).
* It uses `asyncio.gather()` is parallelize concurrent reads/writes.
* When using the `zarr.Array` (not `AsyncArray`), indexing is synchronous. To do this:
  * It uses a dedicated asyncio loop that runs continuously in a dedicated thread.
  * A dedicated `ThreadPoolExecutor` is set on that loop (which will be used to perform the store IO with).
  * Then `asyncio.run_coroutine_threadsafe(the_asyncio_coroutine, dedicated_loop)` to turn the asyncio code into a `concurrent.futures.Future`.
  * Then sync-wait on that future.

It looks like this complexity is one of the reasons why the performance of ome-zarr is hard to get right. The ome-zarr library wraps zarr-python with Dask, which uses thread pools too, which results in a lot of threads being spawned.

### What simplezarr does

* Stores are synchronous.
* `simplezarr.Array.get_chunk_now()` is synchronous (no threading or async).
* `simplezarr.Array.get_chunk_soon()` uses a `ThreadPoolExecutor`. It returns a `concurrent.futures.Future`.
* This is enough to support concurrently reads.
* No asyncio anywhere.
* But can be used in `asyncio` (and other frameworks) using `await asyncio.wrap_future(f)` or `f.add_done_callback(call_soon_threadsafe)`.
