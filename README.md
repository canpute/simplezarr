# simplezarr
A simple, elegant, and efficient Zarr implementation


## Intro

Zarr 3 is a great file format for large datasets. It's nice and elegant.
The `simplezarr` lib is what happens when one implemented this spec as directly as possible in Python.


## Details

Parallelism and async are achieved using a thread-pool and `concurrent.futures.Future` objects.
You can use it with async frameworks (like asyncio), but none of it is forced.
You can use it synchronously and still benefit from parallelism.
Calls that return a `Future` have a `_soon` suffix. Calls that are blocking have a `_now` suffix.

The core is simple, easy to follow, and gives predictable performance.
Extra functionality is provided as functions and classes that are provided in `simplezarr.utils`.
This keeps the lib light, and easy to adopt in a wide variety of use cases.

The Zarr 3.1 spec is fully implemented, except for sharding. The core has 100% test coverage.


## Installation

```
pip install simplezarr
```

## Quick example

Write a Zarr file to an in-memory store:

```py
>>> import simplezarr

>>> store = simplezarr.MemoryStore()

>>> simplezarr.ZarrGroup.create(store, "")

>>> arr = simplezarr.ZarrArray.create(store, "array1", (1000, 1000), "uint16", chunk_shape=(64, 64))

>>> arr[...].set_now(42)
```

Reading:

```py
>>> group = simplezarr.open_zarr(store)

>>> group
<ZarrGroup '' with 1 children at 0x10910a5d0>
    <ZarrArray 'array1' 1000x1000 uint16 at 0x10910a490>

>>> arr = group["array1"]

>>> a = arr[:100, :100].get_now()  # blocking, but reads chunks in parallel

>>> a
array([[42, 42, 42, ..., 42, 42, 42],
       [42, 42, 42, ..., 42, 42, 42],
       [42, 42, 42, ..., 42, 42, 42],
       ...,
       [42, 42, 42, ..., 42, 42, 42],
       [42, 42, 42, ..., 42, 42, 42],
       [42, 42, 42, ..., 42, 42, 42]], shape=(100, 100), dtype=uint16)
```

Parallel/lazy reads:
```py
>>> f1 = array[:200, :200].get_soon()

>>> f2 = array[-200:, -200:].get_soon()

>>> a1, a2 = [f1.result(), f2.result()]
```


## Developers

* Clone the repo.
* Install `rendercanvas` and developer deps using `pip install -e .[dev]`.
* Use `ruff format` to apply autoformatting.
* Use `ruff check` to check for linting errors.
* Use `pytest tests` to run the tests. Or `pytest tests --cov=simplezarr --cov-report=html` to get coverage reporting.


## License

This code is distributed under the MIT license.
