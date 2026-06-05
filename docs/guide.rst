Guide
=====

Installation
------------

You can install ``simplezarr`` via pip (or most other Python package managers).
Python 3.12 or higher is required.

.. code-block:: bash

    pip install simplezarr


Opening a zarr file
-------------------

.. code-block:: py

    # Create a store
    store = simplezarr.LocalStore(filename)

    # Open as zarr
    group = simplezarr.open_zarr(store)

    # Navigate to array
    array = group["foo/array"]


Get some info
-------------

.. code-block:: py

    group.print_metadata()
    group.print_structure()
    group.attributes
    group.children

    array.print_metadata()
    array.dtype
    array.nbytes
    array.ndim
    array.shape
    array.chunk_shape
    array.chunk_grid_shape
    # etc.


Read some data
--------------

.. code-block:: py

    # Get a numpy array (blocking)
    a = array[:200, :200].get_now()

    # Load two regions in parallel
    f1 = array[:200, :200].get_soon()
    f2 = array[1000:1200, :200].get_soon()
    a1, a2 = [f1.result(), f2.result()]


Write some data
---------------

.. code-block:: py

    # Put random data in a region (blocking)
    a = np.random.uniform(size=(200, 200))
    array[:200, :200].set_now(a)

    # Zero out two regions in parallel
    f1 = array[:200, :200].set_soon(0)
    f2 = array[1000:1200, :200].set_soon(0)

    # You want to wait so that any errors during writing are raised
    [f1.result(), f2.result()]


Read and write whole chunks
---------------------------

.. code-block:: py

    # Load first chunk as numpy array
    a = array.chunks[0, 0].get_now()

    # Load a series as chunk as a single array (in parallel, blocking)
    a = array.chunks[0, :].get_now()

    # Writing and `_soon` works the same


Read and write individual chunks
--------------------------------

.. code-block:: py

    # Lower-level function to read individual chunks
    a = array.get_chunk_now((0, 0))

    # Read parallel
    f1 = array.get_chunk_soon((0, 0))
    f2 = array.get_chunk_soon((10, 8))
    a1, a2 = [f1.result(), f2.result()]

    # Writing
    array.set_chunk_now((0, 1), a)

    # Write in parallel
    f1 = array.set_chunk_soon((0, 2), a)
    f2 = array.set_chunk_soon((0, 3), a)
    [f1.result(), f2.result()]


Creating Zarr files
-------------------

.. code-block:: py

    store = simplezarr.MemoryStore()

    simplezarr.ZarrGroup.create(store, "")

    arr = simplezarr.ZarrArray.create(store, "array1", (1000, 1000), "uint16", chunk_shape=(64, 64))
    arr[...].set_now(42)

    arr = simplezarr.ZarrArray.create(store, "array2", (100, 100, 100), "float32", chunk_shape=(10, 10, 10))
    arr[5:-5, 5:-5, 5:-5].set_now(1.0)


More functionality
------------------

That's it really ... any other functionality is built on top of this, and provided via dedicated modules in ``simplezarr.utils``.
