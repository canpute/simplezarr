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
    g = simplezarr.open_zarr()

    # Navigate it
    a = g["foo/array"]

At the time of writing, reading individual chunks is supported, but slicing the array is not yet.


