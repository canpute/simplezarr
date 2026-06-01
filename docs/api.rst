API
===

The lightweigth core of simplezarr.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

Stores
------

.. automodule:: simplezarr.stores
    :members:
    :member-order: bysource
    :show-inheritance:

Nodes
-----

.. automodule:: simplezarr.nodes

.. autofunction:: simplezarr.open_zarr

.. autoclass:: simplezarr.ZarrNode
    :members:
    :member-order: bysource
    :show-inheritance:

.. autoclass:: simplezarr.ZarrGroup
    :members:
    :member-order: bysource
    :show-inheritance:

.. autoclass:: simplezarr.ZarrArray
    :members:
    :special-members: __getitem__
    :member-order: bysource
    :show-inheritance:


Indexing
--------

.. autoclass:: simplezarr.ZarrArraySlice
    :members:
    :member-order: bysource
    :show-inheritance:
