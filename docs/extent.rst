Extending simplezarr
====================

The Zarr spec allows multiple extension points. This doc is a work in progress
to explain how simplezarr can likewise be extended.


Custom codecs
-------------

One can create a custom codec by subclassing ``BaseCodec`` and registering it.

.. autofunction:: simplezarr.codecs.register_codec

.. autoclass:: simplezarr.codecs.BaseCodec
    :members:
    :member-order: bysource

.. autoclass:: simplezarr.codecs.ArrayType
    :members:
    :member-order: bysource
