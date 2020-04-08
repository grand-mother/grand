:mod:`~grand.store` --- Interface to remote data storage
========================================================
.. module:: grand.store

.. warning::
   The :mod:`~grand.store` module is under construction. Its purpose is to
   provide a simple interface for accessing large data files, stored remotly.
   Currently it uses GitHub release area, which is a bit of a hack.

.. autofunction:: grand.store.get

   For example, the following retrieves an `SRTMGL1`_ topography tile from the
   store, as a :class:`bytes` object.

   >>> data = store.get('N39E090.SRTMGL1.hgt')

.. autoclass:: grand.store.InvalidBLOB


.. _SRTMGL1: https://www.arcgis.com/home/item.html?id=cadb028a356046479fcda5207a235560
