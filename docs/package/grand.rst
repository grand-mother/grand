API documentation
=================
.. module:: grand

.. note::
   If not stated explicitly, all examples assume that the :mod:`grand` package
   and :mod:`astropy.units` have been imported as below.

>>> from grand import *
>>> import astropy.units as u


Interfaces
----------

The :mod:`grand` package exposes the following interface modules. Most of those
are wrappers, E.g. for core `C` libraries used in the simulation.

.. toctree::
   :maxdepth: 1

   grand.geomagnet
   grand.io
   grand.montecarlo
   grand.store
   grand.topography


Extensions
----------

In addition, the :mod:`grand` package extends the :mod:`astropy` package with
the following functionalities.

.. toctree::
   :maxdepth: 1

   grand.coordinates
