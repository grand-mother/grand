API documentation
=================
.. module:: grand

.. note::
   If not stated explicitly, all examples assume that the :mod:`grand` package
   and :mod:`astropy.units` have been imported as below.

>>> from grand import *
>>> import astropy.units as u


Wrappers
--------

The :mod:`grand` package exposes the following interface modules. Those are
wrappers, E.g. for core `C` libraries used in the simulation.

.. toctree::
   :maxdepth: 1

   grand.geomagnet
   grand.store
   grand.topography


Extensions
----------

In addition, the :mod:`grand` package extends the following :mod:`astropy` functionalities.

.. toctree::
   :maxdepth: 1

   grand.coordinates
