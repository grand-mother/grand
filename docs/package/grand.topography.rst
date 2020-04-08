:mod:`~grand.topography` --- Topography helpers
===============================================
.. module:: grand.topography

The :mod:`~grand.topography` module provides a simple interface to world wide
elevation models. It implements simple getter functions,
:func:`~grand.topography.elevation` and
:func:`~grand.topography.geoid_undulation`, in order to access the default
topography model. For test purpose alternative models can also be instanciated
as :class:`~grand.topography.Topography` object.

The topography elevation data are stored in a local cache within the ``grand``
user space.  The :func:`~grand.topography.update_data` allows to manage this
cache.

.. warning::
   Topography elevation data are not automatically downloaded. They must be
   requested explicitly with the :func:`~grand.topography.update_data` function
   before calling :func:`~grand.topography.elevation`.


Default interface
-----------------

.. warning::
   The input *coordinates* to the :func:`~grand.topography.elevation` and
   :func:`~grand.topography.geoid_undulations` functions must be a :mod:`grand`
   frame, i.e. :class:`~grand.ECEF` or :class:`~grand.LTP`. If magnetic
   coordinates are used, the observation time must be specified as well.

.. note::
   The :func:`~grand.topography.elevation` and
   :func:`~grand.topography.geoid_undulations` functions accept vectorized data.
   The returned value is a :py:class:`~astropy.units.Quantity` in meters.

.. autofunction:: grand.topography.distance

   .. note::
      The returned distance is signed. A positive (negative) value indicates
      that the initial position is above (below) the ground. The absolute value
      gives the actual distance.

.. autofunction:: grand.topography.elevation

   For example, the following provides the topography elevation, w.r.t. the sea
   level, close to Urumuqi, China, provided that the corresponding
   data have been cached.

   >>> coordinates = ECEF(latitude=43.83 * u.deg, longitude=87.62 * u.deg,
   ...                    representation_type='geodetic')
   >>> elevation = topography.elevation(coordinates,
   ...                                  reference=topography.Reference.GEOID)

   .. note::
      By default, if no explicit reference is specified, the elevation is given
      w.r.t. the ellipsoid if geocentric coordinates are used (ECEF) or in local
      coordinates if local coordinates (LTP) are provided. In the latter case,
      the elevation data naturally include the Earth curvature.


.. autofunction:: grand.topography.geoid_undulation

   For example, the following corrects the previous elevation from the geoid
   undulation.

   >>> elevation += topography.geoid_undulation(coordinates)


.. autofunction:: grand.topography.model

   Currently the default topographic model used in GRAND is `SRTMGL1`_.


Cache management
----------------

Topography elevation data are cached in the user space, under
:func:`~grand.topography.cachedir`. The cache can be managed with the
:func:`~grand.topography.update_data` function.

.. autofunction:: grand.topography.update_data

   If *coordinates* are provided, the corresponding data tiles are downloaded
   to the cache. The requested area can be enlarged by specifying a *radius*
   around the *coordinates*. For example, the following ensures that the
   elevation data within 10 km of *coordinates* are available in the cache:

   >>> topography.update_data(coordinates, radius=10 * u.km)

   If *clear* is set to `True` (defaults to `false`) the cache is emptied. This
   occurs before any other operation, i.e. it can be combined with a data
   request.


.. autofunction:: grand.topography.cachedir


Alternative topography models
-----------------------------

.. warning::
   In order to get the topography elevation from the default GRAND model
   (`SRTMGL1`_), one should not directly instantiate a
   :class:`~grand.topography.Topography` object.  Instead one should use the
   :func:`~grand.topography.elevation` function of the :mod:`~grand.topography`
   module.  This class is only meant to be used for studies of the impact of
   alternative models.

.. note::
   Currently only `SRTMGL1`_ tiles are in the :mod:`~grand.store`. Data for
   alternative models must be provided manually.

.. autoclass:: grand.topography.Topography

   For example, assuming that you have the topography tiles corresponding to
   `ASTER GDEM2`_ in your cache, the following allows to access those.

   >>> topo = topography.Topography('ASTER-GDEM2')


.. _SRTMGL1: https://www.arcgis.com/home/item.html?id=cadb028a356046479fcda5207a235560
.. _ASTER GDEM2: https://www.arcgis.com/home/item.html?id=93545c023ec44b109be1b3425edc72e1
