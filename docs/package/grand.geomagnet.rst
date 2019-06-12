:mod:`~grand.geomagnet` --- Geomagnetic helpers
===============================================
.. module:: grand.geomagnet

The :mod:`~grand.geomagnet` module provides a simple interface to world wide
models of the Earth magnetic field. It implements simple getter functions,
:func:`~grand.geomagnet.field` and :func:`~grand.geomagnet.model`, in
order to access the default geomagnetic model. For test purpose alternative
models can also be instanciated as :class:`~grand.geomagnet.Geomagnet` object.

.. warning::
   The input *coordinates* to the :func:`~grand.geomagnet.field` function or
   method must be a :mod:`grand` frame, i.e. :class:`~grand.ECEF` or
   :class:`~grand.LTP`. In addition, an observation time must be specified.

.. note::
    The :func:`~grand.geomagnet.field` function and method accept vectorized
    input. The frame of the returned value depends on the input size. If a
    single point is provided the magnetic field is returned in local
    :class:`~grand.LTP` coordinates, centered on the input *coordinates*.
    Otherwise the components are given in :class:`~grand.ECEF`.


Default interface
-----------------

.. autofunction:: grand.geomagnet.field

   For example, the following returns the Earth magnetic field at sea level,
   close to Clermont-Ferrand, France, for new year 2019.

   >>> coordinates = ECEF(representation_type="geodetic", latitude=45 * u.deg,
   ...                    longitude=3 * u.deg, obstime="2019-01-01")
   >>> field = geomagnet.field(coordinates)

.. autofunction:: grand.geomagnet.model


Alternative magnetic models
---------------------------

.. warning::
   In order to get the geomagnetic field from the default GRAND model (IGRF12),
   one should not directly instantiate a :class:`~grand.geomagnet.Geomagnet`
   object.  Instead one should use the :func:`~grand.geomagnet.field` function
   of the :mod:`~grand.geomagnet` module.  This class is only meant to be used
   for studies of the impact of alternative models.

Supported models for the Earth magnetic field are: `IGRF12`_ and `WMM2015`_.

.. autoclass:: grand.geomagnet.Geomagnet

   The default model is `IGRF12`_. You can instanciate an alternative model by
   specifying its name, E.g.  as follow:

   >>> magnet = geomagnet.Geomagnet("WWM2015")

   .. automethod:: grand.geomagnet.Geomagnet.field


.. _IGRF12: https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
.. _WMM2015: http://www.geomag.bgs.ac.uk/research/modelling/WorldMagneticModel.html
