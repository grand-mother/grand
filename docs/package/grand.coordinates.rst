Extension of :mod:`astropy.coordinates`
=======================================

The :mod:`grand` package extends :mod:`astropy.coordinates` with geographic
representations and local frames. It adds the :class:`~grand.ECEF` and
:class:`~grand.LTP` frames as well as the
:class:`~grand.GeodeticRepresentation` and
:class:`~grand.HorizontalRepresentation` of coordinates.


Frames
------

.. warning::
   The :mod:`grand` frames are co-moving with the Earth. i.e. the *obstime* can
   be omitted, in which case it is inherited from the source frame during
   transforms. This differs from base :mod:`astropy.coordinates` frames where
   the observation time must be stated explicitly during a transform (see E.g.
   `#8390`_).

If specified, the observation time must be an instance of an `astropy`
:class:`~astropy.time.Time` object or any of its initializers, E.g.
:class:`~datetime.datetime`, or `str`. 

In addition, the frame classes defined in :mod:`astropy.coordinates` do not
distinguish points from vectors when transforming between frames. The `grand`
frames do. An automatic type inference occurs based on the data
:mod:`astropy.units`, as following.

.. warning::
   If the frame data are homogeneous to a length, there are assumed to
   transform as a point. Otherwise, they transform as a vector quantity.


ECEF frame
^^^^^^^^^^

The :class:`~grand.ECEF` class (`Earth-Centered Earth-Fixed`_) is a wrapper for
geocentric frames. It behaves as the :class:`~astropy.coordinates.ITRS` frame
but co-moving with the Earth, i.e. the *obstime* can be omitted.


.. autoclass:: grand.ECEF

   .. note::
      By default, a :class:`~astropy.coordinates.CartesianRepresentation` is
      expected for the coordinates data, i.e. *x*, *y* and *z*. For example,
      the following represents a point located at the origin.

   >>> origin = ECEF(x = 0 * u.m, y = 0 * u.m, z = 0 * u.m)

   .. autoproperty:: grand.ECEF.earth_location
   .. autoproperty:: grand.ECEF.obstime


LTP frame
^^^^^^^^^

The :class:`~grand.LTP` class (`Local Tangent Plane coordinates`_) allows to
define local frames tangent to the WGS84 ellipsoid. The *orientation* property
can be used to define the frame axes, along cardinal directions. E.g. the
following defines a local North, East, Down (NED) frame centered on Greenwich,
as:

>>> from astropy.coordinates import EarthLocation
>>> ltp = LTP(location=EarthLocation.of_site("greenwich"),
...           orientation=("N", "E", "D"))

Alternatively, magnetic coordinates can be used as well by setting *magnetic* to
`True`. By default geographic East, North, Up (ENU) coordinates are used.

.. note::
   One must always specify the local frame origin, as the *location* parameter.
   If magnetic coordinates are used, the *obstime* must be specified as well.

.. autoclass:: grand.LTP

   .. note::
      By default, a :class:`~astropy.coordinates.CartesianRepresentation` is
      expected for the coordinates data, i.e. *x*, *y* and *z*.

   .. autoproperty:: grand.LTP.earth_location
   .. autoproperty:: grand.LTP.magnetic
   .. autoproperty:: grand.LTP.orientation
   .. autoproperty:: grand.LTP.obstime


Representations
---------------

The :mod:`grand` package explictly adds two geographic representations based on
the WGS84 ellipsoid. Note that although `astropy` uses the corresponding
transforms it does not explictly bundle them as coordinates representations.

Geodetic coordinates
^^^^^^^^^^^^^^^^^^^^

A :class:`~grand.GeodeticRepresentation` is analog to a
:class:`~astropy.coordinates.SphericalRepresentation` but mapping to the WGS84
ellipsoid instead of a sphere. It allows to represent :mod:`~grand.ECEF`
coordinates using a `geodetic datum`_ instead of the default
:mod:`astropy.coordinates.Cartesian`, E.g. as:

>>> point = ECEF(latitude=45 * u.deg, longitude=3 * u.deg, height=0.5 * u.m,
...              representation_type="geodetic")

.. autoclass:: grand.GeodeticRepresentation

   .. note::
      The *latitude* angle is measured clockwise, w.r.t. the xOy plane. The
      *longitude* angle is measured counter-clockwise, w.r.t. the x-axis. The
      *height* is w.r.t. the WGS84 ellipsoid. It defaults to `0` meters if
      ommitted.

   .. automethod:: grand.GeodeticRepresentation.from_cartesian
   .. automethod:: grand.GeodeticRepresentation.to_cartesian


Horizontal coordinates
^^^^^^^^^^^^^^^^^^^^^^

The :class:`~grand.HorizontalRepresentation` allows to represent a unit vector,
e.g. a direction, in a local :class:`~grand.LTP` frame using a `horizontal
coordinates system`_. For example, the following defines a unit vector pointing
upwards at Greenwich:

>>> from astropy.coordinates import EarthLocation
>>> up = LTP(location=EarthLocation.of_site("greenwich"),
...          representation_type="horizontal",
...          azimuth = 0 * u.deg, elevation = 90 * u.deg)

.. autoclass:: grand.HorizontalRepresentation

   .. note::
      The *azimuth* angle is measured clockwise, w.r.t. the y axis. The
      *elevation* angle is measured clockwise, w.r.t. the xOy plane.

   .. automethod:: grand.HorizontalRepresentation.from_cartesian

      .. warning::
         The Cartesian unit vector **must** be dimensioneless. Though it is not
         checked if the norm of the vector is indeed unity.

   .. automethod:: grand.HorizontalRepresentation.to_cartesian


.. _#8390: https://github.com/astropy/astropy/issues/8390
.. _Earth-Centered Earth-Fixed: https://en.wikipedia.org/wiki/ECEF
.. _Local Tangent Plane coordinates: https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
.. _geodetic datum: https://en.wikipedia.org/wiki/Geodetic_datum
.. _horizontal coordinates system: https://en.wikipedia.org/wiki/Horizontal_coordinate_system
