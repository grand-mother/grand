Coordinate systems
==================

.. contents::
   :local:
   :depth: 2

Getting a simulated electric field onto a real antenna is, more than anything
else, a coordinate problem.  Air showers are computed in shower coordinates;
antennas sit at geodetic positions on curved, uneven terrain; the radio
emission is driven by the local geomagnetic field.  Reconciling those is what
:mod:`grand.geo.coordinates` is for, and it is the single largest source of
user error in the library.

The frames
----------

============  ==========================================================
Frame         What it is
============  ==========================================================
``Geodetic``  Latitude, longitude, height above the WGS-84 ellipsoid.
``ECEF``      Earth-Centred Earth-Fixed Cartesian; the common pivot.
``LTP``       Local tangent plane at a given origin and orientation.
``GRANDCS``   The array frame: an ``LTP`` with GRAND's conventions.
============  ==========================================================

Every conversion between ``LTP``/``GRANDCS`` and geodetic goes through
``ECEF``.  There is no direct path, and that is deliberate — one pivot means
one place for the ellipsoid constants to live.

The transformation
------------------

For a location at longitude :math:`\phi` and latitude :math:`\theta`, the
basis vectors of the local frame, expressed in ECEF, are

.. math::

   \boldsymbol{E} = \begin{bmatrix} -\sin\phi \\ \cos\phi \\ 0 \end{bmatrix},\quad
   \boldsymbol{N} = \begin{bmatrix} -\sin\theta\cos\phi \\ -\sin\theta\sin\phi \\ \cos\theta \end{bmatrix},\quad
   \boldsymbol{U} = \begin{bmatrix} \cos\theta\cos\phi \\ \cos\theta\sin\phi \\ \sin\theta \end{bmatrix},

and the transformation matrix is :math:`\boldsymbol{R} = [\boldsymbol{E}\;
\boldsymbol{N}\; \boldsymbol{U}]^{\mathsf{T}}`, so that
:math:`\boldsymbol{V}_{\mathrm{ENU}} = \boldsymbol{R}\,\boldsymbol{V}_{\mathrm{ECEF}}`
and :math:`\boldsymbol{V}_{\mathrm{ECEF}} = \boldsymbol{R}^{\mathsf{T}}\,
\boldsymbol{V}_{\mathrm{ENU}}`.

Because :math:`\boldsymbol{R}` is orthogonal, the inverse is the transpose —
which is worth stating because it is the property every round-trip test in
:mod:`tests.geo` relies on.

.. jupyter-execute::

    import numpy as np

    def enu_basis(lat_deg, lon_deg):
        """Returns the ECEF-frame E, N, U basis vectors at a location."""
        th, ph = np.radians(lat_deg), np.radians(lon_deg)
        E = np.array([-np.sin(ph),  np.cos(ph), 0.0])
        N = np.array([-np.sin(th)*np.cos(ph), -np.sin(th)*np.sin(ph), np.cos(th)])
        U = np.array([ np.cos(th)*np.cos(ph),  np.cos(th)*np.sin(ph), np.sin(th)])
        return np.vstack([E, N, U])

    # The GRANDProto300 site at Dunhuang, from arXiv:2408.10926 Fig. 7.
    R = enu_basis(40.98, 93.95)

    print("R is orthogonal:", np.allclose(R @ R.T, np.eye(3)))
    print("round-trip error:", np.abs(R.T @ (R @ np.array([1.0, 2.0, 3.0]))
                                      - np.array([1.0, 2.0, 3.0])).max())

The WGS-84 constants
--------------------

.. jupyter-execute::

    a = 6378137.0                # semi-major axis, m
    e = 0.081819190842622        # eccentricity
    lat = np.radians(40.98)
    r = a / np.sqrt(1.0 - (e*np.sin(lat))**2)
    print("prime vertical radius of curvature at Dunhuang: %.1f m" % r)

.. note::

   The blocks above are executed when this page is built, so they cannot go
   stale.  They are written in plain NumPy rather than calling
   :mod:`grand.geo.coordinates` because the library cannot currently be
   imported without a full ROOT runtime — the constraint that Phase 6 of the
   overhaul exists to remove.  Once it does, these become calls into the
   library and this page becomes its executable specification.

Reference
---------

Appendix A of `arXiv:2408.10926 <https://arxiv.org/abs/2408.10926>`_.
