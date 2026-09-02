Code architecture
=================

This page describes how the source is organised.  See :doc:`coordinates` and
:doc:`datamodel` for what the pieces mean physically.

What GRANDlib is
----------------

GRANDlib simulates comparatively little physics itself.  Air showers and their
radio emission come from ZHAireS or CoREAS, tau propagation from DANTON,
terrain from TURTLE, the geomagnetic field from GULL, and sky brightness from
LFMap.  What GRANDlib owns is three things:

1. **A schema** — what a GRAND event *is*, and the format the collaboration
   stores it in (:mod:`grand.dataio`).
2. **A frame-reconciliation engine** — where things are, in which frame, over
   what terrain, in what magnetic field (:mod:`grand.geo`).
3. **An instrument-response model** — effective length, Galactic noise, RF
   chain, ADC (:mod:`grand.sim`).

Composition
-----------

Measured on the ``dev`` branch, by lines of Python:

======================================  =====  =====
Subpackage                              Lines  Share
======================================  =====  =====
``grand.sim`` — instrument response     4510   31.9%
``grand.dataio`` — data model           3696   26.1%
``grand.geo`` — geometry and geodesy    2375   16.8%
``grand.aoi`` — user-facing API         1602   11.3%
``grand.basis`` — traces and array viz  1512   10.7%
``grand.recon`` — reconstruction        28     0.2%
======================================  =====  =====

The last row is the thing to notice.  The pipeline runs forward only:
shower to field to voltage to ADC.  Reconstruction — recovering direction,
core position, energy and depth of shower maximum from recorded voltages —
is where the experiment's real questions live, and it is not yet here.

Layering
--------

``grand.geo`` and ``grand.dataio`` sit at the bottom and know nothing above
them.  ``grand.sim`` composes both.  ``grand.aoi`` and ``grand.basis`` sit on
top, providing the objects a user handles and the plots they look at.

.. warning::

   The layering is not currently enforced, and one dependency runs the wrong
   way: :mod:`grand.geo.topography` imports :mod:`grand.dataio.protocol`,
   which pulls the whole ROOT-dependent data layer into what should be a
   self-contained geometry module.  This is why importing anything from
   ``grand`` requires ROOT — including, until Phase 6, importing it in order
   to document it.
