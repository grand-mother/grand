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

Measured on ``dev-next`` on 2026-09-24, in lines of Python (``wc -l`` over
each subpackage):

======================================  =====  =====
Subpackage                              Lines  Share
======================================  =====  =====
``grand.sim`` — instrument response     5856   29.1%
``grand.dataio`` — data model           5086   25.3%
``grand.geo`` — geometry and geodesy    3857   19.2%
``grand.aoi`` — user-facing API         2171   10.8%
``grand.basis`` — traces and array viz  1908   9.5%
``grand.analysis`` — reconstruction     1261   6.3%
======================================  =====  =====

The pipeline runs forward, shower to field to voltage to ADC, and since
2026-09 it also runs back: :mod:`grand.analysis` (Marion Guelfand, merged
from ``dev_marion``) reconstructs arrival direction, the distance to the
shower maximum and an electromagnetic-energy proxy from recorded times and
amplitudes, with plane-wave, spherical-wave and ADF fits. It is young: its
tests show the fits recover what their own forward models generate, not yet
that they agree with simulated or measured showers. It replaced
``grand.recon``, a placeholder of two empty constructors.

Layering
--------

The intended layering is that :mod:`grand.geo` and :mod:`grand.dataio` sit at
the bottom and know nothing above them, :mod:`grand.sim` composes both, and
:mod:`grand.aoi` and :mod:`grand.basis` sit on top providing the objects a user
handles and the plots they look at.

That is not what the imports say.

.. image:: _static/modules.svg
   :target: _static/modules.svg
   :alt: dependency graph of the six grand subpackages, measured from their
         import statements, showing a three-edge cycle between geo, dataio and
         basis
   :width: 100%

*Click the figure to open it full size.*

The edges are measured rather than asserted;
``python docs/dev/make_modules_diagram.py --measure`` re-derives them by
walking every import statement under ``grand/``, and the diagram is generated
from the result.  Re-run it after moving code between subpackages.

.. warning::

   **There is a module-level import cycle**, and it is not visible from any one
   file:

   .. code-block:: text

       grand.geo.topography     imports  grand.dataio.protocol
       grand.dataio.root_files  imports  grand.basis.traces_event
       grand.basis.type_trace   imports  grand.geo.coordinates

   Three modules, three subpackages, back to the start.  Nothing breaks today —
   Python tolerates a cycle whose members do not need each other's names at
   import time — but whether it keeps working depends on import order, which no
   one is choosing deliberately.

   The first edge is also why importing anything from ``grand`` requires ROOT:
   it pulls the ROOT-dependent data layer into what should be a self-contained
   geometry module.  See :ref:`issue-import-requires-root`.

A fourth edge would close a second cycle and does not, because it is deferred:
``grand/basis/pipeline.py`` imports :class:`~grand.sim.efield2voltage.Efield2Voltage`
*inside a function* rather than at module level.  That is the pattern to follow
if a genuine back-edge is unavoidable — but breaking the ``geo`` edge properly
is worth more than adding another deferred import.

:mod:`grand.analysis` sits on top: it imports geometry (through the package
namespace, ``from grand import Geodetic``) and nothing in ``grand`` imports it
back.
