API reference
=============

Every public module, class and function of GRANDlib, generated from the
docstrings.  The modules are grouped by what they are for; each page opens
with a table of its modules.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Contents
   * - :doc:`api/geo`
     - Coordinate frames and conversions, terrain and the geoid, the
       geomagnetic field
   * - :doc:`api/dataio`
     - The ROOT trees, and reading and writing files
   * - :doc:`api/sim`
     - The simulation chain: ``Efield2Voltage``, the antenna model, the RF
       chain, Galactic noise, the ADC and the T1 trigger
   * - :doc:`api/aoi`
     - Events, antennas and showers as objects, read from files
   * - :doc:`api/analysis`
     - Reconstruction of the arrival direction, Xmax and energy
   * - :doc:`api/basis`
     - Traces, signal processing and detector layouts
   * - :doc:`api/validate`
     - The input checks behind ``GRANDlib:`` errors and warnings
   * - :doc:`api/support`
     - Logging, provenance and the bindings to the TURTLE and GULL libraries

The most used entry points are :class:`~grand.sim.efield2voltage.Efield2Voltage`,
the tree classes such as :class:`~grand.dataio.event_trees.TEfield` and
:class:`~grand.dataio.event_trees.TVoltage`,
:class:`~grand.aoi.event_list.EventList`, and the frames
:class:`~grand.geo.coordinates.Geodetic` and
:class:`~grand.geo.coordinates.GRANDCS`.  ``grand`` itself re-exports the
common names, so ``from grand import Efield2Voltage, Geodetic`` works.

.. toctree::
   :hidden:

   api/geo
   api/dataio
   api/sim
   api/aoi
   api/analysis
   api/basis
   api/validate
   api/support
