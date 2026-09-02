API reference
=============

Generated from the docstrings.

Every one of the 554 functions and methods in the package carries a
description.  64 % also carry a ``Parameters`` section and 58 % a ``Returns``
section; the remainder are functions that take no arguments or return nothing,
where numpydoc asks for neither.  18 carry a worked example, concentrated on
the entry points where an example earns its keep.

Run ``python quality/docstring_coverage.py`` to reproduce those figures.

.. note::

   The modules below are grouped by what they are for, not by their import
   path.  All of the package's modules are here except
   ``grand/sim/noise/Compute_Plot_Galactic_Noise.py``, which is a script rather
   than a module -- it plots on import -- and would execute during the build.  ``grand.geo.gull`` and ``grand.geo.turtle`` are the ``cffi`` bindings
   to the compiled libraries and are documented here for completeness; most
   users reach them through :mod:`grand.geo.geomagnet` and
   :mod:`grand.geo.topography` instead.

Geometry and frames
-------------------

.. automodule:: grand.geo.coordinates
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: grand.geo.topography
   :members:
   :show-inheritance:

.. automodule:: grand.geo.geomagnet
   :members:

Data model
----------

The ROOT schema and the layer that reads and writes it.  See :doc:`datamodel`
for the conventions these classes assume.

.. automodule:: grand.dataio.run_trees
   :members:
   :show-inheritance:

.. automodule:: grand.dataio.event_trees
   :members:
   :show-inheritance:

.. automodule:: grand.dataio.data_tree
   :members:
   :show-inheritance:

.. automodule:: grand.dataio.descriptors
   :members:
   :show-inheritance:

.. automodule:: grand.dataio.root_files
   :members:

.. automodule:: grand.dataio.data_handling
   :members:

.. automodule:: grand.dataio.protocol
   :members:

Simulation
----------

.. automodule:: grand.sim.efield2voltage
   :members:

.. automodule:: grand.sim.detector.antenna_model
   :members:
   :show-inheritance:

.. automodule:: grand.sim.detector.process_ant
   :members:

.. automodule:: grand.sim.detector.rf_chain
   :members: db2reim, s2abcd, matmul, interpol_at_new_x, RFChain

.. automodule:: grand.sim.noise.galaxy
   :members:

.. automodule:: grand.sim.detector.adc
   :members:

.. automodule:: grand.sim.shower.gen_shower
   :members:

.. automodule:: grand.sim.shower.pdg
   :members:

Signals and traces
------------------

.. automodule:: grand.basis.traces_event
   :members:
   :show-inheritance:

.. automodule:: grand.basis.signal
   :members:

.. automodule:: grand.basis.type_trace
   :members:

.. automodule:: grand.basis.du_network
   :members:

.. automodule:: grand.basis.pipeline
   :members:

Analysis-oriented interface
---------------------------

A higher-level view of the same data: events, antennas and showers as objects,
without the caller handling ROOT trees directly.

.. automodule:: grand.aoi.event
   :members:
   :show-inheritance:

.. automodule:: grand.aoi.event_list
   :members:
   :show-inheritance:

.. automodule:: grand.aoi.antenna
   :members:

.. automodule:: grand.aoi.shower
   :members:

.. automodule:: grand.aoi.timetrace
   :members:

Reconstruction
--------------

.. warning::

   :mod:`grand.recon` is a **placeholder**.  Both classes below define a
   constructor and nothing else — there is no reconstruction algorithm in
   GRANDlib.  Direction, energy and :term:`Xmax` estimation live in separate
   collaboration code.

   The GRANDlib Handbook describes this package as "Reconstruction
   Algorithms", which overstates what is here.

.. automodule:: grand.recon.elec_field
   :members:

.. automodule:: grand.recon.params_shower
   :members:

Support
-------

.. automodule:: grand.manage_log
   :members:

.. automodule:: grand.geo.gull
   :members:

.. automodule:: grand.geo.turtle
   :members:
