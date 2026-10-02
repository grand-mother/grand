Reconstruction
==============

:mod:`grand.analysis` reconstructs arrival direction, the distance to
:term:`Xmax` and an electromagnetic-energy proxy from recorded times and
amplitudes. It needs the optional ``iminuit`` dependency
(``pip install -e ".[analysis]"``; it is in the conda environment).
Notebook 11 (see :doc:`/notebooks`) runs the chain step by step, on showers
with a known answer and on ten GP13 candidates.

.. note::

   Each fit's tests show that it recovers what its own forward model
   generates; agreement with full simulations and measured showers is not yet
   tested.

.. autosummary::

   grand.analysis.fitting.plane_wave
   grand.analysis.fitting.spherical
   grand.analysis.fitting.adf
   grand.analysis.energy_reco.voltage
   grand.analysis.signals.extraction
   grand.analysis.cramer_rao_bounds.cramer_rao
   grand.analysis.physics.cherenkov_angle
   grand.analysis.physics.atmosphere
   grand.analysis.coords.array_shower
   grand.analysis.geom.angles
   grand.analysis.geom.footprint
   grand.analysis.constants

``grand.analysis.fitting.plane_wave``
-------------------------------------

.. automodule:: grand.analysis.fitting.plane_wave
   :members:

``grand.analysis.fitting.spherical``
------------------------------------

.. automodule:: grand.analysis.fitting.spherical
   :members:

``grand.analysis.fitting.adf``
------------------------------

.. automodule:: grand.analysis.fitting.adf
   :members:

``grand.analysis.energy_reco.voltage``
--------------------------------------

.. automodule:: grand.analysis.energy_reco.voltage
   :members:

``grand.analysis.signals.extraction``
-------------------------------------

.. automodule:: grand.analysis.signals.extraction
   :members:

``grand.analysis.cramer_rao_bounds.cramer_rao``
-----------------------------------------------

.. automodule:: grand.analysis.cramer_rao_bounds.cramer_rao
   :members:

``grand.analysis.physics.cherenkov_angle``
------------------------------------------

.. automodule:: grand.analysis.physics.cherenkov_angle
   :members:

``grand.analysis.physics.atmosphere``
-------------------------------------

.. automodule:: grand.analysis.physics.atmosphere
   :members:

``grand.analysis.coords.array_shower``
--------------------------------------

.. automodule:: grand.analysis.coords.array_shower
   :members:

``grand.analysis.geom.angles``
------------------------------

.. automodule:: grand.analysis.geom.angles
   :members:

``grand.analysis.geom.footprint``
---------------------------------

.. automodule:: grand.analysis.geom.footprint
   :members:

``grand.analysis.constants``
----------------------------

.. automodule:: grand.analysis.constants
   :members:
