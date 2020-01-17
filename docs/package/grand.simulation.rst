:mod:`~grand.simulation` --- Simulation helpers
===============================================
.. module:: grand.simulation

The :mod:`~grand.simulation` module provides utilities for the GRAND simulation.
Currently it only contains a single :class:`~grand.simulation.ShowerEvent` class
that allows reading & writing Monte Carlo radio shower events, e.g. generated
with CoREAS or ZHAireS.


Radio shower events
-------------------


Generic shower
^^^^^^^^^^^^^^

The :class:`~grand.simulation.ShowerEvent` class allows 


.. autoclass:: grand.simulation.ShowerEvent


CoREAS shower
^^^^^^^^^^^^^

.. autoclass:: grand.simulation.CoreasShower
