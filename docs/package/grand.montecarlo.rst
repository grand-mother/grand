:mod:`~grand.montecarlo` --- Monte Carlo helpers
================================================
.. module:: grand.montecarlo

The :mod:`~grand.montecarlo` module provides utilities for generating and
handling Monte Carlo data for the GRAND simulation. Currently it only contains
a single :class:`~grand.montecarlo.ShowerEvent` class that allows reading &
writing Monte Carlo radio shower events, e.g. generated with CoREAS or ZHAireS.


Radio shower events
-------------------


Generic shower
^^^^^^^^^^^^^^

The :class:`~grand.montecarlo.ShowerEvent` class allows 


.. autoclass:: grand.montecarlo.ShowerEvent


CoREAS shower
^^^^^^^^^^^^^

.. autoclass:: grand.montecarlo.CoreasShower
