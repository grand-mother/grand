:mod:`~grand.montecarlo` --- Monte Carlo helpers
================================================
.. module:: grand.montecarlo

The :mod:`~grand.montecarlo` module provides utilities for generating and
handling Monte Carlo data for the GRAND simulation. Currently it only contains
a single :class:`~grand.montecarlo.Shower` class that allows reading & writing
Monte Carlo radio shower data, e.g. generated with CoREAS.


Radio showers
-------------


Generic shower
^^^^^^^^^^^^^^

The :class:`~grand.montecarlo.Shower` class allows 


.. autoclass:: grand.montecarlo.Shower


CoREAS shower
^^^^^^^^^^^^^

.. autoclass:: grand.montecarlo.CoreasShower
