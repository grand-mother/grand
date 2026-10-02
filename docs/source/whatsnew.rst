What's new
==========

The main changes for users, newest first.  The :doc:`changelog` lists every
change in detail.

``dev-next``, not yet released
------------------------------

Compared with the ``dev`` branch of 2025.

Changes that alter results
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Galactic noise is** :math:`\sqrt{2}` **larger.**  Since 7 September 2026
  the simulated noise has the level its tables specify; before, it was
  :math:`\sqrt{2}` too low.  The three antenna models (``GP300``,
  ``GP300_nec``, ``GP300_mat``) now each read their own table.  Voltage files
  record the GRANDlib version that wrote them, so files from before the
  change can be told apart (:ref:`issue-galactic-noise-normalisation`).
* **No antenna response from below the horizon.**  A source more than 90°
  from the zenith used to read the response of a direction above the antenna,
  at full strength.  It now gives zero, with a warning.
* **CoREAS conversions have the right azimuth.**  The converter mirrored the
  azimuth of inputs without a shower block; files converted from those should
  be converted again.
* **sim2root writes the right antenna coordinates.**  With several events
  sharing antennas, ``du_geoid`` could be assigned to the wrong antennas, up to
  8 km off.
* **The resampled voltage is consistent.**  Resampling in
  ``Efield2Voltage`` used to leave the trigger position and the run's time
  step at the input rate, so the :term:`ADC` step misread the traces.  Resample the
  electric field with ``convert_efield2efield.py`` instead.

New
~~~

* **Reconstruction**, in :mod:`grand.analysis`: arrival direction, the
  position of :term:`Xmax` and an energy estimate from recorded times and amplitudes,
  with Cramér-Rao bounds (notebook 11).
* **An offline T1 trigger**, ``convert_voltage2adc.py --t1_trigger``
  (:doc:`commands`).
* **Files record the code that wrote them**: the GRANDlib version and git
  commit, in every tree written.
* **Clear errors.**  Public functions check their input and say what is wrong
  in a message that starts with ``GRANDlib:`` (:doc:`troubleshooting`).
* **One installation route**: a single conda environment and an installable
  package (:doc:`installation`).
* **Documentation**: this site, with a quick start, recipes and twelve
  notebooks and ``notebooks/import_notebook.py`` to add or edit a notebook.

Other changes
~~~~~~~~~~~~~

* ``import grand`` no longer needs ROOT.  Reading files no longer loads the
  simulation code.
* Trees are released when no longer referenced, so reading many files no
  longer exhausts memory (:ref:`datamodel-releasing-trees`).
* Two processes writing the same file are refused instead of corrupting it.
* The tree classes work with NumPy 2.

Removed
~~~~~~~

* ``grand.recon``, which held only placeholders; use :mod:`grand.analysis`.
* ``du_type='Horizon'``, whose antenna model files are not in the model data.
