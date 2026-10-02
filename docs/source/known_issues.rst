Known issues
============

Problems known to affect results or to need a decision, with what to do in
the meantime.  The linked GitHub issues hold the details and the latest
status; the `open issues on GitHub
<https://github.com/grand-mother/grand/issues>`_ include problems reported
since this page was updated.  Fixed problems are in :doc:`whatsnew`.

.. contents::
   :local:
   :depth: 1

Physics and simulation
----------------------

.. _issue-vga-gain-ignored:

The VGA gain setting has no effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RFChain(vga_gain=...)`` accepts 20, 5, 0 or -5 dB, but every setting gives
the same transfer function: the stage reads a front-end-board table whatever
the gain and the per-gain tables are never opened.

*Meanwhile:* do not compare gain settings.

.. _issue-t1-clean-simulations:

The offline T1 trigger passes few units on clean simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Issue: `#233 <https://github.com/grand-mother/grand/issues/233>`_

With the default parameters, T1 rejects most clean, strong simulated pulses:
it tries only the first threshold crossing and it rejects a channel whose
crossings are more than ``t_sepmax`` apart.  Noise adds closely spaced
crossings, which is why noisy events pass more often.  Two parameters,
``sepmax_inclusive`` and ``sepmax_ends_count``, select the other readings of
the rule the trigger group has to choose between.

*Meanwhile:* offline T1 results on noise-free simulations are not meaningful.

Smaller physics defects
~~~~~~~~~~~~~~~~~~~~~~~

:Issue: `#254 <https://github.com/grand-mother/grand/issues/254>`_

* The ADF fit's geomagnetic asymmetry factor is always 1.
* At the default ``padding_factor=1.0``, the antenna response wraps around the
  end of the trace, changing the open-circuit voltage by 3 to 5%.
* The ADC truncates instead of rounding and its positive full scale is one
  count too high.
* The effective refractive index is up to 3.4% off for nearby sources and
  NaN when source and antenna are at the same altitude.

*Meanwhile:* pass ``padding_factor=2`` to
:class:`~grand.sim.efield2voltage.Efield2Voltage` where the trace shape
matters.

.. _issue-galactic-noise-normalisation:
.. _issue-galactic-noise-tables:

Voltages simulated before 7 September 2026
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before that date, the simulated noise was :math:`\sqrt{2}` too low and the
three antenna models read noise tables that differed by up to a factor of two.

*Meanwhile:* do not compare noise levels across that date.  Files written
since record the GRANDlib version (``grandlib_version``); older ones carry
``0.1.0.dev0`` or nothing.

.. _issue-geomagnetic-model-expired:

The geomagnetic model ends in 2025
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRANDlib ships IGRF-13, valid to 2025; a later date raises
``LibraryError: missing data``.  Without a date, the field of 1 January 2020
is used.

*Meanwhile:* use a date before 2025.  The fix is to ship IGRF-14.

Data and file format
--------------------

.. _issue-sample-event-times:
.. _issue-xmax-sample-vintage:

The sample simulations are dated May 1976
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Issue: `#225 <https://github.com/grand-mother/grand/issues/225>`_

The committed samples under ``sim2root/Common/`` were converted with a fixed
placeholder time, 13 May 1976.  The converters now use the simulation date;
the samples are kept as written because tests read them.

*Meanwhile:* take their date from ``event_date`` or the folder name.

.. _issue-magnetic-field-units:

``magnetic_field`` is not a vector and its unit is not recorded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TShower.magnetic_field`` holds inclination and declination in degrees, then
the strength: in µT from ZHAireS, in mT or gauss from older CoREAS
conversions.

*Meanwhile:* build the direction from the two angles, or use
:mod:`grand.geo.geomagnet`.

.. _issue-reader-directory-coupling:

The file readers depend on file names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The readers of :mod:`grand.dataio.root_files` find a file's run and shower
trees by name: the three must be separate files named ``run_*``, ``efield_*``
and ``shower_*``, with the analysis level in the name.

*Meanwhile:* keep the layout ``sim2root.py`` writes, or read the trees
directly with :mod:`grand.dataio`.

Fourteen fields change type when a file is read back
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``unsigned char`` fields of ``TADC`` and ``TRunRawVoltage`` read back as
characters: ``[3, 5]`` becomes ``['\x03', '\x05']``.

*Meanwhile:* apply ``ord()`` to each element.

Antenna positions from GPS use a fixed origin by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Issue: `#215 <https://github.com/grand-mother/grand/issues/215>`_

For GP80 data, :mod:`grand.aoi` computes GPS positions relative to a fixed
origin, about 3.8 km from the run's ``origin_geoid``.  ``EventList`` and
``Event`` accept ``gps_origin="run"`` or an explicit origin.  Which one the
GP80 data intend is for their owners to confirm.

Installation and environment
----------------------------

.. _issue-import-requires-root:

Topography, the data layer and the simulation need ROOT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``import grand`` and :mod:`grand.geo.coordinates` work without ROOT;
:mod:`grand.geo.topography` and the simulation still import the data layer
and so need it.

Not supported yet
-----------------

* Topography in the input generation of ``sim2root``: antenna and core
  positions on the terrain
  (`#142 <https://github.com/grand-mother/grand/issues/142>`_) and the
  terrain's shadow (`#141 <https://github.com/grand-mother/grand/issues/141>`_).

Documentation
-------------

.. _issue-handbook-arm-naming:

The Handbook has the X and Y antenna arms swapped
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X is the south-north arm and Y the east-west arm (:doc:`validation`).  The
Handbook says the opposite; the PDF in this documentation carries the
erratum.

Where next
----------

* :doc:`troubleshooting` for errors and surprising numbers.
* The `open issues on GitHub <https://github.com/grand-mother/grand/issues>`_.
