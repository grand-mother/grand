Known issues
============

This page lists the problems known to affect results or to need a decision,
with what to do in the meantime.  For the latest status and for problems
reported since this page was last updated, see the `open issues on GitHub
<https://github.com/grand-mother/grand/issues>`_.  Fixed problems are recorded
in the :doc:`changelog`.

.. contents::
   :local:
   :depth: 1

Physics and simulation
----------------------

.. _issue-vga-gain-ignored:

The VGA gain setting has no effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: any study that varies the amplifier gain
:Test: ``tests/sim/test_rf_chain_physics.py::test_gain_setting_changes_the_transfer_function``
       (expected failure)

``RFChain(vga_gain=...)`` accepts 20, 5, 0 or -5 dB, but every setting gives
the same transfer function: the stage named ``vgaf`` reads
``feb+amfitler+biast.s2p``, a front-end board with an AM filter and a bias
tee, whatever the gain.  The per-gain tables ``filter+vga{0,5,20}db+filter.s2p``
ship with the model data and are never opened.  There is no table for
-5 dB.  Section 8.3 of :cite:`GRAND:2024atu` describes a transfer function that
changes with the gain.

*Meanwhile:* do not compare gain settings with the default chain.  Selecting a
table per gain is for the owners of the RF-chain component configuration to
decide.

.. _issue-t1-clean-simulations:

The offline T1 trigger passes no unit on clean simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: ``convert_voltage2adc.py --t1_trigger``,
          :func:`grand.sim.detector.trigger.t1_du_triggers`
:Issue: `#233 <https://github.com/grand-mother/grand/issues/233>`_

With the default parameters, T1 passed none of the units of clean, strong
simulated events (:term:`ADC` peak 850).  On synthetic pulses there are two reasons:

* Only the first T1 crossing is tried.  It is rejected if it lies within
  the first ``t_quiet/2`` samples (256 at the default 512 ns).
* Consecutive T2 crossings must be closer than ``t_sepmax = 10`` ns, strictly
  and one wider gap rejects the channel.  A clean pulse at 60 to 200 MHz has
  its crossings 10 to 16 ns apart.  Noise adds closely spaced crossings, which
  is why noisy events sometimes pass.

Two parameters, off by default, select the other readings of the gap rule:
``sepmax_inclusive=1`` accepts a gap equal to ``t_sepmax`` and
``sepmax_ends_count=1`` ends the count at a wider gap instead of rejecting the
channel (``--t1_param sepmax_inclusive=1`` on the command line).  With either,
a clean pulse at 100 to 200 MHz counts 9 to 15 crossings, above
``nc_max = 8`` and still does not trigger.

*Meanwhile:* offline T1 results on noise-free simulations are not meaningful.
The firmware behavior is for the trigger group to confirm.

Smaller physics defects
~~~~~~~~~~~~~~~~~~~~~~~

:Issue: `#254 <https://github.com/grand-mother/grand/issues/254>`_

Four defects found in review, each with a known fix awaiting a decision:

* The ADF fit's geomagnetic asymmetry uses the magnetic field in tesla where a
  unit vector is needed, so its :math:`1/\sin\alpha` factor is always 1.
* At the default ``padding_factor=1.0``, the antenna response wraps around the
  end of the trace: the :term:`open-circuit voltage` of the sample shower differs by 3
  to 5% from the one computed with ``padding_factor=2``.
* The ADC truncates toward zero instead of rounding.  Its positive full
  scale is +8192 instead of +8191.
* The effective refractive index of :mod:`grand.analysis.physics.atmosphere`
  is up to 3.4% off for sources within 1 km horizontally and is NaN when the
  source and the antenna are at the same altitude.

*Meanwhile:* pass ``padding_factor=2`` to
:class:`~grand.sim.efield2voltage.Efield2Voltage` where the shape of the trace
matters.

.. _issue-galactic-noise-normalisation:
.. _issue-galactic-noise-tables:

Voltages simulated before 7 September 2026
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: noise levels in files simulated before that date

The Galactic-noise model was corrected on 7 September 2026.  Before that, the
simulated noise was :math:`\sqrt{2}` too low.  The three antenna models
read noise tables that differed from one another by up to a factor of two
(``GP300_nec`` and ``GP300_mat`` read the same file).

*Meanwhile:* do not compare absolute noise levels across that date and quote
the ``du_type`` with any noise level from an older file.  Voltage files record
the GRANDlib version that wrote them (``grandlib_version``); files from before
the correction carry ``0.1.0.dev0`` or no version.

.. _issue-geomagnetic-model-expired:

The geomagnetic model ends in 2025
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: the geomagnetic field and magnetic-north frames, at dates from
          1 January 2025
:Test: ``tests/geo/test_geomagnet_validity.py``

GRANDlib ships IGRF-13, which covers 1900 to 2025.  At a later date,
:mod:`grand.geo.geomagnet` raises ``LibraryError: missing data``.  The default
date in :mod:`grand.geo.coordinates` is 1 January 2020, so code that does not
pass a date works, with the 2020 field.

*Meanwhile:* use a date before 2025.  The fix is to ship IGRF-14, which covers
2025 to 2030.

Data and file format
--------------------

.. _issue-sample-event-times:
.. _issue-xmax-sample-vintage:

The sample simulations are dated May 1976
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: event times read from the four folders under ``sim2root/Common/``
:Issue: `#225 <https://github.com/grand-mother/grand/issues/225>`_
:Test: ``tests/sim2root/test_sample_event_times.py``

The simulations carry no event time.  The converters used to write a fixed
value in its place: ``core_time_s`` and ``du_seconds`` read 200854920, 13 May
1976.  The converters now use the simulation date.  The committed samples are
kept as they were written, because tests read their other values and a
regeneration also changes the :term:`Xmax` frame, the event order and the file names.

*Meanwhile:* take the date of these samples from ``event_date`` or the folder
name, not from the event time.

.. _issue-magnetic-field-units:

``magnetic_field`` is not a vector and its unit is not recorded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: anything that reads ``TShower.magnetic_field``

Both converters store ``[inclination, declination, strength]``, with the
angles in degrees and the strength in µT (:term:`ZHAireS`) or gauss (CoREAS):

==============================================  ===============================
Sample                                          ``magnetic_field``
==============================================  ===============================
``sim_Xiaodushan_..._ZHAireS_0000``             ``[61.6, 0.13, 56.482]`` (µT)
``sim_Dunhuang_..._CoREAS-NJ_0000``             ``[61.605, 0.125, 0.565]`` (G)
==============================================  ===============================

*Meanwhile:* build the field direction from the two angles, as the event
viewer does, or take the field from :mod:`grand.geo.geomagnet`.  Fixing it
changes the data format.

.. _issue-reader-directory-coupling:

The file readers depend on file names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: :mod:`grand.dataio.root_files`
:Test: ``tests/dataio/test_root_files_reader.py``

``FileEfield``, ``FileVoltage`` and their siblings do not read only the file
they are given.  They look for the run and shower trees in the same folder, by
file name, so three conditions must hold:

1. the run, electric-field and shower trees are in separate files, named ``run_*``,
   ``efield_*`` and ``shower_*``;
2. each name carries its :term:`analysis level`, as ``_L0_`` or ``_L1_``;
3. the ``analysis_level`` stored in each tree matches its name.

A file that breaks the first condition, such as one holding all three trees,
fails with ``AttributeError: 'NoneType' object has no attribute
'file_name'``.  Folders written by ``sim2root.py`` and the conversion scripts
meet all three.  ``get_du_count()`` also returns 0 on some files whose traces
hold units (an expected failure in the test).

*Meanwhile:* keep the file layout ``sim2root.py`` writes, or read the trees
directly with :mod:`grand.dataio`.

Fourteen fields change type when a file is read back
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: arithmetic on the ``unsigned char`` fields of ``TADC`` and
          ``TRunRawVoltage``
:Test: ``tests/dataio/test_tree_roundtrip.py``

PyROOT presents ``std::vector<unsigned char>`` as characters.  A field such as
``test_pulse_rate_divider`` written as ``[3, 5]`` reads back as
``['\x03', '\x05']`` and ``sum()`` over it raises ``TypeError``.  No data are
lost.

*Meanwhile:* apply ``ord()`` to each element.  Changing the declared type would
change the file format.

.. _issue-nutrig-field-names:

Two names for the NUTRIG correlation fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: merging ``dev_fix_root_warnings_lwp_new_fields`` and
          ``dev_fix_root_warnings_aoi_levels_lwp``
:Test: ``tests/dataio/test_schema_snapshot.py``

``TADC`` has ``nutrig_rhox`` and ``nutrig_rhoy``.  The branch above adds the
same quantity as ``correlation_x`` and ``correlation_y``.  Only one pair can
be part of the format; the choice is for the author and the :term:`NUTRIG` analysis.
The schema test fails if both appear.

Antenna positions from GPS use a fixed origin by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Affects: antenna positions that :mod:`grand.aoi` computes from GPS
          coordinates, for GP80 data
:Issue: `#215 <https://github.com/grand-mother/grand/issues/215>`_

By default, the positions are taken relative to a fixed origin,
``GPS_ANTENNA_ORIGIN``, rather than the run's ``origin_geoid``, about 3.8 km
away for GP80.  ``EventList`` and ``Event`` accept ``gps_origin="run"`` or an
explicit (latitude, longitude, height); ``event.antennas_origin`` records the
one used.  Which origin the GP80 data intend is for their owners to confirm.

Installation and environment
----------------------------

.. _issue-import-requires-root:

Topography, the data layer and the simulation need ROOT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Test: ``tests/test_lazy_imports.py``

``import grand`` and :mod:`grand.geo.coordinates` work without ROOT.
:mod:`grand.dataio` needs it, as expected, but so do
:mod:`grand.geo.topography` and :mod:`grand.sim.efield2voltage`, through an
import of the data layer that geometry should not need.

.. _issue-docker-unmaintained:

No Docker image is maintained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The newest published images, ``grandlib/dev:1.2`` and ``grandlib/dev:2.0``,
date from 2023 and 2022.  Nobody is responsible for them.  The Handbook
still presents Docker as the first installation route.  The full test suite
passed inside ``grandlib/dev:1.2`` and inside an image built from
``env/docker/grandlib.dockerfile`` (:doc:`installation`), so the route works.
Whether the collaboration publishes and maintains an image is undecided.

Not supported yet
-----------------

* Topography in the input generation of ``sim2root``: antenna and core
  positions on the terrain
  (`#142 <https://github.com/grand-mother/grand/issues/142>`_) and the
  terrain's shadow (`#141 <https://github.com/grand-mother/grand/issues/141>`_).
* A test suite free of tests that cannot fail and of tests that depend on
  untracked files (`#271 <https://github.com/grand-mother/grand/issues/271>`_).

Documentation and repository
----------------------------

.. _issue-handbook-arm-naming:

The Handbook has the X and Y antenna arms swapped
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Test: ``tests/sim/test_antenna_arm_identity.py``

In the NEC and MATLAB effective-length tables, X is the south-north arm and Y
the east-west arm, as the correlation of their patterns with the named HFSS
arms shows.  The GRANDlib Handbook says the opposite.  The code is correct;
the PDF in this documentation carries the erratum.

.. _issue-src-outlib-conflict:

``src_outlib/`` is a stale copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``src_outlib/ZHAireSRawToGRANDROOT.py`` contains merge-conflict markers
committed in 2023 and is not valid Python.
``src_outlib/AiresInfoFunctionsGRANDROOT.py`` is an older, diverged copy of
``sim2root/ZHAireSRawRoot/AiresInfoFunctionsGRANDROOT.py``.  Nothing imports
the directory.  Edit the files under ``sim2root/`` instead.  The directory
will be removed once the branches that still modify it are merged.

Where next
----------

* :doc:`troubleshooting` for errors and surprising numbers.
* The `open issues on GitHub <https://github.com/grand-mother/grand/issues>`_.
