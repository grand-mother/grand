Working with measured data
==========================

This page is for reading what GRAND detectors record: GP13, GP80 and
GRANDProto300 files.  It covers what the files hold, how to read the traces,
times and positions, how to convert counts to volts and how to tell the kind
of trigger that recorded an event.  The code runs when this page is built, on
a GP13 sample that ships with the repository.

.. note::

   A few points, marked **(to confirm)**, come from the code and the examples
   rather than from a specification of the data; the last section lists them.

.. contents::
   :local:
   :depth: 1

What a measured file holds
--------------------------

The DAQ writes binary files.  The collaboration's converter, ``gtot``,
turns them into ROOT files in GRANDlib's format; GRANDlib reads those but
does not run ``gtot`` itself (:doc:`at_scale`).  A converted file holds:

==================  ==============================================================
Tree                Holds
==================  ==============================================================
``TADC``            The traces in ADC counts, four channels per unit, with the
                    trigger, GPS and housekeeping data of each unit
``TRawVoltage``     The same traces converted to µV by ``gtot``
``TRun``            What the run covers: site, units, sampling
``TRunRawVoltage``  The firmware settings: thresholds, gains, channel mapping
==================  ==============================================================

File names follow ``SITE_DATE_TIME_RUNn_MODE_...root``, for example
``GP80_20250711_191026_RUN10127_CD_20dB-GP65-Y2float-60DUs-CD-100000-3619.root``.
Files whose name contains ``-10s-`` hold the 10-second triggers; the
production pipeline monitors the detector with them.  The older GP13 files
are named ``GRAND.TEST-RAW.20230307174423.001.root``.

Getting the data
----------------

The converted files are kept at CC-IN2P3, the computing center of CNRS/IN2P3
in Lyon, by site, year and month: for example
``/sps/grand/data/gp80/GrandRoot/2025/07/``.  Reading them needs a CC-IN2P3
account with access to GRAND's storage; your GRAND group leader can tell you
how to request one.

With an account, work on the CC-IN2P3 machines, or copy the files you need
with ``scp`` or ``rsync`` from ``cca.in2p3.fr``.  ``granddb``, GRAND's data
catalog, can find files by name and copy them for you; notebook 10
(:doc:`notebooks`) shows how.

Without an account, use the GP13 sample in the repository, as this page does.

Opening a file
--------------

:class:`~grand.dataio.DataFile` opens a file, a list of files or a wildcard;
each tree becomes an attribute.  The sample holds 638 background traces
recorded by seven GP13 units at night in February 2024, cut to three
channels:

.. jupyter-execute::

    from pathlib import Path

    import numpy as np
    import grand
    from grand.dataio import DataFile

    SAMPLE = (Path(grand.__file__).parents[1] / "sim2root/Common/LongNoiseTraces"
              / "noice_traces_merged_gp13_2024_02_night_datafiles_410-415.root")

    f = DataFile(str(SAMPLE))
    tadc = f.tadc
    print(tadc.get_number_of_entries(), "entries")

    tadc.get_entry(0)
    counts = np.asarray(tadc.trace_ch)          # (units, channels, samples), ADC counts
    print(counts.shape, "from unit", tadc.du_id)

Each entry is one event; ``du_id`` lists the units that recorded it and the
first axis of ``trace_ch`` follows the same order.

Channels
--------

A GP80 or GP300 unit records four channels.  The analysis examples read
channels 1, 2 and 3 as the X (south-north), Y (east-west) and Z (vertical)
arms; channel 0 is the floating arm **(to confirm for each site and
period)**.  GP80 file names state which arm floats, as ``Y2float`` above.
The run's ``TRunRawVoltage.adc_input_channels_ch`` records how the ADC
inputs were wired.  The sample above was cut to three channels, X, Y and Z.

From counts to volts
--------------------

The ADC maps ±0.9 V to ±8192 counts, so one count is about 110 µV:

.. jupyter-execute::

    microvolt = counts * 0.9e6 / 8192
    print("noise RMS per arm [µV]:", microvolt[0].std(axis=-1).round())

For files converted by ``gtot``, ``TRawVoltage.trace_ch`` already holds the
µV and ``TRunRawVoltage.adc_conversion`` records the factor it used.  The
reverse, µV to counts, is
:func:`~grand.analysis.signals.extraction.convert_voltage_to_ADC`.

Times
-----

``du_seconds`` is the GPS time of each unit's trigger, converted to Unix
seconds; ``du_nanoseconds`` is the fraction of it.  ``trigger_position`` is
the sample at which the unit triggered; at 500 MHz a sample lasts 2 ns:

.. jupyter-execute::

    from datetime import datetime, timezone

    print(datetime.fromtimestamp(tadc.du_seconds[0], timezone.utc),
          "+ %d ns" % tadc.du_nanoseconds[0])

To compare units within an event, subtract the smallest ``du_seconds``
before adding the nanoseconds, as in step 2 of the :doc:`tutorial`:
floating-point seconds since 1970 lose the nanoseconds.

Positions
---------

Each unit reports its GPS position with every event.  In ``TADC`` the
latitude, longitude and altitude are 64-bit floating-point numbers stored in
integer fields: reinterpret the bits, do not convert the value.  Latitude
and longitude are then in radians and altitude in meters **(to confirm)**:

.. jupyter-execute::

    lat, lon, alt = np.array([tadc.gps_lat, tadc.gps_long, tadc.gps_alt],
                             dtype=np.uint64).view(np.float64)[:, 0]
    print("latitude %.5f deg, longitude %.5f deg, altitude %.0f m"
          % (np.degrees(lat), np.degrees(lon), alt))

A single GPS fix is accurate to meters.  For timing analyses use surveyed
positions: ``examples/analysis/`` uses RTK positions of the GP80 units.  To
put positions in the array frame, see :doc:`coordinates` and :doc:`sites`.

Which trigger recorded the event
--------------------------------

========================================  ===============================================
Field                                     Meaning
========================================  ===============================================
``event_type``                            ``0x1000``: 10-second trigger; ``0x8000``:
                                          random trigger; anything else: a shower
                                          candidate
``trigger_pattern_10s``                   Per unit: recorded by the 10-second trigger
``trigger_pattern_20Hz``                  Per unit: recorded by the 20 Hz random trigger
``trigger_pattern_ch0_ch1`` and others    Per unit: the channel combination that
                                          fired the self-trigger
``trigger_pattern_calibration``,          Per unit: calibration and test pulses
``trigger_pattern_external_test_pulse``
========================================  ===============================================

Use the per-unit patterns when they disagree with ``event_type``: every
entry of the sample has ``event_type`` 0 but was recorded by the 10-second
trigger:

.. jupyter-execute::

    ten_second = 0
    for entry in range(tadc.get_number_of_entries()):
        tadc.get_entry(entry)
        ten_second += all(tadc.trigger_pattern_10s)
    print(ten_second, "of", tadc.get_number_of_entries(), "entries are 10-second triggers")
    f.close()

10-second and random triggers record the background: use them for noise
studies, not as shower candidates.  Which ``event_type`` values each
firmware version writes is **(to confirm)**.

From events to a reconstruction
-------------------------------

:class:`~grand.aoi.event_list.EventList` reads a whole file event by event,
with the units' positions and traces together:

.. code-block:: python

    from grand.aoi import EventList

    events = EventList("GP80_..._CD_...root", use_trawvoltage=True,
                       trawvoltage_channels=[1, 2, 3])        # X, Y, Z
    event = events.get_event(event_number=12, run_number=10127)
    for voltage in event.voltages:
        print(voltage.t0, voltage.trace.x.max())      # trigger time, X arm in µV

Pass ``trawvoltage_channels``: the default, ``[0, 1, 2]``, matches
simulated files, not the four-channel measured ones.  ``EventList`` reads
voltages, so it does not open the GP13 sample above, which holds ADC counts
only.

``examples/analysis/main_AOI.py`` continues from there to peak times,
amplitudes and the fits of :mod:`grand.analysis`, on the GP80 events listed
in ``examples/analysis/Flagged_events_July_October.txt``.  It needs the GP80
data from CC-IN2P3.

To confirm
----------

These need an answer from the people who run the detectors.  If you know
one, please say so in a `GitHub issue
<https://github.com/grand-mother/grand/issues>`_.

* The channel-to-arm mapping for each site and period.
* The units and storage of ``gps_lat``, ``gps_long`` and ``gps_alt`` in
  ``TADC`` and ``TRawVoltage``.
* The ``event_type`` values each firmware version writes.
* The meaning of the ``MODE`` field of the file names (``CD``, ``MD``,
  ``TR``).

Where next
----------

* :doc:`data_format` for every field of ``TADC`` and ``TRawVoltage``.
* :doc:`simulation_production` to add this kind of noise to simulations.
* Notebooks 11 and 12 (:doc:`notebooks`) for reconstruction and the event
  viewer on GP13 data.
