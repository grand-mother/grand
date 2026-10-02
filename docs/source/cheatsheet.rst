Cheat sheet
===========

The units, conventions and calls used most, on one page to keep at hand or
print.

.. _units:

Units
-----

=====================  ============  ==========================  ===============
Quantity               Unit          Quantity                    Unit
=====================  ============  ==========================  ===============
Electric field         µV/m          Position, distance          m
Voltage                µV            Angle (trees, simulation)   degrees
ADC trace              counts        Angle (``grand.analysis``)  radians
One ADC count          109.9 µV      Energy                      GeV
Time in a trace        ns            Xmax depth                  g/cm²
Frequency              MHz           Event time                  Unix s + ns
=====================  ============  ==========================  ===============

Two exceptions: the frequency axis of the antenna model is in **Hz**; the
reconstruction (:mod:`grand.analysis`) takes times in **seconds** and gives
angles in radians.  Tree fields state their unit in :doc:`data_format`;
function arguments name it when they can, as ``freqs_mhz`` or ``dt_ns``.

Conventions
-----------

These cause most silent errors; :doc:`coordinates` shows each one in code.

* **Array frame** (``GRANDCS``, ``du_xyz``): x north (magnetic), y west, z up,
  in meters from ``TRun.origin_geoid``.
* **Directions** say where the shower **comes from**: zenith 0° is vertical,
  azimuth is measured from north toward west.
* **Traces** have one row per unit and one column per **arm**: X
  (south-north), Y (east-west), Z (vertical).  Arms are not field components:
  an arm's voltage is the field projected on that arm's :term:`effective
  length`, so the ratio between arms is not the ratio between field
  components.
* **Heights** need a reference: the ellipsoid and sea level differ by up to
  100 m (61 m at the GRANDProto300 site).

Files
-----

A simulation folder is named ``sim_<site>_<date>_<time>_RUN<run>_CD_<label>_<serial>``,
where ``<label>`` is the ``-e`` option of ``sim2root.py``.  It holds one tree
per file::

    run_<run>_L<level>_<serial>.root              TRun
    shower_<events>_L<level>_<serial>.root        TShower
    efield_<events>_L<level>_<serial>.root        TEfield
    voltage_<events>_L<level>_<serial>.root       TVoltage (convert_efield2voltage.py)
    adc_<events>_L<level+1>_<serial>.root         TADC (convert_voltage2adc.py)

``<events>`` is the lowest and highest event number.  The :term:`analysis
level` ``L0`` is the simulation: the simulator's electric field and the
voltages computed from it.  ``L1`` is what the detector would record: the ADC
counts, plus the electric field filtered, resampled and given noise by
``convert_efield2efield.py``.

Common calls
------------

On a simulation folder written by ``sim2root.py``:

.. code-block:: python

    import numpy as np
    from grand import Efield2Voltage, ADC, Geodetic, GRANDCS
    from grand.dataio import TShower, TVoltage, DataDirectory
    from grand.aoi import EventList

    # Every tree of a folder, by type and level
    folder = DataDirectory("my_simulation")
    tshower = folder.tshower_l0

    # Read a tree: list its events, select one, read its fields
    events = tshower.get_list_of_events()            # [(event, run), ...]
    tshower.get_event(*events[0])                    # event number, then run
    print(tshower.zenith, tshower.energy_primary)

    # Events with their antennas and shower
    for event in EventList("my_simulation"):
        print(event.event_number, len(event.antennas))

    # Simulate voltages from the level-0 electric field
    sim = Efield2Voltage("my_simulation", "voltage.root", output_directory="out",
                         seed=1, efield_level=0)
    sim.params["lst"] = 18.0                         # sidereal time of the noise, h
    sim.compute_voltage()

    # Read them back; "with" closes the file at the end of the block
    with TVoltage("out/voltage.root") as tvoltage:
        tvoltage.get_entry(0)
        voltage_uv = np.asarray(tvoltage.trace)      # (units, 3 arms, samples), µV

    # Digitize: resample from 2000 MHz to the ADC's 500 MHz, then convert
    adc = ADC()
    counts = adc.process(adc.downsample(voltage_uv, 2000.0))

    # Positions
    site = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)
    Geodetic(GRANDCS(x=1000.0, y=0.0, z=0.0, location=site))

From the shell, the same simulation written into the folder, then digitized:

.. code-block:: bash

    python scripts/convert_efield2voltage.py my_simulation --level 0 --lst 18 --seed 1
    python scripts/convert_voltage2adc.py my_simulation --t1_trigger

.. _cheatsheet-symptoms:

When something looks wrong
--------------------------

=========================================  =======================================
Symptom                                    Usual cause
=========================================  =======================================
A position lands outside the array         ``GRANDCS`` and ``LTP`` axes mixed up
An angle is off by a factor of 57.3        degrees and radians mixed up
A frequency is off by :math:`10^6`         the antenna model's axis is in Hz
The three arms do not match the field      arms are not field components
Elevation is ``nan``                       terrain tile not downloaded
All events in a list are the same          ``EventList`` reuses one ``Event``
=========================================  =======================================

:doc:`troubleshooting` has more, including every common error message.
