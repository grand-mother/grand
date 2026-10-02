Cheat sheet
===========

The conventions and calls used most, on one page to keep at hand or print.

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

The antenna model's frequency axis is in **Hz**; the reconstruction takes
times in **seconds**.

Conventions
-----------

* **Array frame** (``GRANDCS``, ``du_xyz``): x north (magnetic), y west, z up,
  in meters from ``TRun.origin_geoid``.
* **Directions** say where the shower **comes from**: zenith 0° is vertical,
  azimuth is measured from north toward west.
* **Traces** have one row per unit and one column per **arm**: X
  (south-north), Y (east-west), Z (vertical).  Arms are not field components.
* **Heights** need a reference: the ellipsoid and sea level differ by 61 m at
  the GRANDProto300 site.

Files
-----

A simulation folder, as ``sim2root.py`` writes it, holds one tree per file::

    run_<run>_L<level>_<serial>.root              TRun
    shower_<events>_L<level>_<serial>.root        TShower
    efield_<events>_L<level>_<serial>.root        TEfield
    voltage_<events>_L0_<serial>.root             TVoltage (convert_efield2voltage.py)
    adc_<events>_L1_<serial>.root                 TADC (convert_voltage2adc.py)

Level 0 holds the simulation; level 1 what the detector would record.

Common calls
------------

.. code-block:: python

    from grand import Efield2Voltage, ADC, Geodetic, GRANDCS
    from grand.dataio import TShower, TEfield, TRun, DataDirectory
    from grand.aoi.event_list import EventList

    # Read a tree: open, select, read fields; "with" closes the file
    with TShower("shower_1618-13790_L0_0000.root") as tshower:
        tshower.get_event(13790, 1)                  # event number, then run
        print(tshower.zenith, tshower.energy_primary)

    # List what a file holds
    tshower.get_list_of_events()                     # [(event, run), ...]

    # Every tree of a folder, by type and level
    folder = DataDirectory("my_simulation")
    folder.tefield_l0

    # Events with their antennas and shower
    for event in EventList("my_simulation"):
        print(event.event_number, len(event.antennas))

    # Simulate voltages, then digitize
    sim = Efield2Voltage("my_simulation", "voltage.root", seed=1)
    sim.params["lst"] = 18.0                         # sidereal time of the noise, h
    sim.compute_voltage()
    adc = ADC()
    counts = adc.process(adc.downsample(voltage_uv, 2000.0))  # voltage_uv: (units, 3, samples), µV, 2 GHz

    # Positions
    site = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)
    Geodetic(GRANDCS(x=1000.0, y=0.0, z=0.0, location=site))

From the shell:

.. code-block:: bash

    python scripts/convert_efield2voltage.py my_simulation --lst 18 --seed 1
    python scripts/convert_voltage2adc.py my_simulation --t1_trigger

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
