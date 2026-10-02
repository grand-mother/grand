Producing simulations
=====================

This page goes from a list of shower parameters to a folder of simulated
GRAND events with voltages and ADC counts.  The air shower and its radio
emission are simulated by ZHAireS or CoREAS, which you install separately;
GRANDlib provides the scripts around them.

.. note::

   Points marked **(to confirm)** are read from the scripts, not from a
   procedure the simulation group has published.  If you know the answer,
   please say so in a `GitHub issue
   <https://github.com/grand-mother/grand/issues>`_.

.. contents::
   :local:
   :depth: 1

The steps
---------

.. code-block:: text

    shower parameters
        |   ZHAireSInputGenerator.py            sim2root/ZHAireSRawRoot/
        v
    ZHAireS input (.inp)  -->  ZHAireS          outside GRANDlib
        |   ZHAireSRawToRawROOT.py  (CoREAS: CoreasToRawROOT.py)
        v
    .rawroot
        |   RunSimPipe.py                        sim2root/Common/
        |     sim2root.py                        efield, shower, run
        |     convert_efield2voltage.py          voltage   (L0)
        |     convert_voltage2adc.py             ADC counts (L1)
        v
    sim_<site>_<date>_<time>_RUN<n>_CD_<extra>_<serial>/

1. ZHAireS input files
----------------------

``ZHAireSInputGenerator.py`` writes the input files of one ZHAireS shower:

.. code-block:: bash

    cd my_showers
    python3 /path/to/grand/sim2root/ZHAireSRawRoot/ZHAireSInputGenerator.py \
        Shower0001 2212 1.2345E9 67.89 0 my_parameters.inp my_layout.dat

The arguments are:

* the event name;
* the primary as a PDG code (2212 for a proton);
* the energy in GeV, to five significant digits;
* the zenith and azimuth in degrees, giving the direction the shower comes
  from;
* the simulation parameters;
* the antenna positions.

The repository ships neither of the last two files:

* **Simulation parameters**: an Aires input file with everything except the
  event name, primary, energy, angles and seed: site, atmosphere, thinning,
  ZHAireS radio settings.  Ask the simulation group for the current one
  **(to confirm where it is kept)**.
* **Antenna positions**: one antenna per line, ``name x y z``, in meters in
  the array frame (x north, y west, z above sea level).  :doc:`sites` lists
  the layouts in the repository.

The script runs ZHAireS once without radio to find the shower maximum, draws
a random core within a 5 km hexagon and writes:

* ``<event>.EventParameters``: what was simulated, read later by
  ``sim2root``;
* ``TestArray_<event>.inp``: the input for the radio simulation, with every
  antenna of the layout, shifted to put the core at the origin.

It needs the ``ZHAireS`` executable on your ``PATH`` and stops before the
radio run, the long one: run ``ZHAireS < TestArray_Shower0001.inp`` next, on
a batch node for real showers (:doc:`at_scale`).  To change the
core position or the array name, edit the marked *user code* sections of the
script; its functions can also be imported.

Many showers
~~~~~~~~~~~~

``sim2root/Common/EventParametersGenerator.py`` writes the
``.EventParameters`` file on its own, for a library of showers drawn
another way:

.. code-block:: python

    import sys
    sys.path.append("/path/to/grand/sim2root/Common")
    from EventParametersGenerator import GenerateEventParametersFile

    core_position = [120.0, -340.0, 1264.0]      # m, array frame; z is the ground altitude
    GenerateEventParametersFile("Shower0001", 2212, 1.2345e9, 67.89, 0.0,
                                core_position, "GP300", EventWeight=1.0,
                                EventUnixTime=1700000000, OutMode="w")

``EventUnixTime`` becomes the event's date in the converted files.  Set it:
the default, 0, dates the event to 1970.

2. To RawROOT
-------------

When ZHAireS has finished, convert its output folder:

.. code-block:: bash

    python3 sim2root/ZHAireSRawRoot/ZHAireSRawToRawROOT.py Shower0001/ standard 1 1 Shower0001.rawroot

For CoREAS, ``CoreasToRawROOT.py`` does the same (:doc:`sim2root`).

3. To voltages and ADC counts
-----------------------------

``RunSimPipe.py`` runs the GRANDlib steps on a folder of ``.rawroot`` files.
Run it from its own folder:

.. code-block:: bash

    cd sim2root/Common
    python3 RunSimPipe.py /path/to/rawroot_folder MyProduction -sl GP300

It calls, in order, each step stopping at the first failure:

=============================  ===========================================================
Step                           Settings
=============================  ===========================================================
``sim2root.py``                4.096 µs traces, the pulse at 800 ns
``convert_efield2voltage.py``  Galactic noise, seed 1234, 5 ns timing jitter, 7.5%
                               calibration spread
``convert_voltage2adc.py``     500 MHz, 14 bits, no added noise
``convert_efield2efield.py``   The electric field resampled to 500 MHz, with 22 µV/m
                               noise and the same jitter and spread
=============================  ===========================================================

Two variants change the settings:

* ``RunSimPipeNoJitter.py``: no noise, jitter or spread anywhere.
* ``RunSimPipeADCNoise.py``: measured noise instead of simulated noise
  (next section), 2.048 µs traces with the pulse at 550 ns.

Whether these are the settings of the current Data Challenge is **(to
confirm)**.  For other settings, run the steps by hand with the options in
:doc:`commands`.

Measured noise instead of simulated noise
-----------------------------------------

Measured noise includes everything the detector records: the Galactic
background, the electronics and local interference.  ``convert_voltage2adc.py``
adds it to the digitized traces:

.. code-block:: bash

    python scripts/convert_efield2voltage.py my_simulation --level 0 --no_noise --seed 1234
    python scripts/convert_voltage2adc.py my_simulation \
        --add_noise_from sim2root/Common/LongNoiseTraces/ --seed 1234

``--no_noise`` in the first step leaves out the simulated Galactic noise,
which would otherwise be counted twice.

``--add_noise_from`` takes a folder of ``.root`` files, or a file-name
prefix.  Each must hold a ``TADC`` tree of measured traces with
``adc_samples_count_ch`` filled; the first three channels are taken as X, Y
and Z.  Each simulated unit gets the trace of a different measured entry,
drawn at random without replacement; ``--seed`` makes the draw repeatable.

The repository ships one such file: 638 traces of 2,048 samples (4.096 µs)
recorded by GP13 at night in February 2024 (:doc:`measured_data`).  If your
traces are longer than the noise traces, two traces of the same unit are
joined, with a warning.  If they are more than twice as long, the script
stops.  For other periods or sites, take traces from the 10-second
triggers of the period you want to model, cut to three channels: X, Y and Z
**(to confirm how the collaboration selects them)**.

Checking the result
-------------------

.. code-block:: python

    from grand.dataio import DataDirectory

    DataDirectory("sim_..._0000").print()       # the trees, levels and events

The :doc:`tutorial` reads such a folder and reconstructs the direction of its
shower.

Where next
----------

* :doc:`at_scale` for many showers on a computing cluster.
* :doc:`sim2root` for the converters.
* :doc:`commands` for every option of the scripts.
* :doc:`known_issues` before relying on the trigger or absolute noise levels.
