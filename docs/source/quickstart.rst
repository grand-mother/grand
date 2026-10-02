Quick start guide
=================

This page goes from an installed GRANDlib to the operations most users need:
simulating the voltages of a shower, reading them back, digitizing them and
running the stages of the chain on their own.  Every block on this page is
executed when the documentation is built, in order, on the shower that ships
with the repository.

Your first voltage
------------------

GRANDlib starts from the electric field that :term:`ZHAireS` or CoREAS computed at
each antenna, converted to GRAND's format by ``sim2root`` (:doc:`sim2root`).
The result of that conversion is a folder of ROOT files: ``efield_*`` holds the
traces, ``shower_*`` the shower and ``run_*`` the detector layout.  The
repository includes one such folder, a 3.9 EeV proton shower at
zenith 79.4° seen by 44 :term:`GRANDProto300 <GP300>` antennas:

.. jupyter-execute::

    import tempfile
    from pathlib import Path

    import numpy as np
    import grand
    from grand import Efield2Voltage

    REPO = Path(grand.__file__).parents[1]
    SAMPLE = REPO / "sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"
    out = Path(tempfile.mkdtemp())

    sim = Efield2Voltage(str(SAMPLE), "voltage.root", output_directory=str(out),
                         seed=1, efield_level=0)
    sim.compute_voltage()
    print(sorted(p.name for p in out.iterdir()))

:meth:`~grand.sim.efield2voltage.Efield2Voltage.compute_voltage` runs every
event in the folder through the antenna response, adds Galactic noise for a
:term:`local sidereal time <LST>` of 18 h, applies the GRANDProto300 :term:`RF chain` and writes the
voltage at the :term:`ADC` input to ``voltage.root``.  ``seed`` fixes the noise;
without it, every run draws a new realization.  ``efield_level=0`` picks the
simulated fields, since this folder also holds a level-1 copy.

Reading the result
------------------

Each file holds one ROOT tree and each tree class reads one kind of file.
:class:`~grand.dataio.event_trees.TVoltage` gives the traces as one row per
:term:`detection unit <DU>`, three arms per row:

.. jupyter-execute::

    from grand.dataio import TVoltage

    with TVoltage(str(out / "voltage.root")) as tvoltage:
        tvoltage.get_entry(0)
        run, event = tvoltage.run_number, tvoltage.event_number
        du_id = np.asarray(tvoltage.du_id)
        traces = np.asarray(tvoltage.trace)        # (units, arms, samples), in µV

    print("run %d, event %d: %d units, %d samples each"
          % (run, event, len(du_id), traces.shape[2]))

    peak = np.abs(traces).max(axis=2)              # (units, arms)
    best = np.argmax(peak.max(axis=1))
    print("strongest unit %d: peak %.0f, %.0f, %.0f µV on arms X, Y, Z"
          % (du_id[best], *peak[best]))

The traces are sampled every 0.5 ns, the rate of the input simulation:

.. jupyter-execute::

    import matplotlib.pyplot as plt

    t_ns = 0.5 * np.arange(traces.shape[2])
    fig, ax = plt.subplots(figsize=(8, 3))
    for arm, name in enumerate("XYZ"):
        ax.plot(t_ns, traces[best, arm] / 1e3, lw=0.8, label="arm " + name)
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("voltage [mV]")
    ax.set_xlim(1200, 2400)
    ax.legend()
    plt.show()

The ``with`` form closes the file when the block ends.  When a script reads
many files, use it, or call ``stop_using()`` on each tree, so that memory does
not grow with the number of files (:ref:`datamodel-releasing-trees`).

.. _quickstart-units:

Units
-----

GRANDlib uses the units below throughout.  A tree field that has a unit says
so in its documentation and functions name it in the argument when they can
(``freqs_mhz``, ``dt_ns``).

======================================  ==========================================
Quantity                                Unit
======================================  ==========================================
Electric field                          µV/m
Voltage                                 µV
Digitized trace                         ADC counts (one count is 109.9 µV)
Time                                    ns; event times as Unix seconds plus ns
Frequency                               MHz
Position, distance                      m
Angle (trees, coordinates, simulation)  degrees
Angle (:mod:`grand.analysis`)           radians
Energy                                  GeV
Atmospheric depth (Xmax)                g/cm²
======================================  ==========================================

.. important::

   **Two exceptions.**  The frequency axis that
   :class:`~grand.sim.detector.antenna_model.AntennaModel` loads is in **Hz**,
   not MHz; divide by ``1e6`` before passing it to the RF chain.  And the
   reconstruction package :mod:`grand.analysis`, with the ``*_pwf`` and
   ``*_swf`` fields of ``TRecons``, works in **radians**; it warns when given
   an angle larger than 2π.

Conventions
-----------

These are the ones that most often cause silent errors.  :doc:`coordinates`
shows each of them executing.

*Directions say where the shower comes from.*  A vertical shower has zenith 0°.
Azimuth is measured in the array frame, from north toward west, so 90° is a
shower from the west.

*The array frame is north-west-up.*  In ``GRANDCS``, x points to magnetic
north, y to west and z up.  A local ``LTP`` frame with ``orientation='ENU'``
takes the same three numbers and means east, north, up.

*The three channels are antenna arms, not field components.*  ``trace[:, 0]``
is the south-north arm (X), ``trace[:, 1]`` the east-west arm (Y) and
``trace[:, 2]`` the vertical arm (Z).  The voltage on an arm is the projection
of the field on that arm's :term:`effective length`, so the ratio between arms is not
the ratio between field components.

*A height needs a reference.*  The ellipsoid and the geoid (mean sea level)
differ by up to 100 m; at the GRANDProto300 site the geoid is 61 m below the
ellipsoid.

Terms used throughout
---------------------

*Detection unit* (DU): one antenna station, with its three arms, RF chain and
ADC.  ``du_id`` identifies it.

*Run* and *event*: an event is one candidate air shower, a run a set of events
recorded with the same configuration.  ``run_number`` and ``event_number``
together identify an event.

*Analysis level*: the ``L0``, ``L1``, ... in a file name, also stored in its
trees.  In a simulation, level 0 holds the converted simulator output and the
voltages computed from it; level 1 holds what the detector would record, such
as ADC counts.

*Effective length*: the antenna's response to an incoming field, a vector that
depends on direction and frequency.

*RF chain*: the analog electronics between the antenna and the ADC.

*LST*: local sidereal time, which sets how much of the Galaxy is above the
horizon and therefore the noise level.

Switching stages off
--------------------

``sim.params`` turns the stages on and off before
:meth:`~grand.sim.efield2voltage.Efield2Voltage.compute_voltage` runs.  With
neither noise nor RF chain, the output is the :term:`open-circuit voltage` at the
antenna terminals:

.. jupyter-execute::

    voc = Efield2Voltage(str(SAMPLE), "voc.root", output_directory=str(out),
                         seed=1, efield_level=0)
    voc.params["add_noise"] = False
    voc.params["add_rf_chain"] = False
    voc.compute_voltage()

    with TVoltage(str(out / "voc.root")) as tvoltage:
        tvoltage.get_entry(0)
        voc_traces = np.asarray(tvoltage.trace)

    print("open-circuit peak on unit %d: %.0f, %.0f, %.0f µV"
          % (du_id[best], *np.abs(voc_traces[best]).max(axis=1)))

The other entries are ``lst`` (the sidereal time, in hours), the sampling rate
and trace length of the output (``resample_to_mhz``, ``extend_to_us``),
calibration smearing and timing jitter.
:class:`~grand.sim.efield2voltage.Efield2Voltage` lists them with their
defaults.

Digitizing
----------

The ADC samples at 500 MHz with 14 bits.  Downsample the 2 GHz voltages first,
then convert:

.. jupyter-execute::

    from grand import ADC

    adc = ADC()
    counts = adc.process(adc.downsample(traces, 2000.0))
    print("shape", counts.shape, "- peak on unit %d: %d, %d, %d counts"
          % (du_id[best], *np.abs(counts[best]).max(axis=1)))

From the shell, the same two steps write a voltage file and an ADC file into
the simulation folder:

.. code-block:: bash

    python scripts/convert_efield2voltage.py <simulation folder> --lst 18 --seed 1
    python scripts/convert_voltage2adc.py <simulation folder>

:doc:`commands` describes both scripts and their options.

One stage at a time
-------------------

Every stage can also run on arrays, with no file involved.  The RF chain's
transfer function, from the open-circuit voltage to the ADC input, at
frequencies in MHz:

.. jupyter-execute::

    from grand.sim.detector.rf_chain import RFChain

    freqs_mhz = np.arange(30.0, 251.0)
    chain = RFChain()
    chain.compute_for_freqs(freqs_mhz)
    gain = np.abs(chain.get_tf())                  # (arms, frequencies)
    print("peak |V_out / V_oc| per arm: %.1f, %.1f, %.1f" % tuple(gain.max(axis=1)))

and one realization of Galactic noise for four units at 18 h LST:

.. jupyter-execute::

    from grand.sim.noise.galaxy import galactic_noise

    noise = galactic_noise(18.0, 1024, freqs_mhz, nb_ant=4, seed=0)
    print("spectrum shape (units, arms, frequencies):", noise.shape)

Positions
---------

Frames are constructed from one another.  A point 1 km along the array's x
axis, as latitude, longitude and height:

.. jupyter-execute::

    from grand import Geodetic, GRANDCS

    site = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)
    point = GRANDCS(x=1000.0, y=0.0, z=0.0, location=site)
    print(np.round(np.asarray(Geodetic(point)).ravel(), 5))

The latitude increased: x points north.

Where next
----------

* :doc:`recipes`: short code for the common tasks, from reading a shower to
  reconstructing its direction.
* :doc:`notebooks`: twelve notebooks that work through each part of the
  library with figures.
* :doc:`datamodel`: the trees, what they hold and how they fit together.
* :doc:`troubleshooting`: what to do when a number looks wrong.
