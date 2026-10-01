Quickstart
==========

The canonical operation: take an electric field produced by an air-shower
simulation, and compute the voltage a detection unit would record.

Today
-----

After Listing 4 of `arXiv:2408.10926 <https://arxiv.org/abs/2408.10926>`_,
which passes a single e-field file.  The input is now the simulation folder
``sim2root.py`` writes (``efield_*``, ``run_*`` and ``shower_*`` files), since
the run and shower trees are needed too:

.. code-block:: python

    from grand import Efield2Voltage

    signal = Efield2Voltage("my_simulation", "voltage.root", output_directory=".", seed=1)
    signal.params["add_noise"]    = True
    signal.params["add_rf_chain"] = True

    signal.compute_voltage()      # writes ./voltage.root as a side effect

or, equivalently, from a shell::

    python scripts/convert_efield2voltage.py my_simulation -o voltage.root -od .

.. note::

   This block is marked ``code-block``, not ``jupyter-execute``, and that is
   the problem.  A ``jupyter-execute`` block runs when the page is built, so
   it cannot go stale; but this one cannot run, because it needs a ROOT file
   on disk before the object can even be constructed.  Every example in this
   library has that shape today, which is why only 17 of the 702 functions
   carry one.

After Phase 6
-------------

.. warning::

   **Not runnable.**  The code in this section is a design sketch: the
   modules it imports (``grand.config``, ``grand.io``, ``grand.sim.kernel``)
   do not exist yet.  Use the form above.

Once input, processing and output are delineated, the same operation
separates into steps that can each be demonstrated on their own:

.. code-block:: python

    from grand.config import Site, DetectorUnit, AntennaModel, RFChain, ADC, NoiseModel
    from grand.io import read_efield, write_voltage
    from grand.sim import Efield2Voltage

    du    = DetectorUnit(antenna=AntennaModel.load("HorizonAntenna/hfss"),
                         chain=RFChain.load("GP300/v2", vga_gain_db=20),
                         adc=ADC(n_bits=14, sampling_mhz=500))
    noise = NoiseModel.galactic(table="LFmap/gp13_GP300", lst_hour=18.0)

    run  = read_efield("input_efield.root")
    volt = Efield2Voltage(detector=du, noise=noise,
                          site=Site.load("dunhuang")).run(run)
    write_voltage("output_voltage.root", volt)

and the physics underneath becomes callable without any file at all:

.. code-block:: python

    from grand.sim.kernel import effective_length, voc_from_efield, apply_chain

    leff = effective_length(du.antenna, theta_deg=85.0, phi_deg=0.0,
                            freqs_mhz=freqs_mhz)
    voc  = voc_from_efield(efield_v_per_m, leff, freqs_mhz)
    vout = apply_chain(du.chain.transfer_function(freqs_mhz), voc)

That last form is what makes an executable example possible: arrays in,
arrays out, no filesystem.

Why it matters beyond documentation
-----------------------------------

Comparing detector configurations is one of the library's stated purposes.
After Phase 6 it is a three-line sweep:

.. code-block:: python

    for gain in (0, 5, 20):
        variant = replace(du, chain=RFChain.load("GP300/v2", vga_gain_db=gain))
        results[gain] = Efield2Voltage(detector=variant, noise=noise, site=site).run(run)

Today the same comparison means mutating a dictionary, re-instantiating an
object that re-reads the input file and rebuilds three RF chains, and writing
three files to compare.
