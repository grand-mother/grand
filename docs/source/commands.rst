Command-line tools
==================

The scripts in ``scripts/`` run the simulation chain on a folder of ROOT files
without writing any Python.  Each takes the folder ``sim2root.py`` wrote
(:doc:`sim2root`) and writes its output into the same folder, where the next
step looks for it.  Run them from the repository root, in an environment set
up as in :doc:`installation`; ``--help`` lists every option.

The whole chain
---------------

From electric field to :term:`ADC` counts, on one simulation folder:

.. code-block:: bash

    python scripts/convert_efield2voltage.py my_simulation --lst 18
    python scripts/convert_voltage2adc.py my_simulation

The first writes ``voltage_*_L0_*.root``, the second ``adc_*_L1_*.root``.
The same first step from Python:

.. code-block:: python

    from grand import Efield2Voltage

    signal = Efield2Voltage("my_simulation", "voltage.root", output_directory=".", seed=1)
    signal.params["add_noise"]    = True
    signal.params["add_rf_chain"] = True
    signal.compute_voltage()

Where outputs go
~~~~~~~~~~~~~~~~

Without ``-o``, each script names its output after its input and writes it
into the simulation folder.  A bare name given with ``-o`` also goes into that
folder, unless ``-od`` names another.  An existing output is replaced.  A
folder holding e-field files at several analysis levels is read at the
highest, with a warning; ``--level`` chooses another.

``convert_efield2voltage.py``
-----------------------------

Computes the voltage at the ADC input of every :term:`detection unit <DU>`.

===================================  =====================================================
Option                               Effect
===================================  =====================================================
``--lst H``                          Local sidereal time for the Galactic noise, in hours
                                     (default 18)
``--seed N``                         Fix the noise realization
``--no_noise``                       Leave out the Galactic noise
``--no_rf_chain``                    Leave out the RF chain: the output is the
                                     open-circuit voltage
``--du_type T``                      Antenna model: ``GP300`` (HFSS, default),
                                     ``GP300_nec`` or ``GP300_mat``
``--target_duration_us T``           Pad the traces to this duration, in µs
``--add_jitter_ns S``                Add Gaussian jitter of this width to the trigger
                                     times
``--calibration_smearing_sigma S``   Smear each unit's amplitude calibration by a
                                     Gaussian of this relative width
``--level L``                        Analysis level of the e-field files to read
``-o``, ``-od``                      Output file and folder
===================================  =====================================================

``convert_voltage2adc.py``
--------------------------

Digitizes the voltages: resamples them to 500 MHz and converts them to 14-bit
ADC counts.

===================================  =====================================================
Option                               Effect
===================================  =====================================================
``--add_noise_from DIR``             Add measured noise from the ADC files in ``DIR``.
                                     The voltages should then be simulated without
                                     Galactic noise (``--no_noise`` above).
``-s N``, ``--seed N``               Fix the choice of measured noise traces
``--t1_trigger``                     Apply the offline T1 trigger and set
                                     ``trigger_flag`` for each unit
``--t1_param KEY=VALUE``             Change a T1 parameter, for example
                                     ``--t1_param th1=120``; repeatable
``-o``                               Output file
===================================  =====================================================

The T1 parameters and their defaults are those of
:func:`grand.sim.detector.trigger.t1_du_triggers`.  They are still to be
confirmed by the trigger group (:ref:`issue-t1-clean-simulations`).

``convert_efield2efield.py``
----------------------------

Turns a simulated electric field into one closer to what the hardware would
see: band-pass filtered to 50-200 MHz, resampled (``--target_sampling_rate_mhz``),
padded (``--target_duration_us``), with optional white noise
(``--add_noise_uVm``), timing jitter and calibration smearing.  This is the
step that produces the level-1 e-field files of a simulation folder.

``extract_events.py``
---------------------

Copies selected events, listed in a text file as ``directory,run,event`` lines,
from one or more data folders into a new one:

.. code-block:: text

    python scripts/extract_events.py events.txt selected/

Other scripts
-------------

``scripts/`` also holds plotting tools for the :term:`RF chain` and the noise
(``plot_rf_chain.py``, ``plot_noise.py``), and ``open_grand_file.py`` and
``open_grand_directory.py``, which open a file or folder in an interactive
session with its trees loaded.       ``sim2root/`` has its own converters,
described in :doc:`sim2root`.
