Tutorial notebooks
==================

Worked notebooks live in `notebooks/
<https://github.com/grand-mother/grand/tree/dev-next/notebooks>`_, numbered in
reading order.  Each carries its figures inline, so they can be read on GitHub
without being run.

They are the long form of the :doc:`recipes` and the narrative pages: a
recipe shows a task in a few lines, and a notebook works through it with the
reasoning, the edge cases and the checks behind the numbers.

To run them rather than read them::

    conda env create -f env/conda/grand-dev.yml --solver=libmamba
    conda activate grand-dev
    source env/setup.sh
    jupyter lab notebooks/

Run them from the ``notebooks/`` folder, with the repository root on
``sys.path`` (``export PYTHONPATH=$PWD`` from the root before starting
Jupyter): notebook 08 imports
``tests.sim.test_pipeline_golden``, and notebooks 09, 11 and 12 read
``../sim2root/...`` and ``../examples/...``.

.. note::

   The notebooks are not built into this documentation, since executing them
   would slow every build.  The links below open them on GitHub, with their
   stored outputs.  CI executes them every week and on every change, so the
   stored outputs stay current.

Editing a notebook, or adding one
---------------------------------

The notebooks are written by ``notebooks/make_notebooks.py``, which holds
their source, executes each one and stores its outputs.  A change made only in
the ``.ipynb`` is lost the next time the notebooks are rebuilt.  To keep it,
edit the notebook in Jupyter as usual, then bring it back into the generator:

.. code-block:: bash

    python notebooks/import_notebook.py notebooks/05_galactic_noise.ipynb

A new notebook is added the same way.  Give it the next free number, start it
with a ``# NN — Title`` heading, and keep its data paths relative to
``notebooks/``:

.. code-block:: bash

    python notebooks/import_notebook.py ~/my_analysis.ipynb --name 13_my_analysis.ipynb

The script checks that every code cell compiles and that no cell uses an
absolute path such as ``/home/...``, which would not exist on another machine.
It then writes the notebook's block in ``make_notebooks.py``, executes the
notebook and stores its outputs.  If anything fails, it restores both files
and says why.  For a new notebook, it also reminds you to add it to the list
below.  Commit ``make_notebooks.py`` and the ``.ipynb`` together.

To rebuild notebooks from the generator directly::

    python notebooks/make_notebooks.py                # rebuild and execute all
    python notebooks/make_notebooks.py --only 03,05   # just those two
    python notebooks/make_notebooks.py --check        # check, writing nothing

CI fails if a committed notebook does not match the generator or does not
execute.

The notebooks
-------------

`01. Coordinate systems <https://github.com/grand-mother/grand/blob/dev-next/notebooks/01_coordinates.ipynb>`_
   The four frames and the conversions between them; the two conventions that
   most often catch people out; a detector layout drawn in ``GRANDCS`` and
   again in geodetic coordinates; a shower axis in both.  The long form of
   :doc:`coordinates`.

`02. Reading and writing GRAND data <https://github.com/grand-mother/grand/blob/dev-next/notebooks/02_data_model.ipynb>`_
   Builds a file from nothing and reads it back.  The ``TRun*`` / ``T*`` /
   ``T*Sim`` naming convention, descriptor-driven fields, and the three
   ``DataDirectory`` grouping rules that are written down nowhere else.  The
   long form of :doc:`datamodel`.

`03. The antenna response <https://github.com/grand-mother/grand/blob/dev-next/notebooks/03_antenna_response.ipynb>`_
   The :term:`effective length` against frequency and direction, its phase and the
   group delay that follows, and why the Z arm is not a scaled copy of the
   horizontal arms.  Reproduces Fig. 5 of :cite:`GRAND:2024atu`.

`04. The RF chain <https://github.com/grand-mother/grand/blob/dev-next/notebooks/04_rf_chain.ipynb>`_
   Each stage's own transfer function, why the cascade is done in ABCD rather
   than :term:`S-parameters`, and the total :math:`V_{\rm out}/V_{\rm oc}` of Fig. 8.
   Ends by plotting four VGA gain settings that produce one curve — see
   :ref:`the known issue <issue-vga-gain-ignored>`.

`05. Galactic noise <https://github.com/grand-mother/grand/blob/dev-next/notebooks/05_galactic_noise.ipynb>`_
   Follows one number — the microvolts of noise on an antenna arm — from the
   :term:`LFMap` radio sky to a simulated trace, and then checks it.  The sky is 180
   times brighter at 30 MHz than at 250, which is why every noise spectrum
   falls so steeply; the level varies by a third over a sidereal day; and the
   expected level, rebuilt independently from the shipped tables and the
   antenna impedance, agrees with the simulation to a few tenths of a percent.

`06. From electric field to ADC counts <https://github.com/grand-mother/grand/blob/dev-next/notebooks/06_efield_to_adc.ipynb>`_
   The whole chain on a fixture built in the notebook: each stage isolated,
   the signal-to-noise that follows from the input amplitude, digitization,
   and a round trip back out of the written file.  Includes the trap that the
   three output channels are antenna arms, not Cartesian components of the
   field.

`07. Topography <https://github.com/grand-mother/grand/blob/dev-next/notebooks/07_topography.ipynb>`_
   The two definitions of height and the :term:`geoid undulation` between them, a
   terrain map, and ray-ground intersection for very inclined showers —
   including how badly a flat-ground estimate does near the horizon.  Also the
   silent ``nan`` a missing elevation tile returns.

`08. Pinning the chain <https://github.com/grand-mother/grand/blob/dev-next/notebooks/08_pipeline_regression.ipynb>`_
   For the person about to change the chain: what will tell you the answer
   moved.  Runs the simulation with one thing different at a time — noise
   off, :term:`RF chain` off, a different sidereal hour, a different seed — and
   measures what the regression catches.  Shows that with a symmetric input a
   swapped antenna arm is completely invisible, which is why the reference
   input is asymmetric.

`09. Reading events <https://github.com/grand-mother/grand/blob/dev-next/notebooks/09_reading_events.ipynb>`_
   ``grand.aoi``, the layer above ``grand.dataio``: it hands back an event
   rather than branches, with its antennas on one common clock.  Covers the
   trap that ``EventList`` reuses a single ``Event`` object, so collecting
   events in a list gives the same one ten times over; which :term:`analysis level`
   you get when a directory holds both; and the difference between
   ``event.shower`` and ``event.simshower``.  Ends by reading one event
   through both layers and comparing them.

`10. Finding data <https://github.com/grand-mother/grand/blob/dev-next/notebooks/10_finding_data.ipynb>`_
   ``granddb``, the data catalog that ships with GRANDlib and answers "where
   is this file?".  Builds a config file, finds files in local directories and
   their subdirectories, and shows what the ``[repositories]`` and
   ``[credentials]`` sections look like.  The point it exists to make: **you do
   not need a database to find files** — the ``[database]`` section is optional
   and the whole notebook runs without one.  Says plainly which four things do
   need the catalog.

`11. Reconstructing a shower <https://github.com/grand-mother/grand/blob/dev-next/notebooks/11_reconstruction.ipynb>`_
   ``grand.analysis``, the reconstruction package: from the peak times and
   amplitudes the antennas recorded back to the shower's direction, the
   distance to its source and an energy estimate. Runs each step — plane
   wave, spherical wave, angular distribution function, energy proxy — on a
   shower it generated, so the answer is known, then the whole chain as
   ``examples/analysis/main_AOI.py`` runs it. Ends on the ten :term:`GP13` cosmic-ray
   candidates shipped with the examples, where the data parts company with
   the model: most timing fits have χ²/ndf well above 1, and 8 of 10
   amplitude fits stop on a bound. Says plainly that the fits are checked for
   self-consistency only, not yet against simulated showers.

`12. The event viewer <https://github.com/grand-mother/grand/blob/dev-next/notebooks/12_event_viewer.ipynb>`_
   How to run ``examples/eventviewer/``, from the command line and from
   Python, and what each of its panels computes, redrawn as static figures.
   Covers the magnetic-field fix that turned the shower-plane axes the right
   way in September 2026, and the :term:`Xmax` correction that fixed the angular
   plane on the committed samples (:ref:`issue-xmax-sample-vintage`), which
   had put it off by 6.5° for one event and 1° for the other.

.. note::

   Notebook 07 needs SRTM elevation tiles, which are not in version control.
   The cells that need them detect their absence and say so, so the notebook
   still runs on a fresh checkout; ``topography.update_data()`` fetches what a
   region needs.

   Notebook 11 needs ``iminuit``, which the conda environment carries;
   elsewhere, ``pip install -e ".[analysis]"``. Notebook 12 needs the viewer's
   plotting stack, which the conda environment does not carry: install it with
   ``pip install -e ".[viewer]"``.
