Tutorial notebooks
==================

Worked notebooks live in `notebooks/
<https://github.com/grand-mother/grand/tree/dev-next/notebooks>`_, numbered in
reading order.  Each carries its figures inline, so they can be read on GitHub
without being run.

A recipe (:doc:`recipes`) shows a task in a few lines; a notebook works
through it with the reasoning and the checks behind the numbers.

To run them, start Jupyter from the repository root in an installed
environment (:doc:`installation`), with the root on the Python path:

.. code-block:: bash

    export PYTHONPATH=$PWD
    jupyter lab notebooks/

The notebooks are not executed when this documentation is built; CI runs them
every week and on every change, so their stored outputs stay current.

The notebooks
-------------

`01. Coordinate systems <https://github.com/grand-mother/grand/blob/dev-next/notebooks/01_coordinates.ipynb>`_
   The frames and the conversions between them, with a detector layout and a shower axis drawn in each.  The long form of :doc:`coordinates`.

`02. Reading and writing GRAND data <https://github.com/grand-mother/grand/blob/dev-next/notebooks/02_data_model.ipynb>`_
   Builds a file from nothing and reads it back; how ``DataDirectory`` groups files.  The long form of :doc:`datamodel`.

`03. The antenna response <https://github.com/grand-mother/grand/blob/dev-next/notebooks/03_antenna_response.ipynb>`_
   The :term:`effective length` against frequency and direction and its phase.  Reproduces Fig. 5 of :cite:`GRAND:2024atu`.

`04. The RF chain <https://github.com/grand-mother/grand/blob/dev-next/notebooks/04_rf_chain.ipynb>`_
   Each stage's transfer function and the total of Fig. 8, cascaded in ABCD form.

`05. Galactic noise <https://github.com/grand-mother/grand/blob/dev-next/notebooks/05_galactic_noise.ipynb>`_
   From the :term:`LFMap` sky to the noise on an antenna arm, its variation over a sidereal day and a check against the tables.

`06. From electric field to ADC counts <https://github.com/grand-mother/grand/blob/dev-next/notebooks/06_efield_to_adc.ipynb>`_
   The whole chain, one stage at a time, on an input built in the notebook.

`07. Topography <https://github.com/grand-mother/grand/blob/dev-next/notebooks/07_topography.ipynb>`_
   Heights, the :term:`geoid undulation`, a terrain map and where an inclined shower axis meets the ground.

`08. Pinning the chain <https://github.com/grand-mother/grand/blob/dev-next/notebooks/08_pipeline_regression.ipynb>`_
   For anyone changing the simulation: what the regression test detects when one input changes.

`09. Reading events <https://github.com/grand-mother/grand/blob/dev-next/notebooks/09_reading_events.ipynb>`_
   ``grand.aoi``: events with their antennas on one clock and the pitfalls of looping over them.

`10. Finding data <https://github.com/grand-mother/grand/blob/dev-next/notebooks/10_finding_data.ipynb>`_
   ``granddb``, the data catalog: finding files locally and in repositories, with no database needed.

`11. Reconstructing a shower <https://github.com/grand-mother/grand/blob/dev-next/notebooks/11_reconstruction.ipynb>`_
   Direction, distance to the source and energy from recorded times and amplitudes, on known showers and on ten GP13 candidates.

`12. The event viewer <https://github.com/grand-mother/grand/blob/dev-next/notebooks/12_event_viewer.ipynb>`_
   Running ``examples/eventviewer/`` and what each of its panels shows.

.. note::

   Notebook 07 needs SRTM elevation tiles, which are not in version control.
   The cells that need them detect their absence and say so, so the notebook
   still runs on a fresh checkout; ``topography.update_data()`` fetches what a
   region needs.

   Notebook 11 needs ``iminuit``, which the conda environment carries;
   elsewhere, ``pip install -e ".[analysis]"``. Notebook 12 needs the viewer's
   plotting stack, which the conda environment does not carry: install it with
   ``pip install -e ".[viewer]"``.

Where next
----------

* :doc:`recipes` for the same tasks in a few lines.
* :doc:`quickstart` if you have not run GRANDlib yet.
