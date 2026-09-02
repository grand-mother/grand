Tutorial notebooks
==================

Worked notebooks live in `notebooks/
<https://github.com/grand-mother/grand/tree/dev-next/notebooks>`_, numbered in
reading order.  Each carries its figures inline, so they can be read on GitHub
without being run.

They are the long form of the narrative pages.  A page states a convention and
shows it executing in a few lines; a notebook works through the same material
with the reasoning around it — why the convention is what it is, what happens
at the edges, and what the numbers were checked against.

To run them rather than read them::

    conda env create -f env/conda/grand-dev.yml --solver=libmamba
    conda activate grand-dev
    source env/setup.sh
    jupyter lab notebooks/

.. note::

   The notebooks are **not** built into this documentation.  Executing them on
   every documentation build would be slow, and their stored outputs are more
   useful than freshly computed ones for reading.  A scheduled job runs them
   weekly instead, so they cannot rot unnoticed.

They are generated
------------------

The notebooks are built by ``notebooks/make_notebooks.py``, which owns their
source, executes each one and stores its outputs.  **Edit that file, not the**
``.ipynb`` — anything written into a notebook by hand is lost on the next
rebuild::

    python notebooks/make_notebooks.py                # rebuild and execute all
    python notebooks/make_notebooks.py --only 03,05   # just those two
    python notebooks/make_notebooks.py --no-execute   # while drafting

Generating them keeps the title format, the navigation footers and the shared
conventions structural rather than something seven JSON files have to agree
about, and makes review a diff of Python instead of a diff of embedded output.
The build refuses to finish if a notebook fails to execute, comes back without
stored outputs — which renders blank on GitHub — or is left on disk not
matching the generator.

Available
---------

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
   The effective length against frequency and direction, its phase and the
   group delay that follows, and why the Z arm is not a scaled copy of the
   horizontal arms.  Reproduces Fig. 5 of :cite:`GRAND:2024atu`.

`04. The RF chain <https://github.com/grand-mother/grand/blob/dev-next/notebooks/04_rf_chain.ipynb>`_
   Each stage's own transfer function, why the cascade is done in ABCD rather
   than S-parameters, and the total :math:`V_{\rm out}/V_{\rm oc}` of Fig. 8.
   Ends by plotting four VGA gain settings that produce one curve — see
   :ref:`the known issue <issue-vga-gain-ignored>`.

`05. Galactic noise <https://github.com/grand-mother/grand/blob/dev-next/notebooks/05_galactic_noise.ipynb>`_
   The LFMap-derived tables, the diurnal cycle, the spectral shape, and a
   worked measurement of the normalisation showing that the simulated RMS is
   :math:`1/\sqrt2` of the tabulated value.  Also shows that ``du_type``
   changes the absolute level by up to a factor of two.  See
   :ref:`issue-galactic-noise-normalisation`.

`06. From electric field to ADC counts <https://github.com/grand-mother/grand/blob/dev-next/notebooks/06_efield_to_adc.ipynb>`_
   The whole chain on a fixture built in the notebook: each stage isolated,
   the signal-to-noise that follows from the input amplitude, digitisation,
   and a round trip back out of the written file.  Includes the trap that the
   three output channels are antenna arms, not Cartesian components of the
   field.

`07. Topography <https://github.com/grand-mother/grand/blob/dev-next/notebooks/07_topography.ipynb>`_
   The two definitions of height and the geoid undulation between them, a
   terrain map, and ray-ground intersection for very inclined showers —
   including how badly a flat-ground estimate does near the horizon.  Also the
   two silent ``nan`` returns, one of which is
   :ref:`issue-geoid-longitude-convention`.

.. note::

   Notebook 07 needs SRTM elevation tiles, which are not in version control.
   The cells that need them detect their absence and say so, so the notebook
   still runs on a fresh checkout; ``topography.update_data()`` fetches what a
   region needs.
