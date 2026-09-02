Installation
============

.. warning::

   The four commands below are the **target** state, from Phase 1 of the
   repository overhaul.  They do not work yet: there is no ``pyproject.toml``
   at the repository root, and the pinned conda environment is missing
   ``cffi``, ``pycparser``, ``make`` and a C compiler, all of which
   ``src/build_core.py`` and ``src/Makefile`` need in order to build the
   TURTLE and GULL bindings.  The current procedure is in ``env/conda/readme.md``.

Once Phase 1 lands, installing GRANDlib is::

    conda env create -f env/conda/grand-dev.yml
    conda activate grand-dev
    pip install -e ".[dev]"
    pytest

That sequence is the exit criterion for Phase 1, and this page is what it is
checked against: if the four commands do not succeed on a clean machine, the
phase is not done.

What the environment provides
-----------------------------

* **ROOT 6.36**, for the data format.
* **TURTLE** and **GULL**, compiled from source by ``src/Makefile`` — hence
  the compiler, ``make``, ``cffi`` and ``pycparser``.
* The numerical stack: NumPy, SciPy, matplotlib, h5py, uproot.
* Test, lint and documentation tooling, so that one environment covers
  running, testing and documenting the package.

Optional extras, for database work, are kept separate in
``env/conda/reqmt_db.txt`` — most users never touch PostgreSQL.
