Installation
============

.. contents::
   :local:

GRANDlib needs ROOT, plus the TURTLE and GULL C libraries, which are compiled
from source during setup.  A conda environment is the supported way to get
all of that in one place, and this repository ships the environment file.

Recommended: the provided conda environment
-------------------------------------------

.. code-block:: bash

    conda env create -f env/conda/grand-dev.yml --solver=libmamba
    conda activate grand-dev
    source env/setup.sh

The first command creates an environment named ``grand-dev``; the third
compiles TURTLE and GULL, builds the ``_core.abi3.so`` extension, sets the
environment variables GRANDlib expects, and downloads the data model
(topography, geomagnetic field and antenna models).

``--solver=libmamba`` is worth passing explicitly.  The environment pins
around thirty packages on top of ROOT, and conda's classic solver is slow and
memory-hungry on a problem that size; libmamba resolves it in a fraction of
the time and memory.  If your conda already defaults to it, the flag is
harmless.

Verifying the install
---------------------

.. code-block:: bash

    python -c "import ROOT; print(ROOT.gROOT.GetVersion())"
    python -c "import grand; print(grand.GRAND_DATA_PATH)"

This procedure was last verified on 30 August 2026, on Linux x86-64, giving:

=================  ==========
Component          Version
=================  ==========
Python             3.12.14
ROOT               6.36.04
NumPy              2.5.2
SciPy              1.16.1
cffi               2.1.1
=================  ==========

The environment is about 4.1 GB on disk, plus roughly 1 GB of downloaded
packages in the conda cache.

What the environment provides
-----------------------------

``env/conda/grand-dev.yml`` is the single dependency list.  It consolidates
four that had drifted apart: the previous runtime file, the pip-installed
test and lint tools in ``quality/requirements.txt``, a third set under
``env/docker_*/`` that was the only one carrying ``numba`` and ``lmfit``, and
documentation dependencies that nothing declared at all.

* **ROOT 6.36**, for the data format.
* **The numerical stack**: NumPy, SciPy, matplotlib, h5py, uproot.
* **Build tools**: a C compiler, ``make``, ``cffi`` and ``pycparser``, needed
  by ``src/Makefile`` and ``src/build_core.py`` to compile TURTLE and GULL.

  These are declared explicitly rather than inherited.  ROOT happens to pull
  a toolchain in transitively today, so the build would succeed without
  naming them — but nothing would record that it must, and a ROOT release
  that stopped shipping a compiler would break setup with no dependency list
  to explain why.  ``make`` is not declared by any other environment file and
  currently works only because most Linux hosts provide it; a minimal
  container would fail.
* **Test and quality tooling**: pytest, coverage, pylint, mypy, black, ruff.
* **Documentation tooling**: Sphinx and the extensions ``docs/`` needs, so
  that one environment covers running, testing and documenting the package.

Optional overlays
-----------------

Database work needs additional packages, kept separate because most users
never touch PostgreSQL::

    pip install -r env/conda/reqmt_db.txt

The Snakemake reprocessing pipeline is likewise not in the core environment —
``conda-forge`` has no ``snakemake-minimal`` build for every platform, and
only the pipeline needs it.  Install it alongside if you run that workflow.

Docker
------

There are Dockerfiles under ``env/docker_amd64/`` and ``env/docker_arm64/``,
and published images on Docker Hub, and the Handbook presents them as the
first installation route.  **They are unmaintained.**

The newest published image dates from January 2023 and pins ROOT 6.26.02,
against 6.36.04 here and a CI matrix of 6.36 and 6.38.  Nothing builds the
images and no CI covers them, which is why they drifted.  The route still
*runs* — the setup script never invokes pip, and the package uses no syntax
newer than Python 3.8 — but it puts you three years away from the environment
everybody else is using, silently.

Use the conda environment above.  Whether Docker becomes a supported route
again is an open question rather than a defect to patch; see
:ref:`issue-docker-unmaintained`.

Not yet available
-----------------

.. warning::

   ``pip install -e .`` does not work: there is no ``pyproject.toml`` at the
   repository root, so GRANDlib cannot be installed as a package and is used
   from the source tree via ``env/setup.sh``.  Adding it is Phase 1b of the
   repository overhaul, tracked in PR 154.

Troubleshooting
---------------

**A previous build failed.**  Compilation artefacts from a wrong environment
persist.  Clean them before retrying::

    cd src && make clean && cd ..
    source env/setup.sh

**ARM processors.**  The environment is verified on x86-64.  Apple Silicon
and other ARM hosts have known problems compiling TURTLE and GULL; see
``env/conda/readme.md``.
