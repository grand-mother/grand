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

The conda environment above already carries what ``granddb`` needs —
``sqlalchemy``, ``psycopg2``, ``paramiko``, ``sshtunnel`` and ``scp``.  It
ships as part of this package, so its dependencies belong in the environment
rather than in an overlay a contributor might not have when running the tests.

Installing outside conda, take them from the package's own extra::

    pip install -e '.[db]'

``env/conda/reqmt_db.txt`` predates that extra and is kept for existing
workflows; it installs a wider set than ``granddb`` actually uses.

The Snakemake reprocessing pipeline is likewise not in the core environment —
``conda-forge`` has no ``snakemake-minimal`` build for every platform, and
only the pipeline needs it.  Install it alongside if you run that workflow.

Docker
------

Docker works — measured, not assumed: the full suite passes both inside the
2023 published image and inside a newer one built from this repository.  It is
still not the *supported* route, because nobody maintains the published images
and the collaboration has not decided whether it wants to.

:doc:`docker` has the measurements, the Dockerfile and what each line of it is
for.  Use the conda environment above unless you have a specific reason not
to.

Troubleshooting
---------------

**A previous build failed.**  Compilation artefacts from a wrong environment
persist.  Clean them before retrying::

    cd src && make clean && cd ..
    source env/setup.sh

**ARM processors.**  The environment is verified on x86-64.  Apple Silicon
and other ARM hosts have known problems compiling TURTLE and GULL.

.. admonition:: One route that has worked on ARM
   :class: tip

   Contributed by Tien in March 2025 and not re-tested since, so treat it as a
   starting point rather than a supported path.  The conda environment itself
   is built for amd64, and the trouble begins at the TURTLE and GULL
   compilation rather than at the environment.

   1. Create the conda environment and install the Python packages, following
      ``env/conda/admin/readme.md``.
   2. Build ``turtle`` and ``gull`` by hand instead of leaving them to
      ``env/setup.sh``.
   3. Clone ``grand`` and run ``source env/setup.sh``.
   4. Check the result with ``python -c "import grand"``.

   Most of what goes wrong is a path that points at the wrong place, which is
   usually easy to correct once the failure is read.  If you get this working,
   please say so on the `collaboration issues tracker
   <https://github.com/grand-mother/collaboration-issues/issues>`_, so that it
   can be promoted from one person's anecdote to an instruction.
