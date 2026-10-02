Installation and requirements
=============================

.. contents::
   :local:
   :depth: 2

Requirements
------------

GRANDlib runs on Linux x86-64 with Python 3.10 or later.  Beyond Python
packages, it needs:

* **ROOT**, for the data format.  GRANDlib is tested with ROOT 6.36 and 6.38.
* **A C compiler and** ``make``, to build the :term:`TURTLE` (terrain) and :term:`GULL`
  (geomagnetic field) libraries from source.
* **About 1 GB of model data**: antenna effective lengths, RF-chain
  measurements and Galactic-noise tables, downloaded once (:doc:`data_files`).

The Python dependencies are NumPy, SciPy, matplotlib, h5py, uproot, joblib,
asdf and cffi, declared in ``pyproject.toml``.  Optional extras add what some
parts of the package need:

============  =========================================================
Extra         Adds
============  =========================================================
``analysis``  ``iminuit``, for the reconstruction in :mod:`grand.analysis`
``db``        what ``granddb`` needs to reach the data catalog
``viewer``    the plotting stack of the event viewer in ``examples/``
``docs``      Sphinx and the extensions that build this documentation
``dev``       pytest, coverage and the linters
============  =========================================================

GRANDlib is distributed under the GNU Lesser General Public License, version 3
or later; see ``LICENSE`` and ``COPYING.LESSER`` in the repository.

Installation
------------

GRANDlib is installed from a clone of the repository.  A conda environment is
the simplest route, because it provides ROOT and the compiler together with
the Python packages:

.. code-block:: bash

    git clone https://github.com/grand-mother/grand.git
    cd grand
    conda env create -f env/conda/grand-dev.yml --solver=libmamba
    conda activate grand-dev
    source env/setup.sh

``env/setup.sh`` compiles TURTLE and GULL, sets ``GRAND_ROOT`` and the search
paths, and downloads the model data.  It takes a few minutes the first time.
Run it again (``source env/setup.sh``) in every new shell, or add the
variables it sets to your shell profile.  The environment uses about 4 GB of
disk.

``--solver=libmamba`` resolves the environment in a fraction of the time and
memory of conda's classic solver.  If your conda already uses it by default,
the flag changes nothing.

Does pip work?
~~~~~~~~~~~~~~

Yes, for everything except ROOT.  GRANDlib is not yet published on PyPI, and
ROOT cannot be installed with pip.  With ROOT already available, from a conda
package, a system package or the ROOT Docker image, pip installs the rest from
the clone:

.. code-block:: bash

    pip install -e ".[dev]"          # or ".[analysis,db]", etc.
    source env/setup.sh

``env/setup.sh`` is still needed: pip does not build TURTLE and GULL or fetch
the model data.  Inside the conda environment, ``pip install -e .`` adds an
editable install without changing the packages conda provides.

Docker
~~~~~~

The repository includes a Dockerfile, ``env/docker/grandlib.dockerfile``, that
starts from the official ROOT 6.36 image and installs GRANDlib with pip.
Build it from the repository root, since the build copies the source in:

.. code-block:: bash

    docker build -f env/docker/grandlib.dockerfile -t grandlib:dev .
    docker run --rm -it -v "$PWD:/opt/grandlib" grandlib:dev
    # inside the container:
    source env/setup.sh && pytest tests/ -q

The model data are not in the image; ``env/setup.sh`` downloads them on first
run, or mount a directory that already holds them.  When last checked, on
2 September 2026, the full test suite passed inside this image and inside
``grandlib/dev:1.2``, the 2023 image the Handbook refers to, which carries
ROOT 6.26 and Python 3.8.

No image is currently published to a registry, and arm64 images are untested.
Whether the collaboration maintains Docker images is an open decision
(:ref:`issue-docker-unmaintained`); the conda environment is the supported
route.

Verifying the installation
--------------------------

.. code-block:: bash

    python -c "import grand; print(grand.GRAND_DATA_PATH)"
    python -m grand.basis.data_model        # checks the model data against its manifest
    pytest tests/ -q

The test suite has about 1270 tests and takes about 20 minutes on one core.
:doc:`testing` explains what it checks and what a skip or an expected failure
means.

The last full verification of this procedure, on 30 August 2026, used:

=================  ==========
Component          Version
=================  ==========
Python             3.12.14
ROOT               6.36.04
NumPy              2.5.2
SciPy              1.16.1
cffi               2.1.1
=================  ==========

If the installation fails
-------------------------

**A previous build failed.**  Files compiled in a wrong environment persist.
Remove them and build again:

.. code-block:: bash

    cd src && make clean && cd ..
    source env/setup.sh

``env/setup.sh`` prints the steps that failed at the end of its output, and
returns a non-zero status.

**The download stopped.**  The model data come from ``forge.in2p3.fr``.  The
script retries four times and checks the size of what it received; if it
still fails, run ``source env/setup.sh`` again later.

**ARM processors.**  The environment is built and tested on x86-64.  Apple
Silicon and other ARM hosts have had problems compiling TURTLE and GULL.  One
route reported to work in March 2025, not tested since: create the
environment following ``env/conda/admin/readme.md``, build TURTLE and GULL by
hand, then run ``source env/setup.sh`` and check with ``python -c "import
grand"``.  If you get it working, please say so in a `GitHub issue
<https://github.com/grand-mother/grand/issues>`_ so that it can become an
instruction here.

:doc:`troubleshooting` covers errors that appear after installation.
