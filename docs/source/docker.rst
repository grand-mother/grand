Docker
======

.. contents::
   :local:
   :depth: 1

Status, in one paragraph
------------------------

**The Docker route works.**  That was not obvious — the published images date
from January 2023 and pin ROOT 6.26 against 6.36 everywhere else — so it was
measured rather than assumed.  The full test suite passes inside the 2023
image.  A newer image, built from this repository, also passes.  What is *not*
settled is whether the collaboration wants to support Docker, which is a scope
question rather than a technical one; see :ref:`issue-docker-unmaintained`.

The supported route is still the conda environment in :doc:`installation`.

What was measured
-----------------

Run on 2 September 2026 by ``.github/workflows/docker.yml``, which pulls an
image, checks a branch out inside it, and runs four stages: ``env/setup.sh``,
``import grand``, the ``dataio`` suite, then the whole suite.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Image
     - Environment
     - Full suite
   * - ``grandlib/dev:1.2`` (2023-01-14)
     - ROOT 6.26.02, Python 3.8.10, NumPy 1.23.5
     - **459 passed** on ``dev-next``; 11 failed on ``dev``
   * - built here (below)
     - ROOT 6.36.00, Python 3.13.3, NumPy 2.2.3
     - **459 passed**

The old image runs the branch of ``grand/dataio/descriptors.py`` below both of
its ROOT-version thresholds — the one the conda CI matrix never reaches, since
both its legs are ROOT >= 6.36.  That branch turns out to be not merely present
but functional, which nothing had established before.

One test had to be fixed to get there, and it was the test that was wrong:
``test_get_peakamptime_norm_hilbert`` asserted an exact sample index for a peak
that sits at sample 511.5, exactly between two samples.  See
:ref:`the entry on it <issue-docker-unmaintained>` and the commit history.

Building the image, exactly
---------------------------

``env/docker/grandlib.dockerfile``.  Build it from the repository root, because
the build context has to include the source:

.. code-block:: bash

    docker build -f env/docker/grandlib.dockerfile -t grandlib:dev .

    docker run --rm -it -v "$PWD:/opt/grandlib" grandlib:dev
    # then, inside the container:
    source env/setup.sh && pytest tests/ -q

The Dockerfile is short, and every line in it exists for a reason found by a
failed build rather than by design:

.. code-block:: dockerfile

    FROM rootproject/root:6.36.00-ubuntu25.04

    RUN apt-get update \
     && apt-get install -y --no-install-recommends \
            build-essential make git ca-certificates \
            python3-pip python3-setuptools \
     && rm -rf /var/lib/apt/lists/*

    WORKDIR /opt/grandlib
    COPY . /opt/grandlib
    RUN python3 -m pip install --no-cache-dir --break-system-packages ".[dev]"

    ENV GRAND_ROOT=/opt/grandlib
    ENV PYTHONPATH=/opt/grandlib:${PYTHONPATH}

**The base is pinned to ROOT 6.36** to match ``env/conda/grand-dev.yml``.  A
Docker user should not silently be running a different branch of the schema
code from everybody else.

**Dependencies come from** ``pyproject.toml``, via ``.[dev]``, and not from a
requirements file.  This is the whole design point.  ``env/docker_amd64/``
carries a ``requirements.txt`` of 53 unpinned package names — a second
dependency list maintained by hand, which was the only place declaring
``numba`` and ``lmfit`` and drifted from the conda environment for three years
because nothing built it.  One source of truth, two delivery mechanisms.

**python3-pip and python3-setuptools are installed explicitly.**  The
``rootproject/root`` image ships neither.  Without pip the build fails with
``/usr/bin/python3: No module named pip``; without setuptools the *later*
``env/setup.sh`` fails to compile the ``_core`` extension, because Python 3.13
removed ``distutils`` and ``cffi`` falls back to setuptools.

**PYTHONPATH appends rather than replaces.**  The base image sets
``PYTHONPATH=/opt/root/lib``, and overwriting it makes ROOT itself
unimportable — ``ModuleNotFoundError: No module named 'ROOT'``.

**The ~1 GB data model is deliberately not in the image.**  ``env/setup.sh``
fetches it, compares versions and skips when current, so baking it in would
roughly quadruple the image in order to freeze something on its own release
schedule.  Fetch it at first run, or mount a directory that already has it.

What is not done
----------------

**Nothing publishes this image.**  ``docker.yml`` builds it and runs the suite
inside it, which is enough to know whether it works.  Pushing to a registry is
a decision about what the collaboration distributes, not something a CI job
should start doing on its own; adding it is a login step and a ``docker push``
once that decision is made.

**arm64 is untested.**  ``ubuntu-latest`` runners are x86_64, and the only
published arm64 image, ``grandlib/dev:2.0``, dates from November 2022.  Testing
it needs either the ``ubuntu-24.04-arm`` runners or QEMU emulation.

**A 2025 image exists that nothing here can reach.**  The Handbook's hands-on
chapter distributes ``grand_docker_handson_2025.tar.gz`` — two years newer than
anything on Docker Hub — as a Google Drive tarball rather than through a
registry.  If it works, publishing *it* would be a smaller job than any of the
above.

Running the check yourself
--------------------------

``workflow_dispatch`` does not work for this workflow yet: GitHub only offers
it for files present on the default branch, and this repository's default is
``master``.  Push to the trigger branch instead:

.. code-block:: bash

    git push origin dev-next:ci/docker-test --force

That runs the published-image check against ``dev-next`` and ``dev`` in
parallel, and builds the image above.  See :doc:`ci` for the details.
