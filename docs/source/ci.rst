Continuous integration
======================

.. contents::
   :local:
   :depth: 1

Continuous integration (CI) is the practice of checking every change to the
code automatically, on a clean machine, as soon as it is pushed.  It catches
a broken test, a style error or a documentation page that no longer builds
within minutes, before the change reaches other people, and it catches
problems that only appear on a machine other than the author's.

For GRANDlib, GitHub Actions runs the test suite, the linter and a
documentation build on every push and every pull request, including pull
requests from forks.  The results appear as checks on the commit and on the
pull request: a green check means every step passed, and a red cross links
to the log of the step that failed.  Every check runs a command you can also
run locally (`Running the checks locally`_).

The workflows
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 33 45

   * - Workflow
     - Runs on
     - What it does
   * - ``tests-conda.yml``
     - every push and pull request
     - The test suite, with ROOT 6.36 and with ROOT 6.38
   * - ``lint.yml``
     - every push and pull request
     - Ruff; the documentation build, which fails on any warning; and the
       compilation of the Handbook PDF
   * - ``notebooks.yml``
     - changes under ``notebooks/``; Mondays at 05:00 UTC; on request
     - Executes every tutorial notebook and checks that it matches its
       generator
   * - ``pages.yml``
     - pushes to ``dev-next`` or ``main``; on request
     - Builds the documentation and deploys it to GitHub Pages
   * - ``docker.yml``
     - pushes to a branch named ``ci/docker-test``; on request
     - Runs the suite inside a Docker image (`Testing the Docker route`_)
   * - ``root_version.yml``
     - changes to ``grand/dataio/version``
     - Tags the version of the ROOT data format

The jobs build their environment from ``env/conda/grand-dev.yml``, the same
file used for a local installation, so CI tests what users run.  ROOT 6.36 is
the version the environment pins and the one that gates a merge.  The ROOT
6.38 job reports its result without blocking, until 6.38 becomes the
supported version.

Each job has a time limit (25 minutes for the tests), so a hang fails instead
of holding a runner.

How the checks are decided
--------------------------

A first job, ``changes``, looks at what a push modified.  A push that touches
only documentation skips the test suite, but still reports a result, so a
required check never waits on a job that will not run.  When ``changes``
cannot tell (a new branch, a force push), everything runs.

The documentation build treats every warning as an error, with one exception:
on some processors, ROOT's JIT compiler writes a CPU-feature diagnostic to
standard error, which the notebook extension reports as a warning.  The job
builds with ``--keep-going`` and fails on any warning in the log other than
that one.  ``-W`` is not used, since it would fail on that diagnostic.

The documentation links to the Handbook PDF.  The ``handbook`` job compiles it
from its LaTeX source; the documentation job uses the copy in ``resources/``,
so it needs no LaTeX installation.

The model data
--------------

Every job needs the model data, about 1 GB from ``forge.in2p3.fr``.  The
workflows cache it, keyed on ``data/model_version.flag``, so it is downloaded
only when the model version changes.  There is no fallback to an older
version, so a job never runs against the wrong data.  The download script
retries four times and checks the size of each file, and it replaces the
previous data only after the new data have arrived.

Running the checks locally
--------------------------

.. code-block:: bash

    pytest tests/ -q                                             # the suite
    ruff check grand/ tests/ quality/ notebooks/ docs/dev/ granddb/   # the linter
    cd docs && make html                                         # the documentation
    python notebooks/make_notebooks.py                           # the notebooks

The documentation build should print no ``WARNING`` lines other than the ROOT
diagnostic above.

Testing the Docker route
------------------------

``docker.yml`` checks whether GRANDlib still installs and passes its tests
inside a Docker image.  It is a diagnostic, not a merge gate.  It runs
``env/setup.sh``, ``import grand``, the ``dataio`` tests and the full suite as
separate steps, so a failure shows which stage broke.  By default it tests the
published ``grandlib/dev:1.2`` image against ``dev-next`` and ``dev``, and
builds the image of ``env/docker/grandlib.dockerfile``.

GitHub offers manual runs only for workflows on the repository's default
branch, which is currently ``master``.  Until ``dev-next`` becomes the
default, start it by pushing to the trigger branch:

.. code-block:: bash

    git push origin dev-next:ci/docker-test --force

Publishing the documentation
----------------------------

``pages.yml`` builds this documentation and publishes it at
https://grand-mother.github.io/grand/ on every push to ``dev-next``.

To publish a preview of your own branch, from a fork:

.. code-block:: bash

    gh repo fork grand-mother/grand --clone=false
    gh repo edit <you>/grand --default-branch dev-next
    gh api -X POST repos/<you>/grand/pages -f build_type=workflow
    git push <your fork> dev-next                    # registers the workflows
    gh workflow run pages.yml --repo <you>/grand --ref dev-next

The site then appears at ``https://<you>.github.io/grand/``.  Make clear,
wherever you share the link, that it is a preview.

Coverage reports
----------------

The test job uploads a coverage report to Codecov.  The upload currently
fails, because the repository has no ``CODECOV_TOKEN`` secret and Codecov
refuses tokenless uploads on protected branches.  The failure does not fail
the job.  Adding the secret (administrator access) is all that is needed; until
then, measure coverage locally as described in :doc:`testing`.
