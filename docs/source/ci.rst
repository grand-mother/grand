Continuous integration
======================

Continuous integration (CI) is the practice of checking every change to the
code automatically, on a clean machine, as soon as it is pushed.  It catches
a broken test, a style error or a documentation page that no longer builds
within minutes, before the change reaches other people.  It also catches
problems that only appear on a machine other than the author's.

.. contents::
   :local:
   :depth: 1

For GRANDlib, GitHub Actions runs the test suite, the linter and a
documentation build on every push and every pull request, including pull
requests from forks.  The results appear as checks on the commit and on the
pull request: a green check means every step passed; a red cross links
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
     - Ruff; the documentation style check; the documentation build, which
       fails on any warning; the compilation of the Handbook PDF
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
   * - ``linkcheck.yml``
     - Mondays at 06:17 UTC; on request
     - Checks the external links of the documentation and the package
       metadata
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

A push that changes only documentation skips the test suite.  The model data
are cached between runs and downloaded again only when their version changes.
The comments in the workflow files explain the details.

Running the checks locally
--------------------------

.. code-block:: bash

    pytest tests/ -q                                             # the suite
    ruff check grand/ tests/ quality/ notebooks/ docs/dev/ granddb/   # the linter
    python docs/dev/check_style.py                               # the writing style
    cd docs && make html                                         # the documentation
    python notebooks/make_notebooks.py                           # the notebooks

The documentation build should print no ``WARNING`` lines, except possibly a
CPU-feature diagnostic from ROOT on some processors, which is harmless.

Testing the Docker route
------------------------

``docker.yml`` checks that GRANDlib still installs and passes its tests inside
a Docker image (:doc:`installation`).  It reports, but does not block a merge.
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
