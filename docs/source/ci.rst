Continuous integration
======================

.. contents::
   :local:

What continuous integration is
------------------------------

Continuous integration is a set of checks that GitHub runs automatically on
the collaboration's own machines every time code changes, so that a problem is
found by a machine within minutes rather than by a person weeks later.

Nothing about it is specific to GRANDlib.  The checks are the same ones anyone
can run locally — the test suite, the linter, a documentation build — and the
point is only that they are run *every time*, on a clean machine, by something
that does not forget.  A clean machine matters: it is what catches the case
where the code works because of something installed on one person's laptop.

When it runs
------------

===========================  ==============================================
Event                        What happens
===========================  ==============================================
Push to any branch           The suite, the linter and a docs build
Opening or updating a PR     The same, including pull requests from forks
Merge to ``main``            The above, plus the documentation is deployed
Change to the data-format
version file                 The format version is tagged
===========================  ==============================================

Pull requests from forks matter here.  The workflows this replaces triggered
on ``push`` only, so an outside contributor's pull request received **no
checks at all** — three of the currently open pull requests come from forks
and have never been tested by anything.

Results appear as a check on the commit and on the pull request.  A failing
check is a request to look, not a verdict: the log says which step failed and
the command it ran, and every one of those commands can be run locally.

The workflows
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Workflow
     - Trigger
     - What it does
   * - ``tests.yml``
     - push, pull request
     - Runs the suite on two ROOT versions
   * - ``lint.yml``
     - push, pull request
     - Runs ruff, and builds the docs with warnings as errors
   * - ``pages.yml``
     - push to ``main``
     - Deploys the documentation to GitHub Pages
   * - ``root_version.yml``
     - change to ``grand/dataio/version``
     - Tags the ROOT data-format version

There is no container.  The environment is built from
``env/conda/grand-dev.yml`` with ``conda-incubator/setup-miniconda``, which is
the same file the collaboration installs locally, so what CI tests is what
people run.

Why there is no container
-------------------------

The repository used to run its tests inside two images, and both are why CI
stopped working:

* ``jcolley/grand:0.4``, used by the old ``tests.yml``, was **deleted from
  Docker Hub**.  The workflow had also been disabled by hand, which is why it
  has no runs at all in its history.
* ``jcolley/grandlib_ci:0.1``, used by ``tests_with_docker.yml``, still
  exists but was last pushed in **January 2022**.  It is too old for the
  runner to operate in — GitHub injects its own Node.js to execute
  JavaScript actions, and that needs a newer glibc.  Every job hung in
  container setup and was killed by the 24-hour limit with no step ever
  recorded.

Both images lived on one person's personal Docker Hub account.  The full
diagnosis is in ``docs/dev/FINDINGS_CI.md``.

Design decisions worth knowing
------------------------------

**The path filter lives inside the workflow, not on the trigger.**  A
workflow-level ``paths:`` does not skip the job — it stops the workflow from
existing, so no check is reported on the commit at all.  That is invisible
until the check becomes *required*, at which point a pull request touching
only ``docs/`` waits forever on something that will never report: pending,
not failing, with nothing saying why.  The ``changes`` job reports on every
commit and skips only the expensive steps.

**The gate fails safe in both directions.**  Anything ``changes`` cannot work
out — a new branch, a force push, an unfetched base — reports ``code=true``
and everything runs.  The jobs downstream test ``!= 'false'`` rather than
``== 'true'``, so a crash in the gate still runs the suite.  The failure mode
must be "did too much", never "silently did nothing".

**Jobs carry a timeout.**  ``timeout-minutes`` means a hang fails in
twenty-five minutes instead of occupying a runner for a day, which is what
the old container jobs did.

**Two ROOT versions.**  6.36 is what the environment pins; 6.38 is the release
whose stricter ``None`` comparison produced the warnings that a whole stack of
branches exists to fix.  Testing both is what stops that recurring unseen.

**Any documentation warning fails** ``lint.yml``, which is what keeps the
documentation from drifting from the code the way ``docs/package/grand.rst``
did — describing modules that had been renamed years earlier.

It is not done with ``-W``, and the reason is specific.  ``jupyter-sphinx``
treats *anything* an executed cell writes to stderr as a warning, and ROOT's
cling JIT writes a CPU-feature diagnostic there on some processors:

.. code-block:: text

    warning: invalid feature combination: +avx10.1-256;
             will be promoted to avx10.1-512

That depends on the runner's hardware, not on the documentation, and it appears
on GitHub's runners while not appearing on a typical developer machine.  Under
``-W`` it would fail the build for a reason nobody can act on.  So the job
builds with ``--keep-going`` and greps the log afterwards, filtering that one
diagnostic and its ``jupyter-sphinx`` wrapper; every other warning still fails
it.  The effect is the same gate, with one hardware-dependent exception carved
out explicitly rather than tolerated silently.

``pages.yml`` does not check the log at all: that job publishes, and a warning
is not a reason to leave the site stale.

**The documentation build needs the Handbook PDF to exist, not to be
compiled.**  ``make html`` runs ``docs/dev/build_handbook_pdf.py`` first,
because the pages link to the PDF with ``:download:`` and a missing target is
itself a warning.  Where ``pdflatex`` is absent that script installs the copy
shipped in ``resources/`` instead, which satisfies the link.

Only two jobs therefore carry a LaTeX toolchain, and for different reasons:

===============================  =====  ===================================
Job                              LaTeX  Why
===============================  =====  ===================================
``lint.yml`` / ``handbook``      yes    Its purpose is to verify the LaTeX
                                        still compiles.
``lint.yml`` / ``docs``          no     Needs the file to exist; the
                                        fallback provides it.
``pages.yml``                    yes    Publishes, so it should carry the
                                        *patched* PDF with the errata and
                                        the provenance block.
===============================  =====  ===================================

The ``docs`` job used to install it too, and on one run ``apt`` took **13
minutes 19 seconds** to do so — against 42 seconds for the documentation build
it was serving, and under a minute for the identical install in the
``handbook`` job of the same run.  That is mirror variance rather than
anything in the repository, but paying it in a job that does not need the
compiler is avoidable.

Testing the Docker route
------------------------

``docker.yml`` answers the question :ref:`issue-docker-unmaintained` poses:
does the installation route the Handbook recommends still work?  It is
**manual only** (``workflow_dispatch``) and is a diagnostic rather than a gate
— it is expected to be able to fail, which is the point of running it.

.. warning::

   **It cannot be triggered from the Actions tab yet, and neither can any other
   manual workflow added on this branch.**  GitHub only offers
   ``workflow_dispatch`` for a workflow file that exists on the *default*
   branch, and this repository's default is ``master`` — 1163 commits behind.
   Dispatching it returns

   .. code-block:: text

       HTTP 404: workflow docker.yml not found on the default branch

   The workflow therefore also triggers on a push to a branch named
   ``ci/docker-test``, which works from anywhere:

   .. code-block:: bash

       git push origin dev-next:ci/docker-test --force

   A dedicated branch name rather than a paths filter on the workflow file, so
   that editing it does not run it and running it does not require editing it.
   Phase 9 promotes ``dev-next`` to the default and the dispatch trigger starts
   working then; the push trigger can stay as the branch-local way in.

Trigger it from the Actions tab, once that is possible.  Two inputs, both with
defaults — on the push trigger there are no inputs, so the defaults
``grandlib/dev:1.2`` and ``dev-next,dev`` apply:

``image``
    The published image to test, ``grandlib/dev:1.2`` by default — the newest
    that exists, from January 2023.
``refs``
    Which branches to check out inside it, ``dev-next,dev`` by default.
    Testing both is what separates "Docker is stale" from "``dev-next`` broke
    Docker", a distinction worth having before anyone spends time on a fix.

Two independent jobs, because they answer different questions and can
disagree:

``published-image``
    Pulls the image, checks a branch out inside it, and runs four stages —
    ``env/setup.sh``, ``import grand``, the ``dataio`` suite, then everything.
    Split into separate steps so that GitHub shows *which* stage failed rather
    than one red cross on a long script.  ``dataio`` gets its own stage because
    that is where the ROOT-version branch lives, and the conda matrix never
    exercises it: both legs are ROOT >= 6.36 and take the other path.
``build-images`` — **opt-in**, off unless ``build_images`` is set
    Runs ``build_base.sh`` and ``build_dev.sh`` from ``env/docker_amd64/``.
    Those scripts assemble their requirements files by copying them out of the
    repository before building, so they have to be run rather than the
    Dockerfiles built directly.  This job is what would have caught the copy
    that pointed at ``docs/apidoc-only/`` after that directory was deleted.

    It is off by default because it answers a different question from "does the
    published image work", and is the slower and more fragile half: ``pip``
    resolving 53 unpinned packages against Python 3.8 has a great deal of room
    to backtrack.  The push trigger carries no inputs, so a branch-triggered
    run skips it and costs one job per ref rather than three.

.. note::

   Every ``docker run`` in that workflow passes ``--entrypoint`` explicitly.
   ``grandlib/dev`` declares ``Entrypoint: ["/bin/bash"]``, so the ordinary
   form ``docker run IMAGE bash -lc '...'`` becomes ``/bin/bash bash -lc
   '...'`` — bash handed bash as a script — and dies with

   .. code-block:: text

       /usr/bin/bash: /usr/bin/bash: cannot execute binary file

   and exit 126.  That is what the first run of this workflow did, in all four
   stages and in the two probes that report the image's Python and ROOT
   versions.  Overriding the entrypoint works whatever an image declares, which
   matters because the four candidate images were built separately and need not
   agree.

   Pushing to ``ci/docker-test`` also used to trigger ``Code Quality`` and
   ``Tests``, which had ``branches: ['**']`` — running the whole of CI a second
   time on a commit that had already passed it.  Both now carry a negative
   pattern, ``['**', '!ci/**']``.  It has to be a negative pattern inside
   ``branches`` rather than a separate ``branches-ignore``: GitHub rejects a
   workflow that uses both filters for the same event.

**Why in CI and not on a laptop.**  The image is about 3 GB uncompressed and
the data model another 2 GB at peak.  More to the point, running the container
against a working checkout would recompile TURTLE and GULL into it — built
against ROOT 6.26 and Ubuntu 20.04's glibc — and overwrite the host copies,
breaking the conda environment in a way that would take a while to trace.  On
a runner the whole machine is discarded afterwards.

The data model is cached
------------------------

Every job needs the detector, noise and topography data — about 1 GB, fetched
by ``env/setup.sh`` from ``forge.in2p3.fr``.  That host has no mirror, and the
transfer is not reliable at that size.  On one push two jobs failed on it and a
third passed:

.. code-block:: text

    ROOT 6.36.04   HTTP Error: 502 - Bad Gateway
    Documentation  retrieval incomplete: got only 358625964 out of 976538181 bytes
    Notebooks      (passed)

Three jobs, one commit, three different outcomes — which is what an unreliable
transfer looks like, not an outage.  Re-running the same commit unchanged
turned it green.

Two fixes, at different levels.

**The workflows cache it.**  ``actions/cache`` keyed on
``hashFiles('data/model_version.flag')``, which is the file that records the
model version and changes exactly when the model does.  A restored cache makes
``env/setup.sh`` a no-op, because the download script compares that version
against the copy inside ``data/detector/`` and exits early when they match.

Deliberately **no** ``restore-keys``.  A near-miss would restore a *different*
model version, and silently running against the wrong data is worse than
downloading.

**The download script retries.**  ``data/download_data_grand.py`` now attempts
the fetch up to four times with exponential backoff, and checks the received
size against the reported ``Content-Length`` — ``urlretrieve`` does not raise
on a truncated transfer, it returns a short file, which surfaces later as a
confusing tar error rather than as the network failure it was.

It also **downloads before deleting**.  The order used to be reversed: the
three directories were removed first, so a failed download left the
installation with no data at all rather than with the previous version.  That
is what turned a transient failure into a broken environment.

Action versions
---------------

GitHub deprecated the Node 20 runtime in September 2025, and every run was
carrying an annotation to that effect.  ``actions/checkout`` and
``actions/setup-python`` are pinned at ``@v7``: checkout moved to Node 24 at
v6, and v7 adds a guard against checking out a fork's pull request under
``pull_request_target`` or ``workflow_run``, neither of which these workflows
use.  Both are used with almost no inputs here — checkout with none at all —
so the majors carry no behaviour change for this repository.

Three sets of actions were deliberately **not** bumped.

``conda-incubator/setup-miniconda`` is at ``@v4``
    It was bumped a commit later than the others, and the reason is worth
    recording because the first attempt got it wrong.  It was initially left at
    v3 on the grounds that the deprecation had not flagged it — which was
    false.  The annotations list only the actions used by the jobs that
    actually ran, and the push in question had not triggered the notebook or
    conda jobs, so ``setup-miniconda`` never appeared in them.  It was on Node
    20 the whole time.

    The bump itself is small: v4.0.0's only breaking changes are the Node 24
    runtime and an internal ESM build.  Every input used here —
    ``environment-file``, ``activate-environment``, ``channels``,
    ``conda-remove-defaults`` — is unchanged, and v4 replaces ``conda config``
    subprocesses with direct ``.condarc`` writes, which is where these jobs
    spend their time.

    .. note::

       An earlier version of this page claimed the bump "measurably sped up"
       conda setup, citing the ROOT 6.36 leg going from 3 m 21 s to 2 m 51 s
       across it.  **Withdrawn.**  Measured over eight runs of unchanged code,
       that job takes anywhere from 102 s to 202 s.  A 30-second difference
       between two runs is well inside that spread and is evidence of nothing.

       The rule this repository has learned twice now: a difference between two
       runs is not a measurement.  Either it repeats, or it comes with a
       mechanism — a step that no longer runs, a download that no longer
       happens — or it is noise.  The timings quoted elsewhere on this page are
       step durations read from a job's own breakdown, which is why they can
       carry an argument where this one could not.

    The one input that did change is ``auto-activate-base``, now
    ``auto-activate``.  It was renamed in a **separate commit** from the
    version bump, so that if either broke the environment it would be obvious
    which.  The rename is a no-op: ``src/input.ts`` in the action resolves both
    names to a single option, preferring the old one when it is set, and the
    ``.condarc`` key it writes is chosen by the conda version rather than by
    which input name the workflow used.

``codecov/codecov-action`` is at ``@v7``
    Bumped last, once everything else was clean, so that if the upload
    misbehaved it would be the only change in flight.  v5 was the risky major —
    it moved to the Codecov CLI wrapper and renamed ``file`` to ``files`` — but
    this workflow already passed ``files``, so nothing here was affected.  v6
    brought Node 24 and v7 changes nothing functional.

.. warning::

   **Coverage is not actually reaching Codecov, and has not been.**  The upload
   step fails on every run:

   .. code-block:: text

       Upload queued for processing failed:
       {"message":"Token required because branch is protected"}

   The repository has no ``CODECOV_TOKEN`` secret — only ``PERSONAL_TOKEN`` and
   ``PYPI_TOKEN`` — and Codecov refuses a tokenless upload on a protected
   branch.  The failure is invisible because ``fail_ci_if_error: false``, which
   is the right setting: a coverage service being unreachable is not a reason
   to fail a test run.

   **The fix needs repository admin**: add ``CODECOV_TOKEN`` to the repository
   secrets.  The workflow already passes it, so nothing here changes when it
   appears.  Until then, coverage is available locally with
   ``pytest tests/ -q --cov=grand --cov-report=term``, and the badge in the
   README will not reflect reality.

The GitHub Pages actions in ``pages.yml``
    ``configure-pages``, ``upload-pages-artifact`` and ``deploy-pages`` are
    behind, but that workflow triggers only on ``main`` and has never run.
    Bumping them would be an untestable change.

``root_version.yml``
    Still on ``checkout@v3`` and ``setup-python@v3``, the furthest behind of
    all — left that way for a merge reason rather than a technical one.  Two
    branches, ``dev_io_root_testmerges`` and ``snonis_sim2root_test_merge``,
    *add* this file relative to their merge base: it did not exist when they
    diverged, and their copy is byte-identical to the one here.  While the
    copies agree that merges cleanly; the moment this one is edited it becomes
    an add/add conflict on both.  The bump waits until those branches land.

    This is the general rule for CI changes on ``dev-next``: ``lint.yml``,
    ``tests-conda.yml``, ``notebooks.yml`` and ``pages.yml`` were created here
    and are touched by no branch, so they can be changed freely.  Anything
    that predates the branch cannot.

Running the same checks locally
-------------------------------

.. code-block:: bash

    conda env create -f env/conda/grand-dev.yml --solver=libmamba
    conda activate grand-dev
    source env/setup.sh
    pip install -e .

    pytest tests/ -q                       # the suite
    ruff check grand/ tests/ quality/ notebooks/ docs/dev/   # lint + the ratchet
    cd docs && make html                   # the documentation, warning-free
    python notebooks/make_notebooks.py     # rebuild and execute the notebooks

The documentation build should print no ``WARNING`` or ``ERROR`` lines apart
from the ROOT CPU-feature diagnostic described above.  Do not add ``-W``: on a
machine where ROOT emits that diagnostic it fails the build for a reason
unrelated to the documentation.

Publishing the documentation
-----------------------------

``pages.yml`` builds this manual and deploys it to GitHub Pages.  It has never
run, for two independent reasons, and both have to be dealt with:

1. **Pages is not enabled** on ``grand-mother/grand``.  The API returns 404 for
   the Pages endpoint, so ``deploy-pages`` has nothing to deploy to.
2. **The trigger names a branch GitHub cannot see.**  The workflow fires on
   ``push`` to ``main`` and on ``workflow_dispatch``, but the repository's
   default branch is ``master``, and GitHub only offers ``workflow_dispatch``
   for workflows that exist on the default branch.  ``pages.yml`` lives on
   ``dev-next`` alone, so nothing can start it.

There are two ways to publish, and they are not alternatives so much as a
stopgap and a destination.

**From the collaboration repository.**  Someone with admin on
``grand-mother/grand`` sets *Settings → Pages → Source* to **GitHub Actions**.
Once ``dev-next`` becomes the default branch in Phase 9 the existing trigger
matches and the manual republishes on every push, at
``https://grand-mother.github.io/grand/``.  Before then, add ``dev-next`` to
the workflow's ``branches`` list.

**From a personal fork, in the meantime.**  A fork is a repository you are
admin of, which removes both blockers without anyone's permission:

.. code-block:: bash

    gh repo fork grand-mother/grand --clone=false --remote-name fork
    git push fork dev-next
    gh repo edit <you>/grand --default-branch dev-next
    gh api -X POST repos/<you>/grand/pages -f build_type=workflow
    gh workflow run pages.yml --repo <you>/grand --ref dev-next

Setting the fork's default branch to ``dev-next`` is what makes the last line
work: it puts ``pages.yml`` where GitHub looks for dispatchable workflows.  The
result is a complete, current manual at ``https://<you>.github.io/grand/``,
which is enough to circulate a link — but say plainly that it is a preview of a
branch, not a second home for the project.

Neither route changes a line of the documentation.  The manual that publishes
is the one ``cd docs && make html`` builds locally.
