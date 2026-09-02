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

**The documentation build compiles the Handbook first.**  ``make html`` depends
on ``docs/dev/build_handbook_pdf.py``, because the pages link to the PDF with
``:download:`` and a missing target is itself a warning.  Both the ``docs`` job
and the separate ``handbook`` job install a LaTeX toolchain for this.

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
    spend their time.  Conda setup measurably sped up: the ROOT 6.36 leg went
    from 3 m 21 s to 2 m 51 s across the bump.

    The one input that did change is ``auto-activate-base``, now
    ``auto-activate``.  It was renamed in a **separate commit** from the
    version bump, so that if either broke the environment it would be obvious
    which.  The rename is a no-op: ``src/input.ts`` in the action resolves both
    names to a single option, preferring the old one when it is set, and the
    ``.condarc`` key it writes is chosen by the conda version rather than by
    which input name the workflow used.

``codecov/codecov-action@v4``
    Left behind deliberately.  It runs in one step of one job, uploads
    coverage, and its majors have changed the tokenless-upload behaviour
    before; there is nothing to gain by moving it in the same change as the
    runtime fix.

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
