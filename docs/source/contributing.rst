Contributing
============

.. contents::
   :local:
   :depth: 1

This page is about how to work in this repository, not about the physics.  It
exists because most of what follows is convention rather than configuration,
and conventions that are not written down get broken by people who had no way
of knowing.

Setting up
----------

.. code-block:: bash

    conda env create -f env/conda/grand-dev.yml --solver=libmamba
    conda activate grand-dev
    source env/setup.sh
    pip install -e . --no-deps --no-build-isolation

``env/setup.sh`` compiles the TURTLE and GULL C extensions and downloads the
data models, so it is not optional and it is not fast the first time.  It needs
``make``, which the environment file does not declare; install it from your
distribution if the build stops immediately.  See :doc:`installation` for the
long form and :doc:`data_files` for what gets downloaded.

The checks, and how to run them
-------------------------------

Everything CI runs, you can run.  Nothing here needs a container.

.. code-block:: bash

    python -m pytest tests/ -q                          # the suite
    ruff check grand/ tests/ quality/ notebooks/ docs/dev/
    cd docs && make html                                # the documentation
    python notebooks/make_notebooks.py                  # the notebooks
    python quality/docstring_coverage.py                # docstring coverage

The lint scope is exactly what the CI job checks.  ``sim2root/``, ``granddb/``,
``examples/`` and ``src_outlib/`` are **not** linted — see :doc:`sim2root` for
why, and do not take their style as a model.

The lint ratchet
----------------

``pyproject.toml`` carries a ``per-file-ignores`` table that is a **ratchet**:

    Both lists may shrink and must never grow.  A module converted to numpydoc
    loses its ``D``; a file cleaned of a rule loses that rule.  New entries are
    not added — new code is written clean.

The table exists because CI had not run for years and the package accumulated
336 findings in code that had never been checked.  Recording them let the lint
job start green and become a required check.  Adding to it would defeat the
point.

If your change makes a listed file clean of a listed rule, delete that rule
from its line in the same commit.  That is the mechanism by which the list
empties.

Docstrings
----------

**numpydoc, on every function**, including private ones.  A description, then
``Parameters`` and ``Returns`` where the function has parameters or returns
something, then ``Examples`` where an example earns its keep.

House style, which differs from the numpydoc default in one place:

- Summaries are third person — "Returns the effective length", not "Return the
  effective length".  ``D401`` is disabled for this reason; do not re-enable
  it.
- Do not mix in the legacy ``:param:``/``:type:``/``:return:`` fields.  85
  docstrings carried both at one point, which duplicated the content and broke
  the rendering of several.
- ``.. versionadded::`` and ``.. versionchanged::`` when behaviour changes, with
  the reason.

``python quality/docstring_coverage.py`` reports where you stand.

Tests
-----

Write the test with the change, not after it.  Beyond that, three conventions
that are particular to this repository:

**Build fixtures, do not ship them.**  ``data/`` is gitignored, so a test that
reads a checked-in ROOT file cannot run in CI.
``tests/sim/test_pipeline_end_to_end.py`` constructs its input from the tree
classes instead: it costs nothing in repository size, cannot drift from the
schema, and puts the contents of the fixture in front of the reader.

**Measure, then assert — and say which you did.**  Where a value is disputed,
a test that asserts the disputed value just encodes one side.  Several tests
here measure a ratio and assert only the parts no convention can change; the
module docstring then records the measurement with its date.
``tests/sim/test_galactic_noise_normalisation.py`` is the worked example.

**Seed every random draw, through a local generator.**  ``np.random.default_rng(0)``,
not ``np.random.seed(0)``, so a test does not disturb global state that another
test depends on.  An unseeded draw here failed about one run in six.

Expected failures are a record, not a silencer.  ``tests/conftest.py`` holds a
``KNOWN_FAILURES`` table with a reason per entry, applied with
``strict=False``; the reason is the part that matters, and an xfail that starts
passing is information.  See :doc:`testing`.

Notebooks
---------

**The notebooks are generated.**  Edit ``notebooks/make_notebooks.py``, never
the ``.ipynb`` — anything written into a notebook by hand is lost on the next
rebuild.

.. code-block:: bash

    python notebooks/make_notebooks.py                # rebuild and execute all
    python notebooks/make_notebooks.py --only 03,05   # just those
    python notebooks/make_notebooks.py --no-execute   # while drafting

The build refuses to finish if a notebook fails to execute, comes back without
stored outputs, or is left on disk not matching the generator.  Commit the
executed notebooks: their stored outputs are what a reader sees on GitHub.

They are tutorials, so comment the code cells generously.  A cell that shows
only what to type teaches less than one that says why.

Documentation
-------------

``docs/source/`` is the whole tree; ``make html`` builds it and it should build
with **zero warnings**.  Prose pages carry executable examples through
``.. jupyter-execute::``, so an example that stops working fails the build.

Diagrams are generated too, by ``docs/dev/make_*_diagram.py``, and are
committed as SVG.  Embed them so they can be opened full size:

.. code-block:: rst

    .. image:: _static/pipeline.svg
       :target: _static/pipeline.svg
       :alt: what the diagram shows, for a reader who cannot see it
       :width: 100%

The handbook section under ``docs/source/handbook/`` is generated from
``resources/GRANDlib_Handbook.zip`` by ``docs/dev/build_handbook.py``.  Do not
edit those pages; corrections go in the ``ERRATA`` table in that script.

Editing source with scripts
---------------------------

Twice in this branch a regex over Python source corrupted a file that an
AST-bounded edit would not have: once inserting a docstring into the middle of
a ``for`` loop, once dropping the newline before a function body.

If you are editing many docstrings or signatures mechanically, use ``ast`` to
find the line range and edit only within it, and prefer whole-line deletion
over reconstructing a string literal — that way quoting and escaping cannot
change.  ``python -c "import ast; ast.parse(open(f).read())"`` on every file you
touched, before you commit.

Branches and merging
--------------------

Work goes to ``dev-next``, then to ``dev``, then to ``master``.  Branch names in
this repository are ``dev_<topic>`` by convention, sometimes with an author
suffix.

A clean textual merge is not a compatible merge.  Run the pre-merge check:

.. code-block:: bash

    python quality/premerge_check.py <branch> [<branch> ...]

It looks for the two static ways branches here have been found to conflict
without conflicting — two names for one quantity, and two implementations of
one thing in different files.  The third way, a change of meaning under an
unchanged name, only running the code detects; that is what the numeric tests
are for.

Commits
-------

Say what changed and why, and state measurements rather than impressions.  If a
commit corrects something an earlier commit or a document asserted, say so
plainly and give the number — several entries in :doc:`known_issues` exist
because a commit message did that.
