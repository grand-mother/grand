Contributing
============

.. contents::
   :local:
   :depth: 1

The conventions for changing GRANDlib: setting up, the checks a change must
pass, and how code, tests, notebooks and documentation are written.

Setting up
----------

.. code-block:: bash

    conda env create -f env/conda/grand-dev.yml --solver=libmamba
    conda activate grand-dev
    source env/setup.sh
    pip install -e . --no-deps --no-build-isolation

``env/setup.sh`` compiles the TURTLE and GULL C extensions and downloads the
model data; see :doc:`installation` and :doc:`data_files`.

The checks, and how to run them
-------------------------------

Everything CI runs, you can run.  Nothing here needs a container.

.. code-block:: bash

    python -m pytest tests/ -q                          # the suite
    ruff check grand/ tests/ quality/ notebooks/ docs/dev/ granddb/
    cd docs && make html                                # the documentation
    python notebooks/make_notebooks.py                  # the notebooks
    python quality/docstring_coverage.py                # docstring coverage

The lint scope is the one CI checks.  ``granddb/`` ships with the package, so
it is linted too; its older findings are recorded in the ratchet below, so do
not take its current style as a model.  ``sim2root/``, ``examples/`` and
``src_outlib/`` are not linted yet (:doc:`sim2root`).

The lint ratchet
----------------

``pyproject.toml`` carries a ``per-file-ignores`` table that is a **ratchet**:

    Both lists may shrink and must never grow.  A module converted to numpydoc
    loses its ``D``; a file cleaned of a rule loses that rule.  New entries are
    not added: new code is written clean.

The table records the findings in code written before linting was enforced,
so that the lint job could become a required check.

If your change makes a listed file clean of a listed rule, delete that rule
from its line in the same commit.  That is the mechanism by which the list
empties.

Checking input
--------------

A public function checks what it is given, using :mod:`grand.basis.validate`,
and refuses bad input at the door rather than failing deep inside or, worse,
returning a plausible wrong number.  The helpers convert and check in one line
and write the message, which starts with ``GRANDlib:`` and names the function
and the argument::

    from grand.basis import validate as _validate

    def fit(Xants, tants, sigma=None):
        where = "fit"
        Xants = _validate.as_array(Xants, "Xants", where, shape=(None, 3), finite=True)
        sigma = _validate.positive(_validate.as_real(sigma, "sigma", where), "sigma", where, "s")

Raise ``TypeError`` for the wrong kind of value and ``ValueError`` for a value
of the right kind but out of range or shape; warn with
:func:`grand.basis.validate.warn` for input that is suspicious but usable.
Never ``print`` and return ``None``, never call ``exit()``, and do not use
``assert`` for user input: Python skips asserts under ``-O``, and their message
says nothing.  Asserts remain fine for internal invariants.

Docstrings
----------

**numpydoc, on every function**, including private ones.  A description, then
``Parameters`` and ``Returns`` where the function has parameters or returns
something, then ``Examples`` where an example earns its keep.

House style, which differs from the numpydoc default in one place:

- Summaries are third person: "Returns the effective length", not "Return the
  effective length".  ``D401`` is disabled for this reason; do not re-enable
  it.
- Do not mix in the legacy ``:param:``/``:type:``/``:return:`` fields.  85
  docstrings carried both at one point, which duplicated the content and broke
  the rendering of several.
- ``.. versionadded::`` and ``.. versionchanged::`` when behavior changes, with
  the reason.

``python quality/docstring_coverage.py`` reports where you stand.

Tests
-----

Write the test with the change, not after it.  Beyond that, three conventions
that are particular to this repository:

**Use the committed samples, or build the input.**  ``data/`` is not in
version control, so a test cannot read a file from it.  Use the samples under
``sim2root/``, or build the input from the tree classes, as
``tests/sim/test_pipeline_end_to_end.py`` does.

**Assert what no convention can change.**  Where a value is disputed, a test
that asserts it encodes one side of the dispute.  Assert the properties that
hold either way, and record the measured value, with its date, in the
docstring; ``tests/sim/test_galactic_noise_normalisation.py`` is an example.

**Seed every random draw, through a local generator.**  ``np.random.default_rng(0)``,
not ``np.random.seed(0)``, so a test does not disturb global state that another
test depends on.

Expected failures are a record, not a silencer.  ``tests/conftest.py`` holds a
``KNOWN_FAILURES`` table with a reason per entry, applied strictly: the reason
is the part that matters, and an xfail that starts passing fails the run until
its entry is removed.  See :doc:`testing`.

Notebooks
---------

``notebooks/make_notebooks.py`` writes the notebooks, so a change made only in
an ``.ipynb`` is lost on the next rebuild.  Edit the generator, or edit the
notebook in Jupyter and run ``python notebooks/import_notebook.py
<notebook>``, which brings the change into the generator and executes it
(:doc:`notebooks`).

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

When editing many docstrings or signatures with a script, use :mod:`ast` to
find each line range and edit only within it; a regular expression over
Python source can land inside a loop or a string.  Check every file you
touched with ``python -c "import ast; ast.parse(open(f).read())"`` before
committing.

Branches and merging
--------------------

Work goes to ``dev-next``, then to ``dev``, then to ``master``.  Branch names in
this repository are ``dev_<topic>`` by convention, sometimes with an author
suffix.

A clean textual merge is not a compatible merge.  Run the pre-merge check:

.. code-block:: bash

    python quality/premerge_check.py <branch> [<branch> ...]

It looks for the two static ways branches here have been found to conflict
without conflicting: two names for one quantity, and two implementations of
one thing in different files.  The third way, a change of meaning under an
unchanged name, only running the code detects; that is what the numeric tests
are for.

Commits
-------

Say what changed and why, with measurements where there are any.  A commit
that corrects an earlier commit or a document says so, and gives the number.
