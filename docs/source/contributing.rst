Contributing
============

.. contents::
   :local:
   :depth: 1

The conventions for changing GRANDlib: setting up, the checks a change must
pass and how code, tests, notebooks and documentation are written.

Setting up
----------

Install from a clone as described in :doc:`installation`, then add an
editable install of the package::

    pip install -e . --no-deps --no-build-isolation

The checks and how to run them
-------------------------------

.. code-block:: bash

    python -m pytest tests/ -q                          # the suite
    ruff check grand/ tests/ quality/ notebooks/ docs/dev/ granddb/
    cd docs && make html                                # the documentation
    python notebooks/make_notebooks.py                  # the notebooks
    python quality/docstring_coverage.py                # docstring coverage

This is the lint scope CI checks; ``sim2root/``, ``examples/`` and
``src_outlib/`` are not linted yet.

The lint ratchet
----------------

``per-file-ignores`` in ``pyproject.toml`` lists the lint findings in code
written before linting was enforced.  The list may shrink and must never
grow: new code is written clean and a change that cleans a listed file of a
rule removes that rule from its line.

Checking input
--------------

A public function checks what it is given, using :mod:`grand.basis.validate`
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
Never ``print`` and return ``None``, never call ``exit()`` and do not use
``assert`` for user input: Python skips asserts under ``-O``, where their message
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
- Do not use the legacy ``:param:``/``:type:``/``:return:`` fields.
- ``.. versionadded::`` and ``.. versionchanged::`` when behavior changes, with
  the reason.

``python quality/docstring_coverage.py`` reports where you stand.

Tests
-----

Write the test with the change.  :doc:`testing` lists the conventions:
committed samples or built inputs instead of files in ``data/``, properties
over stored values, seeded random draws and an entry in
``tests/conftest.py`` for a known defect instead of a skip.

Notebooks
---------

The notebooks are written by ``notebooks/make_notebooks.py``, which holds
their source, executes each one and stores its outputs.  A change made only in
the ``.ipynb`` is lost the next time the notebooks are rebuilt.  To keep it,
edit the notebook in Jupyter as usual, then bring it back into the generator:

.. code-block:: bash

    python notebooks/import_notebook.py notebooks/05_galactic_noise.ipynb

A new notebook is added the same way.  Give it the next free number, start it
with a ``# NN — Title`` heading and keep its data paths relative to
``notebooks/``:

.. code-block:: bash

    python notebooks/import_notebook.py ~/my_analysis.ipynb --name 13_my_analysis.ipynb

The script checks that every code cell compiles and that no cell uses an
absolute path such as ``/home/...``, which would not exist on another machine.
It then writes the notebook's block in ``make_notebooks.py``, executes the
notebook and stores its outputs.  If anything fails, it restores both files
and says why.  For a new notebook, it also reminds you to add it to the list
in :doc:`notebooks`.  Commit ``make_notebooks.py`` and the ``.ipynb`` together.

To rebuild notebooks from the generator directly::

    python notebooks/make_notebooks.py                # rebuild and execute all
    python notebooks/make_notebooks.py --only 03,05   # just those two
    python notebooks/make_notebooks.py --check        # check, writing nothing

CI fails if a committed notebook does not match the generator or does not
execute.

Comment the notebooks' code cells: they are tutorials.

Documentation
-------------

``docs/source/`` is the whole tree; ``make html`` builds it and it should build
with **zero warnings**.  Prose pages carry executable examples through
``.. jupyter-execute::``, so an example that stops working fails the build.

Diagrams are generated too, by ``docs/dev/make_*_diagram.py`` and are
committed as SVG.  Embed them so they can be opened full size:

.. code-block:: rst

    .. image:: _static/pipeline.svg
       :target: _static/pipeline.svg
       :alt: what the diagram shows, for a reader who cannot see it
       :width: 100%

The handbook section under ``docs/source/handbook/`` is generated from
``resources/GRANDlib_Handbook.zip`` by ``docs/dev/build_handbook.py``.  Do not
edit those pages; corrections go in the ``ERRATA`` table in that script.

Writing style
-------------

The documentation and the docstrings are written in American English, in
plain sentences that say what the code does now.  In practice:

* American spelling: *behavior*, *normalization*, *meters*, *toward*.
* No comma before *and*: ``A, B and C``.  Where two clauses would be joined
  by `` and``, write two sentences.
* State the fact.  Leave out ``it is worth noting``, ``said plainly``, ``why
  it matters`` and the like.
* Describe the current behavior, not its history: no issue numbers, no ``it
  used to``, no dates except those a reader needs.  The history belongs in the
  changelog and the commit messages.
* Give the unit of every quantity and the frame of every position or
  direction.

``python docs/dev/check_style.py`` checks the pages for the first three.  The
lint job runs it on every push.

Editing source with scripts
---------------------------

When editing many docstrings or signatures with a script, use :mod:`ast` to
find each line range and edit only within it; a regular expression over
Python source can land inside a loop or a string.  Check every file you
touched with ``python -c "import ast; ast.parse(open(f).read())"`` before
committing.

The sim2root converters
-----------------------

``sim2root/README.md`` documents the converters in detail.  Their code is not
yet linted and the tests check them end to end only, by converting the
committed samples (``tests/sim2root/``).  After a change, convert a sample and
read the result back with :mod:`grand.dataio`.  ``Common/raw_root_trees.py``
defines the RawRoot format separately from ``grand/dataio``, so a field added
to one must be added to the other.  Edit the files under ``sim2root/``, not
the stale copy in ``src_outlib/``, which nothing imports.

Branches and merging
--------------------

Work goes to ``dev-next`` through pull requests.  Before merging a branch
that touches the data format or the simulation, run
``python quality/premerge_check.py <branch>``: it finds two names for one
quantity and two implementations of one thing, that a clean merge would not
reveal.

Commits
-------

Say what changed and why, with measurements where there are any.  A commit
that corrects an earlier commit or a document says so and gives the number.
