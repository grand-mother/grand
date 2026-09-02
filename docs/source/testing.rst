Testing
=======

.. contents::
   :local:

Running the suite
-----------------

.. code-block:: bash

    conda activate grand-dev
    source env/setup.sh          # the tests need the compiled C libraries
    pytest tests/ -q

Current state, on the ``dev-next`` branch:

.. code-block:: text

    343 passed, 14 skipped, 8 xfailed
    coverage: 64%

Layout
------

``tests/`` mirrors the package.  ``tests/geo/`` is the oldest and best
covered; ``tests/dataio/`` and ``tests/aoi/`` arrived together and are the
largest; ``tests/sim/`` is the thinnest relative to what it guards.

Known failures are marked, not hidden
-------------------------------------

Eight tests fail for reasons that need a decision rather than a patch.  They
are registered in ``tests/conftest.py`` with the reason for each and who can
settle it, so that the suite can be a required check in CI — a permanently red
gate is a gate nobody looks at.

They are marked ``xfail`` with ``strict=False``, so a test that starts passing
is reported as ``xpassed`` rather than failing the run.  That is the signal to
delete its entry, and it earns its keep: ``test_antenna`` xpassed as soon as
the missing-frame guard was fixed.

Read the registry rather than a summary of it — it is the authority, it is
short, and it says what each failure actually is.

What the tests are for
----------------------

Three kinds, and the distinction matters when adding more.

**Invariants** hold under any correct implementation and survive a rewrite.
``tests/sim/test_galactic_noise_normalisation.py`` asserts that the time
series carries the energy of the spectrum it came from, which is true whatever
normalisation convention is chosen — so it will still be valid whichever way
the open question about that convention is settled.

**Contracts** pin something that other people depend on.
``tests/dataio/test_schema_snapshot.py`` records the ROOT tree layout and
fails until the snapshot is regenerated, which puts a schema change in the
diff where a reviewer sees it instead of letting it arrive silently.  It also
carries a guard against a specific collision: two branches adding the same
NUTRIG quantity under different names.

**Regressions** pin a bug so it cannot come back.
``tests/dataio/test_descriptor_defaults.py`` constructs every tree class with
no arguments, which was impossible under NumPy 2 until recently.

Writing a new test
------------------

Prefer an invariant to a stored value where one exists: it needs no reference
and does not have to be regenerated. Where a stored reference is needed,
prefer one from the published paper — its figures fix the local sidereal time
of maximum Galactic noise per port, the RF-chain transfer function, and an
end-to-end case — over a number produced by the code itself, which only says
the code still does what it did.

Assert on distributions rather than individual random draws.  A per-case
assertion on random input is brittle: one case drifts, the suite goes red, and
the fix is to loosen the bound until it catches nothing.

If an optimisation is being tested, assert first that the fast path actually
ran.  A comparison between an optimised route and a plain one proves nothing
if the optimisation quietly declined — both sides then execute the same code
and agree exactly, which reads as success.

Coverage
--------

Measured with:

.. code-block:: bash

    pytest tests/ -q --cov=grand --cov-report=term

64% today.  The number is worth less than it looks: line coverage counts
executed lines, not verified behaviour, and the end-to-end test executed a
great deal of the simulation chain while asserting only that an output file
appeared.  Appendix C of the GRANDlib paper reports 84%, measured when CI
still ran; that figure carried the same caveat.

Known gaps
----------

The end-to-end test cannot run at all on a fresh checkout, because its input
file is not in version control and is not downloaded by ``env/setup.sh``.
See :ref:`issue-missing-endtoend-fixture`.  A replacement that builds its own
input, ``tests/sim/test_pipeline_end_to_end.py``, covers the same ground and
does run.

**The suite cannot be run twice at once.**  Several tests write to fixed paths
under ``data/`` -- ``test_voltage.root``, ``test_voltage1.root`` -- rather than
to a temporary directory, so two concurrent runs, or ``pytest -n auto``,
occasionally fail on a file another process is writing.  It shows up as a
single failure that disappears on a rerun, which is the most misleading shape
a test failure can take.  New tests should use pytest's ``tmp_path``, as
``test_pipeline_end_to_end.py`` does.
