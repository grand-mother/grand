Test suite
==========

The test suite is the set of automated checks, in ``tests/``, that verify
GRANDlib does what it should: that its physics agrees with independent
calculations, that files it writes read back unchanged, that its documented
commands work and that a fixed bug stays fixed.  Running it tells you whether
your installation works and whether a change you made broke something.  It
uses `pytest <https://docs.pytest.org>`_ and runs automatically on every
change (:doc:`ci`).

.. contents::
   :local:
   :depth: 1

Running it
----------

From the repository root, in an environment set up as in :doc:`installation`:

.. code-block:: bash

    pytest tests/ -q

The suite has about 1270 tests and takes about 20 minutes on one core.  It
needs the compiled TURTLE and GULL libraries and the model data, which
``source env/setup.sh`` provides.  Every test writes into a temporary folder,
so two runs can share a checkout.

A single area runs on its own:

.. code-block:: bash

    pytest tests/geo -q                                # coordinates, terrain, field
    pytest tests/sim/test_pipeline_golden.py -q        # the whole chain

What a run reports
------------------

*Skipped* tests need something the environment lacks, such as an optional
package or a sample file; each says what in its skip reason.

*Expected failures* (``xfailed``) are known defects that need a decision
rather than a patch.  ``tests/conftest.py`` lists them, with the reason for
each and who can settle it; the corresponding entries are in
:doc:`known_issues`.  They are strict: a listed test that starts passing fails
the run until its entry is removed.

Layout
------

``tests/`` mirrors the package: ``tests/geo``, ``tests/dataio``,
``tests/sim``, ``tests/aoi``, ``tests/analysis``, ``tests/basis`` and
``tests/granddb``.  Three directories test what lies outside the package:
``tests/sim2root`` runs the converters on the committed samples,
``tests/scripts`` the command-line tools and ``tests/examples`` the event
viewer.  Files at the top level test properties of the whole package: input
validation, lazy imports, packaging and the commands and recipes the
documentation gives.

What the tests check
--------------------

**Physics against independent calculations.**  The Galactic-noise level is
rebuilt from the shipped tables and the antenna impedance, independently of
the module and compared with the simulated noise
(``tests/sim/test_galactic_noise_normalisation.py``).  The arrival direction
recomputed from the position of Xmax is compared with the ZHAireS summaries
(``tests/geo/test_angle_convention.py``).  The arm of each effective-length
table is identified by correlating its pattern with the named HFSS arms
(``tests/sim/test_antenna_arm_identity.py``).

**Properties that must hold exactly.**  Parseval's theorem on every noise
trace, frame conversions that return their input, the same noise from the same
seed.

**Contracts.**  ``tests/dataio/test_schema_snapshot.py`` compares the layout of
every ROOT tree with a stored snapshot, so a change to the file format appears
in review.  ``tests/dataio/test_backward_compatibility.py`` reads files written
in 2024.

**Regressions.**  ``tests/sim/test_pipeline_golden.py`` runs the whole chain on
a fixed input and seed and compares the result with a stored reference, to
1e-6 of the trace peak.  It shows that the answer has not changed, not that it
is right.  When a change is meant to alter the answer, regenerate the
reference with ``python tests/sim/test_pipeline_golden.py --write`` and say why
in the commit message.

**The documentation.**  ``tests/test_documented_commands.py`` and
``tests/test_documented_recipes.py`` run the commands and code that
:doc:`commands`, :doc:`sim2root` and :doc:`datamodel` give.  The executed
examples of the other pages run when the documentation is built.

:doc:`validation` shows the checks against independent calculations as
figures.

Coverage
--------

.. code-block:: bash

    pytest tests/ -q --cov=grand --cov=granddb --cov-report=term

On 1 October 2026 the suite covered 80% of the lines of ``grand/`` and 23% of
``granddb/``.  Line coverage counts lines executed, not results checked, so it
overstates how well a module is tested.  ``sim2root/``, ``examples/`` and
``src_outlib/`` are not measured.

Writing a test
--------------

* Prefer a property that holds for any correct implementation to a stored
  value: it needs no reference and survives a rewrite.
* Where a reference is needed, prefer one from the GRANDlib paper or an
  independent code to one produced by GRANDlib itself.
* With random input, fix the seed with a local generator
  (``np.random.default_rng(seed)``) and assert on distributions rather than
  on single draws.
* When testing an optimized path against a plain one, first assert that the
  optimized path ran.
* Write files into pytest's ``tmp_path``, never into the repository.
* For a known defect that needs a decision, add the test with an entry in
  ``tests/conftest.py`` and in :doc:`known_issues`, rather than skipping it.
