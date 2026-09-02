sim2root: the input boundary
============================

.. contents::
   :local:
   :depth: 1

GRANDlib starts from an electric field.  Producing that field — simulating the
air shower and its radio emission — is done by ZHAireS or CoREAS, neither of
which is part of GRANDlib.  ``sim2root/`` is the seam between them: it converts
the output of those codes into the GRAND schema so the rest of the library can
read it.

It is in this repository but it is not part of the ``grand`` package.  It is
not imported by it, not covered by its tests, and not checked by its linter.
This page says what it does and what state it is in, because using it without
knowing either is how people lose an afternoon.

Where it sits
-------------

.. code-block:: text

    ZHAireS / CoREAS                       an air-shower simulation
        |
        |   CoreasToRawROOT.py  /  ZHAireSRawToRawROOT.py
        v
    RawRoot                                a common intermediate format
        |
        |   Common/sim2root.py
        v
    GRANDRoot                              TRun, TEfield, TShower  ->  grand

Two steps, deliberately.  *RawRoot* exists so that the two shower codes reach a
common format before anything GRAND-specific happens; ``sim2root.py`` then
produces the trees :doc:`datamodel` describes.

Running it
----------

From a CoREAS simulation directory:

.. code-block:: bash

    cd sim2root/CoREASRawRoot
    python3 CoreasToRawROOT.py proton/

From a ZHAireS one, where the long form takes the identifiers explicitly and
the short form works them out:

.. code-block:: bash

    cd sim2root/ZHAireSRawRoot
    python3 ZHAireSRawToRawROOT.py <InputDirectory> standard <RunID> <EventID> <Output>
    python3 ZHAireSRawToRawROOT.py <InputDirectory>

Then, in either case:

.. code-block:: bash

    python3 sim2root/Common/sim2root.py <path>/*.rawroot -d 20221026 -t 180000 -e DC2Alpha

``sim2root.py --help`` lists the rest.  ``sim2root/README.md`` is the
authoritative usage document and is kept by the people who wrote the
converters.

What is in it
-------------

About 8600 lines, none of it small:

=========================================================  ======  ================================
File                                                       Lines   What it does
=========================================================  ======  ================================
``ZHAireSRawRoot/AiresInfoFunctionsGRANDROOT.py``           2095   Reads ZHAireS output files
``Common/raw_root_trees.py``                                1434   The RawRoot schema
``ZHAireSRawRoot/ZHAireSRawToRawROOT.py``                   1049   ZHAireS to RawRoot
``Common/sim2root.py``                                      1011   RawRoot to GRANDRoot
``CoREASRawRoot/CoreasToRawROOT.py``                         593   CoREAS to RawRoot
``Common/IllustrateSimPipe.py``                              571   Plots for the pipeline example
``ZHAireSRawRoot/ZHAireSInputGenerator.py``                  492   Generates ZHAireS inputs
``Common/EventParametersGenerator.py``                       342   Event parameter files
``CoREASRawRoot/CorsikaInfoFuncs.py``                        335   Reads CORSIKA output
``ZHAireSRawRoot/ZHAireSCompressEvent.py``                   214   Compresses an event
``Common/RunSimPipe*.py``                                    310   Three pipeline examples
=========================================================  ======  ================================

``Common/raw_root_trees.py`` is worth knowing about independently: it is a
second schema, parallel to ``grand/dataio``, describing the intermediate
format.  A schema change on one side does not automatically reach the other.

State of the code
-----------------

Said plainly, because the alternative is finding out by accident.

**It is outside the quality gates.**  The CI lint job checks ``grand/``,
``tests/``, ``quality/``, ``notebooks/`` and ``docs/dev/``.  It does not check
``sim2root/``, and neither does the test suite: there are no tests for any of
the files above.

**Ruff reports about 900 findings there**, against zero in the gated scope.
The largest groups are ``F405`` (371, names possibly undefined from star
imports), ``D103`` (108, missing docstrings) and ``F821`` (99, undefined
names).

**Ninety-eight of those undefined names are in one block.**
``ZHAireSRawToRawROOT.py`` has a longitudinal-tables section guarded by
``if(NLongitudinal):`` that calls ``SimShower.*`` and ``HDF5handle``, neither of
which exists anywhere in the file — they are leftovers from an HDF5-based
predecessor.  The block is dead as written, because ``NLongitudinal=False`` is
hard-coded at line 85 and the parameter that set it is commented out of the
signature above it.  A commented usage example further down the same file
passes ``NLongitudinal=True``; doing that would raise ``NameError`` on the
first call.  The comment above the block says "not implemented yet", which is
accurate.

None of this means the converters are wrong — they are what produced the Data
Challenge datasets.  It means they have not been through the same cleanup as
the package, so treat a change there as unguarded: nothing will tell you if you
break it.

If you work on it
-----------------

- Read ``sim2root/README.md`` first; it is more current than the Handbook.
- Test by round-tripping.  Convert, then read the result back with
  ``grand.dataio`` and check the fields you touched.  Notebook 02 shows the
  reading side, and ``tests/sim/test_pipeline_end_to_end.py`` shows how to
  build a small file from the tree classes rather than shipping one.
- If you add a field on one side, check the other.  The RawRoot and GRANDRoot
  schemas are maintained separately, and
  :ref:`issue-nutrig-field-names` is what happens when two branches name one
  quantity twice.
- Adding ``sim2root/`` to the lint gate would be worth doing, but not by
  recording 900 findings in the ratchet — that would defeat the ratchet's
  purpose.  The undefined-name block is the place to start, since it is the
  only group that is a latent error rather than a style finding.
