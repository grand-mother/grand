Converting simulations: sim2root
================================

.. contents::
   :local:
   :depth: 1

GRANDlib starts from an electric field.  The air shower and its radio
emission are simulated by ZHAireS or CoREAS, outside GRANDlib.  The converters
in ``sim2root/`` turn the output of those codes into GRAND's ROOT format, which
the rest of the library reads.

``sim2root/`` is in this repository but is not part of the ``grand`` package:
it is not imported by it, and its code is not linted.  The tests run the
converters on the committed samples.

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

The first step brings both shower codes to a common intermediate format,
*RawRoot*; ``sim2root.py`` then writes the trees :doc:`datamodel` describes.

Running it
----------

From a CoREAS simulation directory:

.. code-block:: bash

    cd sim2root/CoREASRawRoot
    python3 CoreasToRawROOT.py -d proton -o converted

which writes ``converted/Coreas_004100.rawroot``.  Without ``-o`` the output
goes to the current folder, where an existing file is refused (here, the
committed sample) unless ``--overwrite`` is given.

From a ZHAireS one, where the long form takes the identifiers explicitly and
the short form works them out:

.. code-block:: bash

    cd sim2root/ZHAireSRawRoot
    python3 ZHAireSRawToRawROOT.py <InputDirectory> standard <RunID> <EventID> <Output>
    python3 ZHAireSRawToRawROOT.py <InputDirectory>

Then, in either case:

.. code-block:: bash

    python3 sim2root/Common/sim2root.py <path>/*.rawroot -sl GP300 -d 20221026 -t 180000 -e DC2Alpha \
        --trigger_time_ns 800 --target_duration_us 4.096

``-sl``, the site layout, is required.  A run stores one trace window, so when
the input showers were simulated with different windows -- as the two committed
ZHAireS samples were -- ``--trigger_time_ns`` and ``--target_duration_us`` give
them a common one; without them ``sim2root.py`` stops and says so.  The output
is a folder named ``sim_<site>_<date>_<time>_RUN<run>_CD_<extra>_<serial>``,
made in the current directory or in the one ``-o`` names.

``sim2root.py --help`` lists the rest.  ``sim2root/README.md`` is the
authoritative usage document and is kept by the people who wrote the
converters.

What is in it
-------------

About 8600 lines:

=========================================================  ======  ================================
File                                                       Lines   What it does
=========================================================  ======  ================================
``ZHAireSRawRoot/AiresInfoFunctionsGRANDROOT.py``           2095   Reads ZHAireS output files
``Common/raw_root_trees.py``                                1434   The RawRoot schema
``ZHAireSRawRoot/ZHAireSRawToRawROOT.py``                   1049   ZHAireS to RawRoot
``Common/sim2root.py``                                      1122   RawRoot to GRANDRoot
``CoREASRawRoot/CoreasToRawROOT.py``                         733   CoREAS to RawRoot
``Common/IllustrateSimPipe.py``                              574   Plots for the pipeline example
``ZHAireSRawRoot/ZHAireSInputGenerator.py``                  492   Generates ZHAireS inputs
``Common/EventParametersGenerator.py``                       353   Event parameter files
``CoREASRawRoot/CorsikaInfoFuncs.py``                        455   Reads CORSIKA output
``ZHAireSRawRoot/ZHAireSCompressEvent.py``                   219   Compresses an event
``Common/RunSimPipe*.py``                                    287   Three pipeline examples
=========================================================  ======  ================================

``Common/raw_root_trees.py`` defines the RawRoot format, a second schema
parallel to ``grand/dataio``.  A change to one does not reach the other.

State of the code
-----------------

**It is outside the quality gates.**  The CI lint job checks ``grand/``,
``tests/``, ``quality/``, ``notebooks/`` and ``docs/dev/``.  It does not check
``sim2root/``.  The test suite does cover it, from outside: ``tests/sim2root/``
runs the converters on the committed samples (conversion, the trace window,
Xmax, the no-antenna case, the documented commands), but nothing tests the
modules piece by piece.

**Ruff reports 837 findings there**, against zero in the gated scope.
The largest groups are ``F405`` (372, names possibly undefined from star
imports), ``D103`` (104, missing docstrings) and ``F821`` (98, undefined
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

The converters produced the Data Challenge datasets and are tested end to
end, but they have not had the cleanup the package has had.  A change there is
checked only by the end-to-end tests.

A stale twin: ``src_outlib/``
------------------------------

``src_outlib/`` holds an abandoned copy of part of this tooling; check which
copy you are editing before changing anything named ``AiresInfo*``.

``src_outlib/AiresInfoFunctionsGRANDROOT.py`` is a diverged copy of the file of
the same name under ``ZHAireSRawRoot/`` — 1814 lines against 2095, missing a
series of ``Get*FromSry`` functions the live one has.  And
``src_outlib/ZHAireSRawToGRANDROOT.py`` has not been valid Python since 30 June
2023, when a merge conflict was committed unresolved and never cleaned up.

**``sim2root/ZHAireSRawRoot/`` is the live copy.**  Nothing imports
``src_outlib/``, so a change there has no effect.

It is not deleted yet because four branches still touch it; see
:ref:`issue-src-outlib-conflict`.

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
- Adding ``sim2root/`` to the lint gate is planned.  The undefined-name block
  comes first, since it is the only group of findings that is an error rather
  than a matter of style.
