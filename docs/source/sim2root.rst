Converting simulations: sim2root
================================

GRANDlib starts from an electric field.  The air shower and its radio
emission are simulated by :term:`ZHAireS` or CoREAS, outside GRANDlib.  The converters
in ``sim2root/`` turn the output of those codes into GRAND's ROOT format, which
the rest of the library reads.

.. contents::
   :local:
   :depth: 1

``sim2root/`` is in this repository but is not part of the ``grand`` package:
it is not imported by it.  Its code is not linted.  The tests run the
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

Changing the converters
-----------------------

``sim2root/README.md`` documents the converters in detail.  Their code is not
yet linted and the tests check them end to end only, by converting the
committed samples (``tests/sim2root/``).  After a change, convert a sample and
read the result back with :mod:`grand.dataio`.  ``Common/raw_root_trees.py``
defines the RawRoot format separately from ``grand/dataio``, so a field added
to one must be added to the other.  Edit the files under ``sim2root/``, not
the stale copy in ``src_outlib/`` (:ref:`issue-src-outlib-conflict`).

Where next
----------

* :doc:`commands` for the next steps: voltage, then ADC counts.
* :doc:`datamodel` for the trees the converters write.
