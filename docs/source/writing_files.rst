Writing your own files
======================

GRANDlib's tree classes write files as well as read them.  This page writes
a run and two events of voltages in the layout the rest of GRANDlib reads,
so that the result opens with :class:`~grand.dataio.DataDirectory`,
the :doc:`commands` and the :doc:`recipes`.  The code runs as written when
this page is built.

.. contents::
   :local:
   :depth: 1

The run: where the antennas are
-------------------------------

Every folder needs one run file, written first.  It holds what does not
change from event to event: the site, the origin of the array frame and the
position of each detection unit.

.. jupyter-execute::

    import os
    import tempfile

    import numpy as np
    from grand.dataio import TRun, TVoltage

    os.chdir(tempfile.mkdtemp())
    os.makedirs("my_run")

    trun = TRun()
    trun.run_number = 1
    trun.site = "Dunhuang"
    trun.origin_geoid = [40.98, 93.95, 1200.0]     # latitude, longitude (deg), height (m)
    trun.t_bin_size = [0.5] * 3                    # ns per sample, one value per unit
    for du, xyz in enumerate([[0, 0, 0], [500, 0, 0], [0, 500, 0]]):
        trun.du_id.append(du)
        trun.du_xyz.append(xyz)                    # m: x north, y west, z up
    trun.comment = "three units, 500 m apart"
    trun.fill()
    trun.write("my_run/run_1_L0_0000.root")
    trun.stop_using()

``fill()`` adds the values set so far as one entry; ``write()`` saves the
entries to the file.  Every field and its unit is in the :doc:`data_format`.

The events
----------

Each event is one entry: set its fields, then call ``fill()``.  Write once,
after the last event:

.. jupyter-execute::

    rng = np.random.default_rng(0)

    tvoltage = TVoltage()
    tvoltage.comment = "Gaussian noise, 20 µV"
    for event in range(2):
        tvoltage.run_number = 1
        tvoltage.event_number = event
        tvoltage.du_id = [0, 1, 2]
        tvoltage.du_count = 3
        tvoltage.du_seconds = [1_700_000_000] * 3        # GPS second of each trace
        tvoltage.du_nanoseconds = [0, 120, 340]
        tvoltage.trace = rng.normal(0, 20, (3, 3, 1024)).astype(np.float32)  # µV, (unit, arm, sample)
        tvoltage.fill()
    tvoltage.write("my_run/voltage_1-2_L0_0000.root")
    tvoltage.stop_using()

A second entry with the same ``run_number`` and ``event_number`` raises
``NotUniqueEvent``.  Writing a tree into a file that already holds one of the
same kind raises ``FileExistsError``: to add events, open the file with
``TVoltage("my_run/voltage_1-2_L0_0000.root")``, fill and write again, or
pass ``overwrite=True`` to replace it.

Reading it back
---------------

.. jupyter-execute::

    from grand.dataio import DataDirectory

    folder = DataDirectory("my_run")
    folder.tvoltage.get_event(1, 1)                 # event 1 of run 1
    print(np.asarray(folder.tvoltage.trace).shape, folder.tvoltage.du_nanoseconds)
    print(folder.tvoltage.comment)
    print(folder.tvoltage.modification_software, folder.tvoltage.modification_software_version)

GRANDlib fills the provenance itself: the software, its version and git
commit, the creation date and the analysis level.  Set ``comment`` to say
what the file holds; set ``source_datetime`` and ``analysis_level`` when they
are not the defaults (the Unix epoch and 0).

File names
----------

The readers find trees by file name, so follow the pattern above:

* the tree type first: ``run_``, ``efield_``, ``voltage_``, ``adc_``,
  ``rawvoltage_``, ``shower_``;
* the analysis level and a serial number last: ``_L0_0000.root``;
* one run file per folder and one file per tree type and analysis level.

A file that does not follow it is skipped by :class:`~grand.dataio.DataDirectory`
with a warning that names it.

Closing files
-------------

``write()`` closes the file it opened.  ``stop_using()`` releases a tree you
are done with; a tree filled and never written is lost when it is released.
Opening a tree in a ``with`` block, as when reading, closes it at the end of
the block.  In a notebook, close what you write before reading it from
another process.

Where next
----------

* :doc:`data_format` for every tree and field.
* ``examples/dataio/data_storing.py`` fills every tree type, including
  ``TADC`` and ``TShower`` (:doc:`examples`).
