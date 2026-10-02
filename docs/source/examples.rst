The examples folder
===================

The repository's ``examples/`` folder holds scripts and notebooks written by
the developers over several years.  They are less maintained than the
:doc:`notebooks`, which are executed on every change, so start with the
notebooks and come here for a task they do not cover.  This page lists the
examples worth using, by task, then the ones that no longer run.

Run each script from its own folder after ``source env/setup.sh``.

By task
-------

==============================================  =======================================================
Task                                            Example
==============================================  =======================================================
Write the trees of a file by hand               ``dataio/data_storing.py``, then ``data_reading.py``
                                                to read it back (:doc:`writing_files`)
Look inside any GRANDlib file                   ``dataio/datafile_use.py FILE.root``
Read a simulation event by event                ``aoi/browse_sim2root_events_example.py``
                                                ``sim2root/Common/sim_...``: waits for Enter
                                                between events
Read measured GP13 events                       ``aoi/browse_gp13_events_example.py``: needs a GP13
                                                file (:doc:`measured_data`)
Reconstruct measured candidates                 ``analysis/main_AOI.py`` and ``main_DOI.py``, plotted
                                                by ``analysis/display.py``; see
                                                ``analysis/README.md``
The RF chain, stage by stage                    ``sim/rf_chain_example.py galactic --lst 18`` and
                                                ``read_modify_RF_chain_elements.ipynb``
Galactic noise and the antenna response         ``sim/galactic_noise.ipynb``, ``sim/antenna.ipynb``
Coordinates, the geomagnetic field, topography  the three notebooks in ``geo/``
Generate an array layout                        ``geo/grids.py`` (:doc:`sites`)
Ground elevation around a site                  ``geo/local_topography.py --download``
                                                (fetches about 50 MB of tiles)
View events in 3D                               ``eventviewer/`` after ``pip install -e ".[viewer]"``
==============================================  =======================================================

What does not run
-----------------

* ``old/``: outdated interfaces and paths of other people's machines.  Kept
  for reference only.
* ``datalib/datamanager_example.py``: needs access to the collaboration's
  database servers and an edited ``config.ini``.
* ``aoi/browse_gp13_events_example.py``: runs, but only with measured GP13
  data, which are not in the repository.
* ``dataio/readme.md`` describes an older layout of the data classes; the
  scripts beside it are current.

``examples/README.md`` has the full list with what each example needs.

Where next
----------

* :doc:`recipes` for short, tested code for common tasks.
* :doc:`notebooks` for the maintained walk-throughs.
