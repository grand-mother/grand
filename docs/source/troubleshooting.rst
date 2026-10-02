Troubleshooting
===============

Problems grouped by what you see: an error message, a ``nan``, a number that
looks wrong, or a message that only looks like an error.  For known defects,
see :doc:`known_issues`.

.. contents::
   :local:
   :depth: 1

Errors and warnings that start with ``GRANDlib:``
-------------------------------------------------

Public functions check their input.  A message that starts with
``GRANDlib:`` names the function and the argument, what was expected and what
was given::

    ValueError: GRANDlib: Geodetic: 'latitude' must be between -90 and 90 degrees, got 200.0
    TypeError: GRANDlib: TRun.run_number: must be an integer, got 1.7
    FileNotFoundError: GRANDlib: EventList: no such file or directory: /data/run_1

:ref:`troubleshooting-messages` below lists the common ones, with what to do.

The exceptions are the standard ones, so ``except ValueError`` and the like
work as usual:

- ``TypeError``: the wrong kind of value (a string where a number is
  expected, a fraction for an integer field);
- ``ValueError``: the right kind but out of range, or the wrong shape or
  length (antenna positions that are not (N, 3), fewer antennas than a fit
  needs);
- ``FileNotFoundError``, ``OSError``: a missing file or directory, or a file
  that is not a ROOT file.

Values that are suspicious but usable give a :class:`~grand.basis.validate.GRANDlibWarning`
instead and are used as given.  A tree field outside its physical range is
one: reading a file goes through the same code as writing one and existing
files hold placeholders such as ``xmax_grams = -201``, which must stay
readable.  ``NaN`` in a coordinate is another.  To find where they come from,
turn them into errors::

    import warnings
    from grand.basis.validate import GRANDlibWarning
    warnings.simplefilter("error", GRANDlibWarning)

Two things are deliberately *not* errors.  Opening a tree on a file that does
not exist creates it, because that is how trees are written, so a mistyped
file name gives an empty tree; only a directory that does not exist is
refused.  And an empty directory is a valid :class:`~grand.dataio.DataDirectory`
for the same reason; :class:`~grand.aoi.event_list.EventList`, which only
reads, refuses it.

Nothing raised, but the answer is ``nan``
-----------------------------------------

**An elevation lookup returned ``nan``.**  There is no :term:`SRTM` tile for that
one-degree square.  Tiles are not in version control and a fresh checkout has
none:

.. code-block:: python

    from grand.geo import topography
    from grand.geo.coordinates import Geodetic

    site = Geodetic(latitude=40.98, longitude=93.95, height=0.0)
    topography.update_data(site, radius=50e3)

The ``nan`` propagates through ``elevation(..., reference='sea')``, which
subtracts the undulation from it.  **Check for ``nan`` after every elevation
lookup.**

The numbers are wrong but nothing failed
----------------------------------------

**The three arms do not match the field components you put in.**
They are not meant to.  ``trace[:, 2]`` is the Z antenna arm, not
:math:`E_z`: the response is the projection of the field onto the effective
length in the spherical basis of the *arrival direction*, so which arm sees
what depends on the geometry.  A field with components 1.0 : 0.6 : 0.2 can come
out as 600 : 400 : 1.  See :doc:`simulation` and notebook 06.

**Changing ``vga_gain`` changes nothing.**  It is ignored.  See
:ref:`issue-vga-gain-ignored`.

**Two noise levels disagree by a factor of two.**  If either was simulated
before 7 September 2026, see :ref:`issue-galactic-noise-tables`.  Since then
the three antenna models agree to about 10%.

**A frequency is out by** :math:`10^6`.  :class:`~grand.sim.detector.antenna_model.AntennaModel`
stores its frequency axis in **hertz**; everything in
:mod:`grand.sim.detector.rf_chain` uses **megahertz**.  The attribute name
carries no unit suffix.  Divide by ``1e6`` when crossing between them.

**An angle is out by a factor of 57.3.**  The trees, the simulation chain and
the antenna tables take angles in **degrees**; :mod:`grand.analysis` and the
event viewer work in **radians**.  The analysis functions warn when given a
value larger than :math:`2\pi`.

**``leff_theta`` is ``None``.**  The loaded tables hold the real/imaginary form
in ``leff_theta_reim``; the polar attributes ``leff_theta``, ``leff_phi``,
``phase_theta`` and ``phase_phi`` exist and are never populated.

.. _troubleshooting-messages:

Error and warning messages
--------------------------

The messages GRANDlib gives most often, as they appear.  Search this page for
a few words of the message you got; ``…`` stands for the part that changes.

Files and folders
~~~~~~~~~~~~~~~~~

``no such file or directory: …``
    The path does not exist.  A relative path is relative to the directory
    Python runs in; in a notebook, that is ``notebooks/``.

``no ROOT files (*.root) in …``
    The folder exists but holds no ROOT file: check that it is the folder
    ``sim2root.py`` wrote, not its parent.

``no GRAND event files recognized in …; files must be named <type>_<events>_L<level>_<serial>.root``
    The folder holds ROOT files whose names do not follow GRAND's convention,
    such as ``efield_1-2_L0_0000.root``, so their trees cannot be found.
    Rename the files, or open each one with its tree class
    (:class:`~grand.dataio.event_trees.TEfield`, ...).

``… holds no efield file`` / ``… holds no run file`` / ``… holds no shower file``
    :class:`~grand.sim.efield2voltage.Efield2Voltage` needs the whole folder
    ``sim2root.py`` wrote, with its ``efield_*``, ``run_*`` and ``shower_*``
    files; a single e-field file is not enough.

``… holds efield files at levels 0, 1: reading the highest`` (warning)
    The folder holds several :term:`analysis levels <analysis level>`.  Choose
    one with ``efield_level=0`` in Python or ``--level 0`` on the command line.

``… holds an efield file at level … but no run file at that level``
    The run file carries the sampling interval, so it must exist at the level
    of the e-field file.  Convert the simulation again, or choose another
    level.

``cannot tell which run and shower trees belong to …: a GRAND filename carries its analysis level as '_L0_' or '_L1_'``
    The file readers of :mod:`grand.dataio.root_files` find a file's run and
    shower trees by its name, which must carry ``_L0_`` or ``_L1_``.  Rename
    the file, or read its trees directly (:ref:`issue-reader-directory-coupling`).

``DataDirectory: … is named level … but its … tree is level …`` (warning)
    The ``_L0_`` or ``_L1_`` in a file name disagrees with the
    ``analysis_level`` stored in its trees; the trees' level is used.  Rename
    the file to match.

``convert_voltage2adc: no voltage_*_L<level>_*.root in …``
    Run ``convert_efield2voltage.py`` on the folder first (:doc:`commands`).

Writing files
~~~~~~~~~~~~~

``… is being written by another process`` / ``… was changed by another process``
    Two processes tried to write the same ROOT file, for example two batch
    jobs with the same output name.  Only one process may write a file at a
    time; the other is refused rather than allowed to lose events or corrupt
    the file.  Give each job its own output file, or wait for the first to
    finish and run the second again.

``NotUniqueEvent: An event with (run_number,event_number)=(…) already exists``
    Two events with the same run and event numbers were written into one
    tree, most often by running a simulation twice into the same output file.
    Give each run its own output file, or change ``event_number``.

Reading events
~~~~~~~~~~~~~~

``no event … in run … in the input; it holds (event, run) …`` / ``no event with event number … and run number …``
    The numbers are wrong, or swapped: GRANDlib takes the event number first.
    ``tree.get_list_of_events()`` lists the (event, run) pairs a file holds.

Values
~~~~~~

``'lst' must be 0 to 24 h`` / ``'f_lst' (local sidereal time) must satisfy 0 <= f_lst < 24 hours``
    The local sidereal time is in hours, not degrees.

``'…' = … is outside -6.28319 to 6.28319 rad; is it an angle in degrees?`` (warning)
    :mod:`grand.analysis` works in radians; convert with ``np.radians``.
    The trees and the simulation use degrees.

``the peak times span … s, more than a shower front takes to cross any array; are they in ns rather than s?`` (warning)
    The reconstruction takes times in seconds; divide nanoseconds by ``1e9``.

``Source direction at zenith … deg, outside the antenna table (…; below the antenna's horizon): its effective length is taken as zero`` (warning)
    The shower maximum is below the antenna's horizon, so the antenna is
    taken to see nothing from it.  This happens for nearly horizontal
    showers over uneven ground, or when Xmax is wrong in the input.

Installation
~~~~~~~~~~~~

``data model incomplete: … is missing`` / ``data model damaged: … has … bytes``
    A file of the model data is missing or was not downloaded completely.
    Run ``python data/download_data_grand.py`` (``source env/setup.sh`` does
    it).

``ModuleNotFoundError: No module named 'ROOT'``
    The environment is not active, or ROOT is not installed.  ``import grand``
    works without ROOT, but reading files, topography and the simulation need
    it (:ref:`issue-import-requires-root`).

    .. code-block:: bash

        conda activate grand-dev

``ImportError`` for ``turtle`` or ``gull``
    The C extensions are not built.  They compile from source:

    .. code-block:: bash

        source env/setup.sh

Messages that look like errors and are not
------------------------------------------

``No valid trun TTree in the file ...  Creating a new one.``
    Expected when writing: constructing a tree class on a file that does not
    yet contain that tree creates it.  When *reading* a file you expected to
    hold the tree, it means the tree is absent or named differently.

``TClass::Init:0: RuntimeWarning: no dictionary for class ... is available``
    ROOT could not find a dictionary for a class it does not need.  Harmless.

A CPU-feature warning during a documentation build
    ROOT's JIT compiler writes it on some processors.  It is harmless; see
    :doc:`ci`.

Reading files
-------------

**``DataDirectory`` returned fewer handles than there are runs.**  It groups
by tree type and analysis level, not by run, so two runs in one folder do not
give two handles.  Notebook 02 works through this.

**A bare attribute gave the wrong level.**  A bare attribute such as
``tefield`` follows the highest level present.  Ask for the level explicitly,
as ``tefield_l0``.

**A reader wants a directory, not a file.**  Some readers are coupled to
directory layout rather than taking a path; see
:ref:`issue-reader-directory-coupling`.

**Memory grows with every file.**  Release each tree when done with it: use
``with TADC(path) as tadc:`` or call ``tadc.stop_using()`` at the end of each
iteration (:ref:`datamodel-releasing-trees`).

Environment and build
---------------------

**The conda solve takes forever or fails.**  Use libmamba:

.. code-block:: bash

    conda env create -f env/conda/grand-dev.yml --solver=libmamba

**A result changed after a ROOT upgrade.**  Check first that the computation
is deterministic: seed every random draw and reproduce the difference twice
before attributing it to ROOT.

**A test fails only in a full run, never alone.**  Look for an unseeded random
draw, or one that uses NumPy's global generator.

Still stuck
-----------

Check :doc:`known_issues` and the `open issues on GitHub
<https://github.com/grand-mother/grand/issues>`_.  If the problem is not
there, please open an issue with the code that reproduces it.
