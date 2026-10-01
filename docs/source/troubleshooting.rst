Troubleshooting
===============

.. contents::
   :local:
   :depth: 1

Things that go wrong, in rough order of how often they catch someone new.

Most of these are not exceptions.  GRANDlib's characteristic failure is a
``nan`` or a silently wrong number that stays plausible for several steps, so
the entries below are grouped by what you *see*, not by what raised.

Errors and warnings that start with ``GRANDlib:``
-------------------------------------------------

Since 2026-10 the public functions check their input, and say what is wrong
in a message that starts with ``GRANDlib:``, names the function and the
argument, gives what was expected and shows what was given::

    ValueError: GRANDlib: Geodetic: 'latitude' must be between -90 and 90 degrees, got 200.0
    TypeError: GRANDlib: TRun.run_number: must be an integer, got 1.7
    FileNotFoundError: GRANDlib: EventList: no such file or directory: /data/run_1

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
instead, and are used as given.  A tree field outside its physical range is
one: reading a file goes through the same code as writing one, and existing
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

**An elevation lookup returned ``nan``.**  There is no SRTM tile for that
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

**A geoid undulation returned ``nan`` and the coordinates look fine.**  The
site is west of Greenwich and you are on a version before the fix for
:ref:`issue-geoid-longitude-convention`: the keyword form did not wrap a
negative longitude.  Upgrade, or pass a ``Geodetic``.

The numbers are wrong but nothing failed
----------------------------------------

**The three trace channels do not match the field components you put in.**
They are not meant to.  ``trace[:, 2]`` is the Z antenna arm, not
:math:`E_z`: the response is the projection of the field onto the effective
length in the spherical basis of the *arrival direction*, so which arm sees
what depends on the geometry.  A field with components 1.0 : 0.6 : 0.2 can come
out as 600 : 400 : 1.  See :doc:`simulation` and notebook 06.

**Changing ``vga_gain`` changes nothing.**  It is ignored.  See
:ref:`issue-vga-gain-ignored`.

**Two noise levels disagree by a factor of two.**  Compare the ``du_type``.
The three values resolve to two distinct sets of numbers whose levels differ by
up to 2.1×, and two of the three read a byte-identical file.  See
:ref:`issue-galactic-noise-tables`.

**A frequency is out by** :math:`10^6`.  :class:`~grand.sim.detector.antenna_model.AntennaModel`
stores its frequency axis in **hertz**; everything in
:mod:`grand.sim.detector.rf_chain` uses **megahertz**, and the attribute name
carries no unit suffix.  Divide by ``1e6`` when crossing between them.

**An angle is out by a factor of 57.3.**  Every angle in GRANDlib is in
**degrees**, never radians — including ``zenith``, ``azimuth``, and the
``phi``/``theta`` axes of the antenna tables.

**``leff_theta`` is ``None``.**  The loaded tables hold the real/imaginary form
in ``leff_theta_reim``; the polar attributes ``leff_theta``, ``leff_phi``,
``phase_theta`` and ``phase_phi`` exist and are never populated.

Exceptions
----------

``KeyError`` on the first ``compute_voltage()``
    Fixed.  Four parameters — ``resample_to_mhz``, ``extend_to_us``,
    ``calibration_smearing_sigma`` and ``add_jitter_ns`` — used to be set only
    by ``scripts/convert_efield2voltage.py``, so the command line worked while
    the documented Python usage raised.  If you still see this, you are on an
    older revision; pin it with
    ``tests/sim/test_pipeline_end_to_end.py::test_default_params_are_complete``.

``NotUniqueEvent: An event with (run_number,event_number)=(0,0) already exists``
    You wrote two events with the same key into one tree, most often by running
    a simulation twice into the same output file.  Give each run its own output
    path, or advance ``event_number``.

``OSError: ... is being written by another process`` (or ``was changed by another process``)
    Two processes tried to write the same ROOT file, for example two batch
    jobs with the same output name.  Only one process may write a file at a
    time; the other is refused rather than allowed to lose events or corrupt
    the file.  Give each job its own output file, or wait for the first to
    finish and run the second again.

``ModuleNotFoundError: No module named 'ROOT'``
    The environment is not active, or ROOT is not installed.  ``import grand``
    requires ROOT at import time, not lazily; see
    :ref:`issue-import-requires-root`.

    .. code-block:: bash

        conda activate grand-dev

``ImportError`` for ``turtle`` or ``gull``
    The C extensions are not built.  They compile from source and are not on
    PyPI:

    .. code-block:: bash

        source env/setup.sh

    That script needs ``make``, which the conda environment does not declare —
    install it from your distribution if it is missing.

``ValueError`` naming a file and the ``_L0_``/``_L1_`` convention
    A file name's analysis level does not match the ``analysis_level`` stored
    in its tree.  Rename the file, or fix the tree.  The two must agree; see
    :doc:`datamodel`.

Messages that look like errors and are not
------------------------------------------

``No valid trun TTree in the file ...  Creating a new one.``
    Expected when writing.  Constructing a tree class on a file that does not
    yet contain that tree creates it.  Only worry if you see it while *reading*
    a file you expected to be populated — that means the tree is absent or
    named differently.

``TClass::Init:0: RuntimeWarning: no dictionary for class ... is available``
    ROOT could not find a dictionary for a class it does not need.  Harmless.

A CPU-feature warning on stderr during a documentation build
    ROOT's JIT writes it once at first use.  It is on stderr, so
    ``jupyter-sphinx`` reports it as a warning and ``sphinx-build -W`` would
    make it fatal.  The documentation build therefore does not use ``-W``; the
    log is grepped instead, with that one line filtered.  See :doc:`ci`.

Reading files
-------------

**``DataDirectory`` returned fewer handles than I have runs.**  It groups by
tree type and analysis level, not by run.  Two runs in one directory do not
give two handles.  This is the trap notebook 02 works through, and it is
written down nowhere else.

**A bare attribute gave me the wrong level.**  Both levels are returned, and a
bare attribute follows the highest one present.  Ask for the level you want
explicitly.

**A reader wants a directory, not a file.**  Some readers are coupled to
directory layout rather than taking a path; see
:ref:`issue-reader-directory-coupling`.

**Memory grows with every file and the job is killed out of memory.**  Tree
instances are kept alive in ``grand_tree_list``, together with the ROOT files
they opened, until you release them.  Use ``with TADC(path) as tadc:`` or call
``tadc.stop_using()`` at the end of each iteration; see
:ref:`datamodel-releasing-trees`.

Environment and build
---------------------

**The conda solve takes forever or fails.**  Use libmamba:

.. code-block:: bash

    conda env create -f env/conda/grand-dev.yml --solver=libmamba

**A result changed after a ROOT upgrade.**  Check first that the computation is
deterministic.  A difference between two runs on different ROOT versions was
once recorded here as a ROOT effect and turned out to be an unseeded random
draw in the test itself — see :ref:`issue-root-638-numerical-difference`, which
is retained as a worked example of the mistake.  Seed everything, reproduce the
difference twice, and only then look at ROOT.

**A test fails only in a full run, never alone.**  Look for unseeded
randomness.  One such test existed in this repository and failed about one run
in six; it is now seeded through a local generator so it does not disturb
global random state.

Still stuck
-----------

:doc:`known_issues` lists what is known to be wrong, with what was measured and
what would settle it.  If the behaviour is not there and not here, it is worth
reporting — and worth writing a test that reproduces it, because most of the
entries on that page began as one.
