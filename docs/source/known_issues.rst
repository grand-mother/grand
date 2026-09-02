Known issues
============

.. contents::
   :local:
   :depth: 1

Open problems that affect results or block work, with what is measured, what
is not, and what would settle each one.  Fixed issues move to the changelog
rather than staying here.

.. _issue-galactic-noise-normalisation:

Galactic-noise normalisation: which constant is right depends on one definition
--------------------------------------------------------------------------------

:Status: open, blocking — but the question is now sharp
:Affects: every simulated voltage and everything downstream — trigger
          studies, sensitivity estimates, Data Challenge outputs
:Blocks: ``dev_snonis`` (PR 153), ``refact_galaxy`` (PR 146)
:Test: ``tests/sim/test_galactic_noise_normalisation.py``

:func:`grand.sim.noise.galaxy.galactic_noise` constructs the spectrum as

.. code-block:: python

    amp   = rng.normal(loc=0, scale=v_amplitude[np.newaxis, ...],
                       size=(nb_ant, 3, nb_freq))
    phase = 2 * np.pi * rng.random(size=(nb_ant, 3, nb_freq))
    v_complex = np.abs(amp * size_out / 2) * np.exp(1j * phase)

where ``v_amplitude`` is the tabulated voltage spectral density, converted to
a per-bin amplitude by multiplication with :math:`\sqrt{\Delta f}`.  The
branch ``dev_snonis`` changes the constant from ``size_out / 2`` to
``size_out / sqrt(2)``, which scales every simulated noise voltage by about
1.41.

**Measurement.**  Made on 2 September 2026, comparing each ``du_type``
against the table that ``du_type`` actually reads, at LST 18 h, with 221 bins
of 1 MHz placed in a transform of length ``size_out``:

=================  =====================  ==============================
``du_type``        simulated / tabulated  compare :math:`1/\sqrt2=0.7071`
=================  =====================  ==============================
``GP300``          0.7050                 −0.3 %
``GP300_nec``      0.7049                 −0.3 %
``GP300_mat``      0.7049                 −0.3 %
=================  =====================  ==============================

The ratio is :math:`1/\sqrt2` to a fraction of a percent, and it does not
move with ``size_out`` (1024, 2048, 4096 all give the same value) or with the
number of antennas.  The implementation produces an RMS that is exactly
:math:`1/\sqrt2` of the tabulated value — which is the relation between the
RMS and the peak of a sinusoid.

**So the decision reduces to one definitional question**, and each answer
picks a different constant:

=========================================================  ==============================
If ``Vocmax_30-250MHz_uVperMHz`` is a **maximum**           the current ``size_out/2`` is right
If it is an **RMS** spectral density                        PR 153's ``size_out/sqrt(2)`` is right
=========================================================  ==============================

The filename says *max*.  Nothing else in the repository states which is
meant, and the question belongs to the authors of the table.

.. note::

   **Correction to an earlier write-up.**  A measurement made on 30 August
   2026 reported a ratio of 0.33, concluded that "neither candidate is 1",
   and inferred an unexplained factor of roughly 2.  That comparison ran the
   default ``GP300`` simulation against ``Vocmax_..._hfss.npy`` — a table
   that ``GP300`` never reads.  Those two are the same antenna model at two
   different normalisations (see below), so the missing factor was the
   normalisation difference between the tables, not a defect in the
   transform.  Compared like with like, there is no unexplained factor.

**What is not in doubt.**  The implementation is internally consistent: the
time series obtained from the returned spectrum carries that spectrum's
energy, to one part in :math:`10^9`.  Whatever convention is chosen, the
transform itself is right.

**A second, separate discrepancy.**  Section 8.2 of `arXiv:2408.10926
<https://arxiv.org/abs/2408.10926>`_ states that the module randomises the
*phase* of the sky-averaged noise.  The code also randomises the modulus,
drawing it from a normal distribution and taking the absolute value.  The two
agree in mean power and differ in their fluctuations, so this is not
necessarily a defect — but the published description does not match either
candidate implementation, and one of the two should be corrected.

**Why it matters.**  A 1.41 error in noise amplitude is not cosmetic in an
experiment whose sensitivity is set by a trigger threshold.  Until this is
resolved, results that depend on the noise level should record which version
of :mod:`grand.sim.noise.galaxy` produced them; ``TRun.software_version``
exists for exactly that.

.. _issue-galactic-noise-tables:

The shipped Galactic-noise tables do not say what the docstring says
--------------------------------------------------------------------

:Status: open, not blocking
:Affects: any study that selects a ``du_type``
:Test: ``tests/sim/test_galactic_noise_normalisation.py``

Three separate problems with the data files under ``data/noise/``, all found
while settling the normalisation question above.

**The NEC and MATLAB tables are the same file.**
:func:`~grand.sim.noise.galaxy.galactic_noise` documents ``GP300_nec`` and
``GP300_mat`` as the NEC and MATLAB variants of the antenna response.  All
three pairs of shipped tables are byte-identical:

=================================  ================================
Table                              ``nec`` and ``mat`` SHA-256
=================================  ================================
``Vocmax_30-250MHz_uVperMHz``      identical
``Pocmax_30-250_Watt_per_MHz``     identical
``Voutmax_30-250MHz_uVperMHz``     identical
=================================  ================================

So the two options select the same numbers, and one of the files was
presumably copied over the other.  Which one is the survivor is not
recoverable from the repository.

**The default is not the model the docstring names.**  ``du_type='GP300'``
does not read any ``Vocmax_*.npy`` table: it recomputes the voltage from
``PG_ALL_jifen.mat`` as :math:`V_{\rm oc}^2 = 4 P R_{\rm ant}`.  The result
is a nearly constant 0.463 times the ``hfss`` table — the ratio is flat
across the band to 2.4 %, which is what a normalisation difference looks
like, whereas the ratio to the ``nec`` table varies by 11 %, which is what a
genuinely different antenna model looks like.  ``GP300`` and ``hfss`` are
therefore one model at two normalisations differing by a factor of about
2.16.

**The highest-level tables are unreachable.**  The three ``*_hfss.npy`` files
ship but no ``du_type`` opens them.  A study that wanted them would have to
load them by hand.

**Consequence.**  The four nominal antenna models resolve to two distinct sets
of numbers, differing in band-integrated level by up to a factor of 2.1:

=====================  =========================  =====================
Selector               Reads                      Band RMS at LST 18 h
=====================  =========================  =====================
``GP300`` (default)    ``PG_ALL_jifen.mat``       27.8, 35.6, 31.4 µV
``GP300_nec``          ``Vocmax_..._nec.npy``     44.5, 55.6, 53.4 µV
``GP300_mat``          the same file as ``nec``   44.5, 55.6, 53.4 µV
*(unreachable)*        ``Vocmax_..._hfss.npy``    59.6, 75.5, 66.9 µV
=====================  =========================  =====================

A noise level quoted without its ``du_type`` is ambiguous by a factor of two.
This is independent of the :math:`\sqrt2` question above: that one is a single
constant applied to all models alike.

.. _issue-handbook-arm-naming:

The Handbook has the X/Y antenna-arm mapping backwards
--------------------------------------------------------

:Status: erratum in ``resources/GRANDlib_Handbook.pdf``; the code is correct
:Affects: anyone reading the Handbook to interpret a trace
:Test: ``tests/sim/test_antenna_arm_identity.py``

The GRANDlib Handbook, in its description of the antenna models, says the
effective-length files hold the

    "East--West arm, denoted as EW or X arm, South--North denoted as SN or Y
    arm and Vertical denoted as Zarm"

The code assigns them the other way round.
:class:`~grand.sim.detector.antenna_model.AntennaModel` reads
``Light_GP300Antenna_nec_Xarm_leff.npz`` into ``leff_sn`` and
``..._nec_Yarm_leff.npz`` into ``leff_ew``.

**The code is right.**  The HFSS files are named after the physical arm
(``EWarm``, ``SNarm``) and carry no X/Y ambiguity, so the NEC and MATLAB files
can be identified by correlating their patterns against them over 60-200 MHz:

===========  =================  =================  ==============
File         vs ``hfss_EW``     vs ``hfss_SN``     identified as
===========  =================  =================  ==============
``nec_X``    −0.176             **+0.722**         south-north
``nec_Y``    **+0.651**         −0.170             east-west
``mat_X``    −0.128             **+0.716**         south-north
``mat_Y``    **+0.618**         −0.143             east-west
===========  =================  =================  ==============

The two reference arms correlate with each other at +0.003, so they are
independent and the identification is not an artefact of the method.

So **X is the south-north arm and Y is the east-west arm**, which is also what
:class:`~grand.geo.coordinates.GRANDCS` implies: its ``x`` axis points north.

**Why it matters.**  Following the Handbook would swap two of the three
channels for every event simulated with ``du_type='GP300_nec'`` or
``'GP300_mat'``.  A channel swap does not look like a bug — it looks like a
polarisation measurement.

The output channel order is fixed by :mod:`grand.sim.efield2voltage`, which
fills ``voc[:, 0]`` from the SN arm, ``voc[:, 1]`` from EW and ``voc[:, 2]``
from Z.  With the identification above, that is X, Y, Z in that order, which is
how the documentation and the notebooks label them.

.. _issue-geoid-longitude-convention:

``geoid_undulation`` returns NaN for a negative longitude
----------------------------------------------------------

:Status: open, not blocking
:Affects: any site west of Greenwich
:Test: ``tests/geo/test_topography_conventions.py``

The EGM96 undulation map shipped as ``data/egm96.png`` is indexed over
longitude 0-360 degrees.  :func:`grand.geo.topography.geoid_undulation` has two
calling conventions, and only one of them normalises:

.. code-block:: python

    >>> topography.geoid_undulation(latitude=-35.20, longitude=-69.32)
    nan
    >>> topography.geoid_undulation(latitude=-35.20, longitude=290.68)
    -44.371455192615110
    >>> topography.geoid_undulation(
    ...     Geodetic(latitude=-35.20, longitude=-69.32, height=0.0))
    -44.371455192615110

Nothing raises.  The ``nan`` propagates into whatever geometry follows, and
through ``elevation(..., reference='sea')``, which subtracts the undulation.

**Workaround.**  Pass a :class:`~grand.geo.coordinates.Geodetic` rather than
the ``latitude=``/``longitude=`` keywords.  That path normalises and is what
the documentation and notebook 07 use throughout.

**Fix.**  Wrap the longitude into :math:`[0, 360)` in the keyword branch.  It
is one line; it has not been made here only because changing a function's
domain silently is worse than documenting it, and no user of the keyword form
in the western hemisphere has been identified.

This is a specific case of a wider pattern in :mod:`grand.geo.topography`: a
point with no SRTM tile also returns ``nan`` rather than raising.  Both are
pinned in ``tests/geo/test_topography_conventions.py``.

.. _issue-docker-unmaintained:

The Docker installation route is unmaintained
----------------------------------------------

:Status: open — needs a decision, not a patch
:Affects: anyone following the Handbook's installation chapter
:Verified: from the files, the build scripts and Docker Hub.  **Not** by
           building an image: no container runtime was available.

The Handbook presents Docker as the first installation route and names
published images.  Those images exist and still pull — and they are the newest
there are:

===============  ========  ==============  ==============
Tag              Size      Last updated    Architecture
===============  ========  ==============  ==============
``1.2``          944 MB    2023-01-14      amd64
``2.0``          812 MB    2022-11-18      arm64
``1.1``          944 MB    2022-05-13      amd64
``1.0``          940 MB    2022-05-12      amd64
===============  ========  ==============  ==============

Nothing about the route is *broken* in the sense of refusing to run.  The
Handbook's steps end at ``source env/setup.sh``, which sets ``PYTHONPATH`` and
never invokes pip, so the ``requires-python = ">=3.10"`` in ``pyproject.toml``
is not reached — and the package uses no syntax or standard-library module
newer than 3.8, so the code itself would import.  What is wrong is that the
environment has drifted three years from the one everybody else uses.

*ROOT is ten minor versions behind.*  The images and
``env/docker_*/base.dockerfile`` pin ROOT 6.26.02, against 6.36.04 in the conda
environment and a CI matrix of 6.36 and 6.38.  This repository already records
a case where :ref:`ROOT 6.38 changes a numerical result
<issue-root-638-numerical-difference>`; someone three releases further back is
not running the same software as anybody else, and nothing would tell them.

*The base image is out of support.*  ``rootproject/root:6.26.02-ubuntu20.04``
— Ubuntu 20.04 left standard support in April 2025.  The tag does still exist
upstream, so a build would still start.

*The requirements are unpinned.*  ``env/docker_*/requirements.txt`` lists 53
packages with no version constraint at all, so a rebuild resolves to whatever
pip offers that day.  That is the opposite of what a container is for, and it
is how this set drifted from the conda environment — it was the only one
carrying ``numba`` and ``lmfit`` until those were folded in.

**Why it drifted, from the repository's own run history.**  There *was* a CI
job — ``tests_with_docker.yml`` — and it has not executed since at least
2025-08-07.  All 41 runs in that window ended ``cancelled`` after ``24h0m2s``:

.. code-block:: text

    The job has exceeded the maximum execution time while awaiting a runner

It pinned ``runs-on: ubuntu-20.04``, a runner image GitHub has since retired,
so the job queued forever for a label that no longer exists.  **It never
failed; it stopped being scheduled.**  That is the whole explanation, and it
matters for how a failure in the new workflow should be read: there is no
evidence Docker broke for any reason to do with Docker, only that nothing has
checked in over a year.

Two consequences.  34 of the 36 branches still carry that file, so every push
to them still burns a 24-hour queue slot and shows a red cross — retiring it
repo-wide belongs with the Phase 10 cleanup.  And the last commits touching
``env/docker_amd64/`` and ``env/docker_arm64/`` are 2023-09-13 and 2023-02-13,
with no file outside those directories referencing either.

**Four images exist, and they are not interchangeable.**

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Image
     - Date
     - What it is
   * - ``grandlib/dev:1.2``
     - 2023-01-14
     - what the Handbook tells users to pull
   * - ``grandlib/dev:2.0``
     - 2022-11-18
     - the arm64 build
   * - ``jcolley/grandlib_ci:0.1``
     - 2022-01-31
     - what the old CI used
   * - ``grand_docker_handson_2025``
     - 2025
     - newest that exists, but distributed as a tarball on Google Drive rather
       than a registry, so CI cannot pull it

That last one is the interesting one: the Handbook's hands-on chapter
distributes a **2025** image, two years newer than anything on Docker Hub.  If
a working modern image already exists, publishing it to a registry would be a
far smaller job than rebuilding the Dockerfiles.

.. note::

   **A regression introduced on this branch, and fixed.**  ``build_dev.sh``
   assembles its two requirements files by copying them out of the repository
   before building.  One of its sources was
   ``docs/apidoc-only/doxygen-rtd/requirements.txt``, which this branch deleted
   when the Sphinx tree was rebuilt — so ``build_dev.sh`` would have failed at
   the ``cp`` on ``dev-next`` while working on the 34 other branches that still
   have that file.  Both architectures now copy ``docs/requirements.txt``
   instead.

   The wider point stands: nothing would have caught this.  A build script with
   no CI behind it is a script that is already broken and has not been told.

**What this needs is a decision, not a patch.**  Is Docker a supported
installation route?

*If yes*, the work is: repin the base to
``rootproject/root:6.36.00-ubuntu25.04`` to match the conda environment, pin
the requirements from that environment rather than listing names, rebuild and
publish the images, and — the part that prevents a repeat — add a CI job that
builds them.

*If no*, say so on the installation page and retire ``env/docker_*`` with the
bulk cleanup in Phase 10.

Deleting it is not proposed here.  ``tian-conda-arm`` carries ARM install
notes, so that path has a user, and containers are the right answer for
reproducibility if anyone maintains them.  Merge exposure is low either way:
``env/docker_arm64`` is touched by no branch and ``env/docker_amd64`` by two.

**How to find out.**  ``.github/workflows/docker.yml`` runs the question:
trigger it from the Actions tab and it pulls the published image, checks out
``dev-next`` and ``dev`` inside it, and runs ``env/setup.sh``, ``import
grand``, the ``dataio`` suite and the full suite as separate stages.  It is
manual and is not a gate — it is expected to be able to fail.  Running it
against both branches separates "Docker is stale" from "``dev-next`` broke
Docker".

Nobody has run it yet, so the question in the title of this entry is still
open.

**In the meantime**, the supported route is the conda environment; see
:doc:`installation`.

.. _issue-src-outlib-conflict:

``src_outlib/`` carries an unresolved merge conflict from 2023
---------------------------------------------------------------

:Status: open, not blocking — the directory is unreachable
:Affects: anyone who edits ``src_outlib/``, and anyone merging the four
          branches that still touch it
:Test: ``tests/sim2root/test_converter_defects.py``

``src_outlib/ZHAireSRawToGRANDROOT.py`` contains committed conflict markers and
has not been valid Python since the day they landed:

.. code-block:: text

    $ python -c "import ast; ast.parse(open('src_outlib/ZHAireSRawToGRANDROOT.py').read())"
    SyntaxError: invalid syntax
      <<<<<<<< HEAD:examples/dataio/ZHAireSRawToGRANDROOT.py

.. code-block:: text

    358967a  2023-06-30  lwpiotr  Merging master into this branch

Both sides of the conflict import modules that no longer exist —
``grand.io.root.run`` and ``grand.dataio.root_trees`` — so neither branch of it
would run even once the markers were removed.

**Why it went unnoticed.**  Nothing imports the directory from outside itself,
it is not packaged (``pyproject.toml`` includes only ``grand*``), and the
linter does not cover it.  There is no path by which the syntax error could
reach anybody.

**It also duplicates a live file.**
``src_outlib/AiresInfoFunctionsGRANDROOT.py`` is a diverged copy of
``sim2root/ZHAireSRawRoot/AiresInfoFunctionsGRANDROOT.py`` — 1814 lines against
2095, with the sim2root copy carrying a series of ``Get*FromSry`` functions
this one lacks.  Two versions of the same ZHAireS reader, one of them stale.
Editing the wrong one is an easy mistake and would fail silently, since nothing
imports this copy.

**Not deleted here, deliberately.**  Four branches still touch
``src_outlib/``.  Removing a directory that unmerged work modifies converts a
clean merge into a delete/modify conflict on each of them, which is a poor
trade for tidying code that cannot execute.  It belongs with the bulk cleanup
in Phase 10, once the merge queue has drained — the same reasoning that defers
branch deletion.

**What to do in the meantime.**  Treat ``sim2root/ZHAireSRawRoot/`` as the live
copy and ``src_outlib/`` as abandoned.  If you find yourself editing the
latter, you are almost certainly in the wrong file.

.. _issue-nutrig-field-names:

Two names for the NUTRIG correlation fields
--------------------------------------------

:Status: open, blocking
:Blocks: ``dev_fix_root_warnings_lwp_new_fields`` and, behind it,
         ``dev_fix_root_warnings_aoi_levels_lwp``
:Test: ``tests/dataio/test_schema_snapshot.py``

Two branches add the same NUTRIG correlation quantity to
:class:`~grand.dataio.event_trees.TADC`, by the same author, with the same
type, under different names:

===========================================  ==================================
Branch                                       Fields
===========================================  ==================================
``dev_nutrig_fields`` *(merged)*             ``nutrig_rhox``, ``nutrig_rhoy``
``dev_fix_root_warnings_lwp_new_fields``     ``correlation_x``, ``correlation_y``
===========================================  ==================================

Field names enter the ROOT schema and become part of the data contract, so
only one may exist.  The choice belongs to the author and to whoever writes
the NUTRIG analysis code that reads them.

``tests/dataio/test_schema_snapshot.py`` fails if both spellings ever appear
together, so the collision cannot be merged past silently.

.. _issue-numpy2-descriptors:

Tree classes cannot be constructed under NumPy 2
--------------------------------------------------

:Status: **fixed** 2026-08-30 — kept here until it appears in a release changelog
:Affects: the whole data layer — ``TRun()`` raises
:Test: visible in ``tests/dataio/`` (about 150 tests)

**Symptom.**  Constructing any run or event tree fails::

    >>> from grand.dataio.run_trees import TRun
    >>> TRun()
    ValueError: setting an array element with a sequence.

**Cause.**  In :mod:`grand.dataio.descriptors`, ``TTreeScalarDesc.__set__``
receives the descriptor object itself as ``value`` when a dataclass field
takes its default.  The guard for that case reassigns the instance array to
itself:

.. code-block:: python

    if isinstance(value, TTreeScalarDesc):
        value = getattr(obj, self.attrname)   # same array as `inst` below
    inst = getattr(obj, self.attrname)

    inst[0] = value                           # inst[0] = inst

``arr[0] = np.array([x])`` was tolerated by NumPy 1 and is rejected by NumPy 2:

.. code-block:: text

    numpy 2.5.2: arr[0] = array([x])  ->  ValueError

This is why it appears now.  Nothing in the code changed; the environment
moved.  The CI container that last ran successfully dates from January 2022
and carried NumPy 1.

**Fix.**  The instance array is already populated by ``create_default(obj)``
on the preceding line, so the assignment is a no-op in intent and is now
skipped:

.. code-block:: python

    if isinstance(value, TTreeScalarDesc):
        return          # default already installed by create_default()

**Measured effect** on the full suite, 30 August 2026:

=====================  ==========  ==========
Suite                  failed      passed
=====================  ==========  ==========
before                 123         216
after                  **15**      **324**
=====================  ==========  ==========

Guarded by ``tests/dataio/test_descriptor_defaults.py``, which constructs
every run and event tree with no arguments and checks that the default
survives as a scalar.  That the default is still correct after skipping the
assignment is the part worth testing: it confirms ``create_default`` was
always doing the work.


.. _issue-import-requires-root:

The package cannot be imported without ROOT
--------------------------------------------

:Status: open
:Affects: documentation builds, unit tests of pure functions, any use of the
          coordinate or geometry code on its own

``grand/dataio/descriptors.py`` evaluates ``ROOT.gROOT.GetVersionInt() >= 63600``
at module import time, and :mod:`grand.geo.topography` imports
:mod:`grand.dataio.protocol`.  The result is that importing anything under
``grand`` requires a full ROOT runtime, including importing it in order to
document it — the Sphinx configuration carries a typed mock for this reason.

Deferring the version check to first use, and breaking the geometry-to-dataio
dependency, are part of the interface work.

.. _issue-missing-endtoend-fixture:

The end-to-end test has no input
---------------------------------

:Status: open
:Affects: the only test that exercises the whole pipeline
:Test: ``tests/sim/test_efield2voltage.py``

``tests/sim/test_efield2voltage.py`` reads ``data/test_efield.root``.  That
file is not in version control: ``data/.gitignore`` excludes everything except
the readme, the download scripts and itself.  Nor is it produced by
``env/setup.sh``, which fetches the topography, geomagnetic and antenna data
but no test input.

So the test cannot pass on a fresh checkout or in CI.  The copy that happens to
exist in this working tree is 615 bytes and contains no trees at all.

This is also why the test asserts so little.  Its only check after
``compute_voltage()`` is that an output file exists -- which is all you can
assert when there is no reliable input to compare against.

**What it needs.** Either a small ROOT file committed as a fixture, or one
built in a pytest fixture from the tree classes themselves.  The second is
better: it costs nothing in repository size, it cannot drift from the schema,
and it makes the fixture's contents visible in the test rather than opaque.
Once it exists, the end-to-end regression described in the recovery plan --
peak voltage, trace RMS, band-integrated power against stored references --
becomes possible.

.. _issue-vga-gain-ignored:

The VGA gain setting has no effect
-----------------------------------

:Status: open
:Affects: any study that varies the amplifier gain
:Test: ``tests/sim/test_rf_chain_physics.py::test_gain_setting_changes_the_transfer_function``

``RFChain(vga_gain=...)`` accepts 20, 5, 0 or -5 dB, stores the value, asserts
it is one of those four, and logs it — and then loads the same S-parameter
file whatever it was.  The transfer function is identical for every setting:

.. code-block:: text

    vga_gain= 0   max|TF| = 94.7612
    vga_gain= 5   max|TF| = 94.7612
    vga_gain=20   max|TF| = 94.7612

**Cause.**  In ``VGAFilter._set_name_data_file`` the line that used the gain
is commented out, and the replacement reads a single fixed path from a
component configuration:

.. code-block:: python

    assert self.gain in [-5, 0, 5, 20]
    logger.info(f"vga gain: {self.gain} dB")
    #filename = os.path.join("detector", "RFchain_v2", "filter+"f"vga{self.gain}db+filter.s2p")
    filename = components["Filter"]["s2p_file"] if components["Filter"]["enabled"] else None

``components["Filter"]["s2p_file"]`` resolves to
``detector/RFchain_v2/feb+amfitler+biast.s2p``.  That is worth stating plainly:
the stage is not loading the *wrong* VGA table, it is **not loading a VGA table
at all** — that file is a front-end board with an AM filter and a bias tee, a
different component.  The three per-gain files ship in
``data/detector/RFchain_v2/`` and are never opened:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - File
     - Read by anything?
   * - ``filter+vga0db+filter.s2p``
     - no
   * - ``filter+vga5db+filter.s2p``
     - no
   * - ``filter+vga20db+filter.s2p``
     - no

Note also the mismatch between the assertion and the data: four values are
accepted, ``[-5, 0, 5, 20]``, and only three files exist.  There is no
``filter+vga-5db+filter.s2p``.  Uncommenting the line above would therefore fix
three of the four accepted settings and turn the fourth into a missing-file
error, so the assertion needs narrowing at the same time.

**Why it matters.**  Section 8.3 of `arXiv:2408.10926
<https://arxiv.org/abs/2408.10926>`_ states that the total transfer function
changes with the choice of VGA gain, and one of the library's stated purposes
is assessing the effect of changes to the detector design.  A comparison
across gain settings currently returns the same answer three times, with no
error and no warning.

**Not fixed here.**  Restoring the commented line would bypass the component
configuration that replaced it, which appears to be deliberate and part of
other work.  Whoever introduced that configuration should decide how the gain
selects a file within it.

.. _issue-reader-directory-coupling:

The file readers depend on an undocumented naming convention
-------------------------------------------------------------

:Status: open
:Affects: :mod:`grand.dataio.root_files`, and anything that opens a file
          through it
:Test: ``tests/dataio/test_root_files_reading.py``

``_FileEventBase.__init__`` does not read the file it is given in isolation.
It constructs a :class:`~grand.dataio.data_handling.DataDirectory` for the
*containing directory*, then looks up the run and shower trees by attribute
names chosen from a substring of the filename:

.. code-block:: python

    if f_name.find("_L0_") > 0:
        self.tt_shower = data_dir.tshower_l0
        self.tt_run = data_dir.trun_l0
    elif f_name.find("_L1_") > 0:
        self.tt_shower = data_dir.tshower
        self.tt_run = data_dir.trun

So opening one voltage file requires that its name carry ``_L0_`` or ``_L1_``,
that the analysis level recorded *inside* its trees agree with the one in the
name, and that the directory also hold matching run and shower trees grouped
under the naming scheme :meth:`DataDirectory.get_list_of_files_handles`
expects.  None of that is documented, validated or stated in an error message.

**Consequences.**  The module sits at 21% test coverage, not because it is
unimportant but because a valid input is difficult to construct: the existing
tests skip when a fixture that is not in version control is absent.  A user
who renames a file, or writes one from the tree classes directly, gets an
``AttributeError`` naming an attribute they have never heard of.

**What it needs.**  The reader should take the trees it needs, or accept them
explicitly, rather than rediscovering them from a directory listing.  That is
the sort of change Phase 6 of the recovery plan exists to make; until then the
convention should at least be written down and checked with a clear error.

One part of this is already improved: a bare ``raise`` with no active
exception, which produced ``RuntimeError: No active exception to reraise`` for
any file lacking the level marker, now raises a :class:`ValueError` naming the
file and the convention.

**The mechanism, measured.**  ``DataDirectory`` groups files by **filename
prefix**, not by the trees a file contains: ``ftshowers`` collects files whose
name begins with ``shower_``.  So a single file holding ``TRun``, ``TEfield``
and ``TShower`` is only ever seen as an efield file, and
:class:`~grand.dataio.root_files.FileEfield` then fails on it with

.. code-block:: text

    AttributeError: 'NoneType' object has no attribute 'file_name'

Three conditions must hold together for a file to be readable:

1. the run, efield and shower trees are in **separate files**, named
   ``run_*``, ``efield_*`` and ``shower_*``;
2. each name carries its analysis level as ``_L0_`` or ``_L1_``;
3. the ``analysis_level`` stored *in each tree* matches the name.

None of that is stated anywhere in the code.  It is now pinned by
``tests/dataio/test_root_files_reader.py``, which builds a conforming fixture
and also asserts that the single-file layout fails — so that the day the reader
looks inside the file it was handed, the test says so.

**Two smaller defects found while writing those tests.**
``get_du_count()`` returns 0 on a file whose ``TRun`` was written with
``du_id`` set, while the traces carry the right number of units; a caller
sizing an array from it gets nothing.  Recorded as an expected failure.  And
``get_du_nanosec_ordered()`` returns a ``(times, origin)`` tuple, not the
``ndarray`` its docstring claimed — ``np.asarray`` on the result raises.  The
docstring is corrected.

.. _issue-root-638-numerical-difference:

ROOT 6.38 changes the result of a NumPy-only test
---------------------------------------------------

:Status: open, cause not established
:Affects: unknown; one test is currently known to differ
:Test: ``tests/basis/test_traces_event.py::test_remove_trace_low_signal``

The first run of the new ROOT matrix found a difference that has nothing
obviously to do with ROOT.

``test_remove_trace_low_signal`` builds five traces whose three components are
set to 1, 10, 0.5, 11 and 1, giving Euclidean norms of about 1.73, 17.3, 0.87,
19.05 and 1.73.  With a threshold of 5, two traces should survive.  Under ROOT
6.36.04 two do.  **Under ROOT 6.38.02, three do.**

The comparison is not close to the threshold, so this is not a rounding
difference at a boundary.  Python, NumPy and every other package are identical
between the two legs -- 3.12.14 and 2.5.2 respectively -- and only the ROOT
version differs.  The function itself, ``Handling3dTraces.remove_trace_low_signal``,
is pure NumPy and never touches ROOT.

**What is not yet known.**  Whether importing ROOT 6.38 changes floating-point
behaviour process-wide (its JIT does emit CPU-feature promotions on some
hardware), whether some shared state differs between the two runs, or whether
the test is order-dependent in a way that only manifests on one leg.  Anything
written here beyond the observation would be a guess.

**Why it matters.**  If importing ROOT can change the result of NumPy code
that does not use ROOT, that is a much broader problem than one test, and it
would affect every number the pipeline produces.  If instead the test has
hidden state, the test is wrong and should be fixed.  The two possibilities
are very different and the first one is worth ruling out promptly.

**Handling meanwhile.**  The 6.38 leg of the matrix reports but does not block:
6.36 is the supported version and gates, while 6.38 stays visible without
holding up work.  This is the difference the matrix was added to find, so it
should not be silenced.

.. _issue-geomagnetic-model-expired:

The geomagnetic model expired on 1 January 2025
-------------------------------------------------

:Status: open
:Affects: any use of the geomagnetic field, or of a magnetic-north frame, at
          a present-day date
:Test: ``tests/geo/test_geomagnet_validity.py``

``data/geomagnet/IGRF13.COF`` is the thirteenth generation of the
International Geomagnetic Reference Field, which is defined from 1900 to
**2025**.  GULL rejects any date at or beyond the end of that range:

.. code-block:: text

    2020-01-01: OK
    2024-06-01: OK
    2025-01-01: LibraryError: missing data in file .../IGRF13.COF
    2026-01-01: LibraryError: missing data in file .../IGRF13.COF

So a magnetic-north frame, or a field evaluation, at any date since the start
of 2025 fails outright.

**Why nobody noticed.**  The default observation time throughout
:mod:`grand.geo.coordinates` is the literal string ``"2020-01-01"``:

.. code-block:: python

    obstime: Union[str, datetime] = "2020-01-01",  # calculate declination of what date?

That default sits comfortably inside the valid range, so everything works
until someone passes a real date -- at which point it stops working, more than
a year after the model lapsed.  The trailing comment in the source suggests
the choice of epoch was never settled.

**Why it matters.**  The geomagnetic field is one of the three physics inputs
the library owns, and it drives the radio emission.  A simulation of data
taken in 2025 or 2026 cannot currently evaluate the field at the time the data
were taken; it can only use a stale epoch, silently, via the default.

**The fix.**  IGRF-14 was released at the end of 2024 and covers 2025 to 2030.
Shipping it, and reconsidering whether a hard-coded default epoch is wanted at
all, would resolve both halves of this. A default that is a fixed date in the
past is the kind of thing that works for years and then quietly stops being
right.
