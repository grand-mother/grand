Known issues
============

.. contents::
   :local:
   :depth: 1

Open problems that affect results or block work, with what is measured, what
is not, and what would settle each one.  Fixed issues move to the changelog
rather than staying here.

.. _issue-galactic-noise-normalisation:

Galactic-noise normalisation does not match the tabulated model
---------------------------------------------------------------

:Status: open, blocking
:Affects: every simulated voltage and everything downstream — trigger
          studies, sensitivity estimates, Data Challenge outputs
:Blocks: ``dev_snonis`` (PR 153), ``refact_galaxy`` (PR 146)
:Test: ``tests/sim/test_galactic_noise_normalisation.py``

**Symptom.**  The RMS of the simulated Galactic noise does not equal the
value Parseval's theorem gives for the tabulated model it is built from.

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

**Measurement.**  Made on 30 August 2026 against
``data/noise/Vocmax_30-250MHz_uVperMHz_hfss.npy`` at LST 18 h, with 600
antennas, 221 bins of 1 MHz placed in a 2048-point transform:

===========================  ===================
Normalisation                simulated / model
===========================  ===================
``size_out / 2`` (current)   0.33
``size_out / sqrt(2)``       0.47
===========================  ===================

Neither is 1.  **The √2 change moves the result towards the model but does
not reconcile it**, and a factor of roughly 2 remains unaccounted for.  The
disagreement is therefore not a choice between two constants, which is how it
has been framed.

**What is not in doubt.**  The implementation is internally consistent: the
time series obtained from the returned spectrum carries that spectrum's
energy, to one part in :math:`10^9`.  Whatever convention is chosen, the
transform itself is right.

**What would settle it.**  One definitional question, for the authors of the
table:

    Is ``Vocmax_30-250MHz_uVperMHz`` an **RMS** voltage spectral density, or a
    **maximum**?

The name says *max*.  If it is a peak rather than an RMS quantity, the
reference used in the measurement above is wrong by a known factor and the
comparison shifts accordingly — possibly onto one of the two candidates.  If
it is an RMS, then neither candidate is correct and the chain needs a closer
look.

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
