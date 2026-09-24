# Changelog

All notable changes to **GRANDlib** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

Note that the **package** version tracked here is distinct from the **ROOT
data-format** version in `grand/dataio/version`, which has its own cycle: a
package release does not imply a schema change, or the reverse.

## [Unreleased]

Work on the `dev-next` integration branch, ahead of the first tagged release.

### Fixed

- **Xmax is placed right whatever convention a file uses**
  (grand-mother/grand#160). The committed ZHAireS samples store Xmax above sea
  level, 1264 m too high; today's converter stores it above the ground. Before
  this change, only `root_files.py` corrected for it, and it subtracted the
  ground altitude from every file, so freshly converted data came out 1264 m
  low. `gen_shower.py` (the antenna response), `aoi/event.py` and the event
  viewer corrected nothing, so they were wrong on the samples, the viewer's
  angular plane by up to 6.5°. The new `grand/dataio/xmax_frame.py` reads the
  frame from the file: Xmax must lie along the stored arrival direction, and
  only one reading does (0.001° against at least 0.49° on all 12 committed
  events). All four readers use it. A value that fits neither reading is left
  as stored, with a warning. Decided 2026-09-24: detect the frame rather than
  regenerate the samples, since DC2 data was written the old way too.

- **The Handbook PDF warns against `requirements_vers.txt`.** A new erratum,
  because the PDF is compiled from the unchanged LaTeX; the online page was
  already corrected.

- **The recovery paperwork counts branches correctly.** The plan said "5
  still out" when three were: `docs/dev/branch_facts.py` measured branches
  against the local `dev-next`, which can lag the remote, and counted the
  recovery's own working branch as undecided work. It now measures against
  `origin/dev-next` and shows that branch as the working branch. The
  regenerated diagrams also show the 634 tests CI reports, and no longer list
  the reconstruction scope as blocked, since it was decided on 2026-09-24.

- **Installed GRANDlib contains the Galactic noise model.** `grand/sim/noise/`
  had no `__init__.py`, so package discovery skipped it and a built package
  shipped without `galaxy.py`. Found by a new test that checks every
  directory of code under `grand/` is packaged; the same test caught
  `grand/analysis/coords/` before it could ship.

- **The CoREAS site table is in metres and names an unknown site.** It
  stored altitudes in centimetres (kept out of the output only by a later
  override line) and crashed on any other site with "not enough values to
  unpack". Xiaodushan, a real GRAND site, was one of them. It now raises
  `ValueError: unknown site 'Xiaodushan': ... knows only Dunhuang, Lenghu`.

- **The event viewer's vxB axes point the right way.** It treated
  `magnetic_field` -- [inclination, declination, strength] -- as a vector
  and normalised it, giving a field 104° from the real one at Xiaodushan
  and turning its shower-plane and angular-plane axes by 91–114°. It now
  builds the direction from the angles; a new test compares it with the
  geomagnetic model at the site, and fails on the old code.

- **The golden pipeline test no longer fails at random on CI.** It divided
  each difference by that sample's own value, so rounding on a sample near a
  zero crossing read as a large relative error. On 2026-09-24 one leg failed
  at 2.59e-5 on a sample of -1.178 in a trace peaking at 3258, where the
  actual difference was 9.4e-9 of the peak. Differences are now measured
  against each trace's peak; real changes (the √2 noise fix, a one-sample
  shift, a 1e-5 change) still fail it.

- **The CoREAS converter runs on its own fixture.** `CoreasToRawROOT.py -d
  proton` raised `UnboundLocalError: Xmax_NWU` partway through and left a
  447-byte `.rawroot` behind (grand-mother/grand#159). `Xmax_NWU` was computed
  only when the `.reas` carries `ShowerZenithAngle`, and the short
  `SIMxxxxxx.reas` that directory mode reads never does. That branch has no
  `DistanceOfShowerMaximum`, so it now writes NaN for the position; zeros would
  have read downstream as a real Xmax at the array origin. The CoREAS chain now
  runs end to end through `sim2root.py` for the first time, and writes
  `origin_geoid[2] = 1200` in metres, where the 2024 fixture carries `114200`.

  NaN is one of several defensible choices, and the known-issues entry had
  left that choice to the owners of `sim2root/`. Reading the long per-event
  `.reas`, which the fixture has, would give a real position instead. See
  `issue-coreas-xmax-unbound`.

- **A vertical CoREAS shower keeps its own parameters.** The same converter
  tested `if read_params(...ShowerZenithAngle):`, so a zenith of exactly 0.0
  was falsy and fell through to hard-coded Dunhuang values. It now tests for
  presence (`is not None`). This one is reasoned from `read_params`, not
  measured; no fixture has a vertical shower.

- **Tree classes can be constructed again under NumPy 2.** `TRun()`,
  `TADC()` and every other tree raised `ValueError: setting an array element
  with a sequence`. `TTreeScalarDesc.__set__` bound `value` to the same array
  as `inst` when a dataclass field took its default, so the assignment became
  `inst[0] = inst`; NumPy 1 tolerated a one-element array in a scalar slot and
  NumPy 2 does not. `create_default()` had already installed the default, so
  the assignment is now skipped.

  Nothing in the code changed when this appeared — the environment moved. The
  container in which CI last ran successfully dates from January 2022 and
  carried NumPy 1, which is why no test caught it. The suite went from 123
  failed / 216 passed to 15 failed / 336 passed.

- Malformed table in the architecture documentation, where em-dashes pushed
  the first column past its rule.

### Added

- **The DC1 analysis scripts are kept, in `examples/old/dc1/`.** `beta_dc1`
  (grand-oma, January 2023) was merged to keep its two display scripts and
  their history. They use the pre-2023 API and do not run; the README lists
  the four renames needed to port them.

- **Two notebooks: reconstruction, and the event viewer.**
  `notebooks/11_reconstruction.ipynb` runs `grand.analysis` step by step: plane
  wave, spherical wave, angular distribution function and energy proxy. It
  starts on showers it generated, so the answer is known, and ends on the ten
  GP13 candidates in `examples/analysis/`. There it shows what the tests could
  not: most of those events have timing χ²/ndf far above 1, and 8 of 10
  amplitude fits stop on the upper bound of the ring width, which nothing
  flags. `notebooks/12_event_viewer.ipynb` shows how to run the viewer, redraws
  each panel as a static figure so it reads on GitHub, and measures the
  angular-plane error from #160 on both sample events (6.5° and 1.0°), which
  the Xmax detection below now corrects.
  Both are generated by `notebooks/make_notebooks.py` and listed in the
  documentation. The weekly notebook job now installs the viewer extra.

- **The ADF fit has a round-trip test.** Amplitudes generated by
  `ADF_parameters` fit back to their direction, width and scale. It is the
  only fit independent of timing, and it had no test. Checked to fail when
  the fit's azimuth bound is narrowed.

- **Reconstruction: `grand.analysis`** (Marion Guelfand, from `dev_marion`).
  Plane-wave, spherical-wave and ADF fits of arrival direction and Xmax
  distance, an electromagnetic-energy proxy, signal extraction, footprint
  geometry and a Cherenkov-angle model, with results in a new `TRecons` tree.
  Needs `iminuit` (`pip install -e ".[analysis]"`; in the conda
  environment). Tested for self-consistency -- each fit recovers what its
  forward model generates -- not yet against simulated or measured showers.
  Examples in `examples/analysis/`.

- **GRAND's angle convention is stated and pinned.** A shower's zenith and
  azimuth name where it comes from; the coordinates page now says so, and
  `tests/geo/test_angle_convention.py` checks the core transform against the
  ZHAireS summaries. Chosen over `snonis_sim2root_test_merge`, which flipped
  the transforms and would have put every computed angle at odds with every
  stored one.

- **Decision material for the three branches waiting on the collaboration**
  (`dev_marion`, `grandio_light`, `snonis_sim2root_test_merge`), in
  `resources/dev/dev-next/DECISIONS.md`: what each changes, what merging it
  would also do, and the question it asks, each measured.

- **The schema snapshot notices new trees.** It compared only a hard-coded
  list of tree classes, so a new tree -- the largest format change there is
  -- passed unseen: `dev_marion`'s 27-field `TRecons` left it green. A new
  test finds every tree class in the data modules and fails on any that is
  not pinned. It also found `TRunRawVoltage`, on the trunk and never pinned;
  it is now in the snapshot.

- **A `Tests gate` CI check** that is green only when the test suite ran and
  passed, or when a commit changed documentation only, in which case it says
  that nothing ran. A skipped test job used to read as a pass. Designed to be
  the required check for `dev-next`; making it required is an admin setting.

- **Acknowledgement of the NCN OPUS grant** (no. 2022/45/B/ST2/02889) in
  `README.md`, copied verbatim from `dev`, where it was added to `README.rst`
  on 2026-09-15 after `dev-next` had replaced that file.

- **One environment for everything.** `env/conda/grand-dev.yml` consolidates
  four dependency lists that had drifted apart: the previous runtime file, the
  pip-installed test and lint tools, a third set under `env/docker_*/` that was
  the only one carrying `numba` and `lmfit`, and documentation dependencies
  that nothing declared. Build with `--solver=libmamba`.

- **The package is installable.** `pyproject.toml` makes `pip install -e .`
  work, so `import grand` no longer needs `PYTHONPATH`. Scope is deliberately
  narrow: the C extension is still built by `env/setup.sh`, and no console
  entry points are declared yet.

- **Documentation, rebuilt.** A single Sphinx tree at `docs/source/`, replacing
  five scattered sources. Narrative pages for installation, quickstart,
  coordinates, the data model, architecture and known issues, with examples
  that execute against the real library when the page is built. Appendix A of
  the paper is ported into the coordinates page.

- **Known issues page** recording the Galactic-noise normalisation, the NUTRIG
  field-name collision, and the import-time ROOT dependency, each with what is
  measured, what is not, and what would settle it.

- **Tests.** Schema snapshot pinning the ROOT tree layout, with a guard against
  the NUTRIG name collision; Galactic-noise normalisation, asserting what holds
  under any convention and recording what does not; descriptor-default
  regression covering every tree class.

- **The converters are run, not just read.** `tests/sim2root/test_xmax_frame.py`
  runs the ZHAireS and CoREAS converters on the repository's own fixtures and
  checks Xmax against the ZHAireS `.sry` summaries. Until now `tests/sim2root/`
  only parsed the sources, which is how a crash on the committed CoREAS
  fixture went unnoticed. Each test was checked to fail with its defect put
  back.

- **Tools.** `quality/premerge_check.py` reports what a branch adds and flags
  fields that duplicate an existing field's meaning; `quality/docstring_coverage.py`
  measures the numpydoc conversion; `docs/dev/make_recovery_diagram.py` and
  `make_frames_diagram.py` generate the documentation figures.

- Numpydoc docstrings, in progress, enforced by a ruff `D` ratchet whose
  ignore list may shrink and must never grow.

### Changed

- **The Handbook's "Directory Structure" page is maintained by hand.**
  `docs/dev/build_handbook.py` rewrote every Handbook page from the LaTeX
  source, and in September 2026 a regeneration silently undid the 2026-09-08
  correction on that page, which removed the instruction to install
  `requirements_vers.txt`: a 2022 pin set including a Pillow with a critical
  advisory. The correction was restored at the time; the decision now (the
  repository owner's) is that the hand edit wins. The generator keeps any page
  listed in `HAND_MAINTAINED`, and refuses, before touching anything, if such
  a page no longer matches a section of the source. The PDF is built from the
  LaTeX and still carries the old instruction.

- **`grand.recon` removed.** It held two classes with a constructor and
  nothing else; reconstruction is `grand.analysis`.

- **Reading GRAND files no longer loads the physics.** `grand/__init__.py`
  loads its public names on first use instead of importing the geometry and
  simulation code eagerly. `import grand.dataio` now loads the data layer
  only, and works without the compiled C core; `import grand` and
  `grand.geo.coordinates` no longer need ROOT. Every public name still
  resolves, and `from grand import *` now works (it raised on `adc`, listed
  in `__all__` but never imported). Chosen over the `grandio_light` branch,
  which reached the same goal by deleting the physics.

- Merged from the outstanding queue: `dev_fix_root_warnings_lwp` (ROOT 6.38
  warnings), `dev_nutrig_fields`, `dev_reprocessing` (Snakemake pipeline),
  `dev_Event_write`, and `dev_aoi_unittest` (~3000 lines of tests, stripped of
  ten generated summary documents and three stray fixtures).

### Known

- `magnetic_field` stores [inclination, declination, strength] with the
  strength in µT from ZHAireS and in gauss from CoREAS, and no unit
  recorded. See `issue-magnetic-field-units`.
- The Galactic-noise normalisation does not reproduce the tabulated model: the
  simulated RMS is 0.33 of the Parseval value with the current `size_out/2`,
  and would be 0.47 with the proposed `size_out/sqrt(2)`. Neither is 1. See
  the known-issues page.
- Two branches add the same NUTRIG correlation fields to `TADC` under
  different names; the schema decision is unresolved.
- CI has not completed a run in a long time, for three independent reasons
  recorded in `docs/dev/FINDINGS_CI.md`.
