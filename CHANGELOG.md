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

- **Seven open issues, fixed together in 2026-09.**
  - **Simulation files record Xmax and the shower direction** (#104).
    `tshower.xmax_pos` and `tshower.direction` were all zeros in `sim2root`
    output. `xmax_pos` is now Xmax in the site frame (that of `du_xyz` and
    `shower_core_pos`): the above-ground `xmax_pos_shc` plus the core, the
    point readers already computed; NaN when unknown. `direction` is the unit
    vector of propagation, minus the "comes from" vector of zenith and
    azimuth, so its z is negative for a downgoing shower; it equals the
    `primary_inj_dir_shc` the ZHAireS converter already wrote. The AOI now
    writes both `xmax_pos` and `xmax_pos_shc`: it wrote only `xmax_pos`, as
    the core-relative point, and read `xmax_pos_shc`, so Xmax was lost on a
    round trip. The `TShower` zenith and azimuth docstrings said "pointing
    to"; they now say "comes from", as decided on 2026-09-24.
  - **Reading many files no longer leaks their memory** (#71). A tree that
    opened its ROOT file from a name never closed it, because PyROOT gives up
    ownership after `TTree.SetDirectory()`. `stop_using()` now closes that file
    once no other tree uses it; a `TFile` the caller passed in is left open.
    Trees are also context managers, `with TADC(path) as tadc: ...`.
    Measured on 200 distinct ADC files: 2.65 MB per file before, 0.15 after.
    About 0.08 MB per file remains inside ROOT 6.36 itself, even from C++.
    Releasing trees is now documented in the data-model and troubleshooting
    pages.
  - **Showers that hit no antenna go through the pipeline** (#91). They are
    kept at every level, with their shower and run information and
    `du_count` 0, so that effective-area studies can count them. Before,
    `sim2root` stopped with "ecef coordinates must be n x 3", and
    `Efield2Voltage` and `convert_voltage2adc.py` with an `IndexError`.
  - **No more `TTreeCache::FillBuffer` errors when appending events** (#89).
    Writing more than about 100 events one at a time, as
    `convert_efield2voltage.py` does, made ROOT report a cache inconsistency
    on every event. Values were right; the message was noise, and hidden
    since 2024-08 by the error level `grand.dataio` sets. The cache's entry
    window is now reset before every full pass over a tree.
  - **The ZHAireS refraction-model reader works** (part of #140).
    `GetRefractionIndexModelFromSry` failed on every call (a misspelt file
    handle) and would otherwise have answered "Constant" for any file.
    Nothing calls it yet; the layered model #140 asks for will need it.

- **The reconstruction examples save their results again.**
  `examples/analysis/main_DOI.py` stored every reconstructed value in a
  `TShower`, and `main_AOI.py` in the event's `Shower`, but neither has those
  fields. Setting an undeclared field on a tree raises nothing, and the file
  was written without it, so nothing was saved; the printout read back the
  in-memory object and looked right. The examples relied on a `TShower`
  extension that was never committed (817bb79e). Both now use `TRecons`,
  which declares every one of these fields and which `display.py` already
  reads. `main_AOI.py` writes it to `reconstructed_events_AOI/recons.root`
  beside its per-event files. A test checks that every value the two scripts
  store is a `TRecons` field. Also fixed there: `sep='\s+'`, an invalid
  escape that Python warns about.

- **Four long-standing issues.**
  - `EventList.get_event` with an event number the input does not hold now
    returns `None`, as documented (#95). It used to crash deep in the reader,
    or, after a valid event, return an event labelled with the requested
    number but carrying the previous event's traces.
  - `convert_voltage2adc.py` finds noise files in a directory given without
    a trailing '/', and stops when it finds none (#122). Before, every ADC
    sample became 8192. A path prefix still works as before.
  - The CoREAS converter reads the hadronic model and CoREAS version from the
    log instead of matching one literal string (#123). Other models and
    versions are no longer written as "n/a"; Sibyll 2.3d and 1.4 read as
    before.
  - A `creation_datetime` or `source_datetime` set on a tree now reads back
    as the datetime that was set, not its integer timestamp. Found while
    triaging #136; an unset value still reads as 0, as its maintainer chose.

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

- **Files record the code that wrote them** (#137). Every new tree's
  `modification_software` and `modification_software_version` hold
  "GRANDlib" and the package version with the git branch, commit and whether
  tracked files were modified, e.g. `0.1.0.dev1 (git dev-next 1aeac037...)`.
  Both were always empty. Outside a git checkout, only the version is given,
  so an install inside another repository never reports that repository's
  commit. In `grand.provenance`.

- **An offline T1 trigger in `convert_voltage2adc.py`** (#139). The
  DAQ-style T1 logic of `scripts/T1_trigger_offline.py` is now
  `grand.sim.detector.trigger`, and it evaluates every DU: the script looked
  only at the first. The converter gains an opt-in `--t1_trigger` (with
  `--t1_param KEY=VALUE` overrides) that sets `trigger_flag` per DU (1 =
  passed); without it the output is unchanged, checked branch by branch. The
  default parameters are the offline script's and **need confirming by the
  trigger group**: with them, no DU of the committed sample triggers, even
  with pulses of about 1000 ADC, because crossings must be at most 8 ns apart
  (`t_sepmax=10`).

- **Cramér-Rao bounds for the reconstruction** (Sebastián Castro-Isern,
  PR 150). `grand.analysis.cramer_rao_bounds` (also `grand.analysis.crb`)
  gives the smallest uncertainty the PWF, SWF and ADF fits can reach, from
  the timing and amplitude uncertainties. `TRecons` gains ten `crb_*` fields
  to store them, a data-format addition (schema snapshot updated). Files
  written before this still read, with the bounds reading as 0.
  `examples/analysis/main_DOI.py` computes and stores them. Ported by hand
  because the PR targets `dev`; the other files in PR 150 were already on
  `dev-next` from `dev_marion`, apart from its copy of the example, which
  duplicates `main_DOI.py`. Two changes to the code as submitted:
  - an azimuth, or any other parameter, of exactly 0 got a derivative step
    of 0, and every bound came back NaN; it now gets a step of 1e-6;
  - three docstrings are corrected: one named the wrong model and two called
    the zenith the azimuth.
  Tested against 1000 refits of noisy plane-wave times: the PWF bound matches
  their scatter to within 3 %.

- **Branches can be retired without losing them.** `docs/dev/archive_branches.py`
  tags each settled branch as `archive/<branch>-<YYYY-MM>`, with the reason in
  the tag message, before deleting it. Every commit stays in the repository,
  and `git branch <name> <tag>` restores one. `BRANCHES.md` gains an
  "Archived as" column and keeps listing deleted branches from their tags. Its
  "Still out" heading, which counted decided branches too, is now "Not in
  `dev-next`", with the breakdown.

- **Every branch is now settled.** `refact_galaxy`, the last one open, is
  not merged, with luckyjim's agreement: its fixes are already on the trunk
  and the rest re-implements the verified galactic-noise model.

- **Star-shape trace interpolation is kept, in `examples/old/radio/`.** From
  the 2019 `radio` branch (Anne Zilles, Valentin Niess): the only code in the
  repository that interpolates an electric-field trace to a new antenna
  position. Copied with the two helpers it imports; it does not run as it is,
  and the README says what porting would take. The rest of the branch is
  rewritten on the trunk or unfinished, so the branch is retired.

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
