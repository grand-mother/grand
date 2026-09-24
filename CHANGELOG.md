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

- Merged from the outstanding queue: `dev_fix_root_warnings_lwp` (ROOT 6.38
  warnings), `dev_nutrig_fields`, `dev_reprocessing` (Snakemake pipeline),
  `dev_Event_write`, and `dev_aoi_unittest` (~3000 lines of tests, stripped of
  ten generated summary documents and three stray fixtures).

### Known

- The Galactic-noise normalisation does not reproduce the tabulated model: the
  simulated RMS is 0.33 of the Parseval value with the current `size_out/2`,
  and would be 0.47 with the proposed `size_out/sqrt(2)`. Neither is 1. See
  the known-issues page.
- Two branches add the same NUTRIG correlation fields to `TADC` under
  different names; the schema decision is unresolved.
- **The committed ZHAireS samples put Xmax 1264 m too high, and one reader
  corrects for it** (grand-mother/grand#160). The converter itself is right:
  run on the committed `.sry` files today, it writes the ground-relative
  height exactly (4499.41 m for event 1618). The samples predate that
  subtraction. `root_files.py` compensates with its "DC2 FIX"; `gen_shower.py`
  (antenna response) and `aoi/event.py` do not. So on the committed samples
  those two are wrong, and on regenerated samples `root_files.py` would be
  wrong instead, by 1264 m in the other direction. No vintage of the data
  makes every reader right. See `issue-xmax-sample-vintage`;
  `tests/sim2root/test_xmax_frame.py` fails if either the samples or the
  reader changes alone, so the two cannot drift apart unnoticed.
- CI has not completed a run in a long time, for three independent reasons
  recorded in `docs/dev/FINDINGS_CI.md`.
