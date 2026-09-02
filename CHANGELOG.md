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
- CI has not completed a run in a long time, for three independent reasons
  recorded in `docs/dev/FINDINGS_CI.md`.
