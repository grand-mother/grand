# `dev-next` beta test: plan and tracker

**Status:** wave 2 running; Critical fixes in PR #208 · **Last updated:** 2026-10-01 · **Branch under test:** `dev-next` at `91d30a1b`

This page is both the plan and the live record of the beta test of GRANDlib's
`dev-next` branch. It is updated as the test runs: every confirmed problem
gets a GitHub issue, listed in [the tracker](#tracker) below with its fix.

---

## 1. Goal

Before `dev-next` becomes the default branch (Phase 9 of the
[recovery plan](../RECOVERY_PLAN.md)), find what is broken, missing or
misleading, from the point of view of the people who will use it:

- **a beginner**, who follows the documentation from a fresh clone and must
  be able to get results without help;
- **an expert**, who uses the library directly and needs it to do exactly
  what its documentation says;
- **anyone feeding it real-world input**: missing, empty, corrupt or unusual
  files must give a clear error, never a crash deep in the code or a silent
  wrong result.

The test also checks that the physics is right, independently of the tests
already in the repository, and that the documentation matches the code.

## 2. What is tested

Everything a user can reach on `dev-next`:

| Area | Where |
|---|---|
| Installation and setup | `env/` (conda environment, `env/setup.sh`), `pyproject.toml`, `README.md` |
| Notebooks 01–12 | `notebooks/` (coordinates, data model, antenna response, RF chain, galactic noise, e-field → ADC, topography, pipeline regression, reading events, finding data, reconstruction, event viewer) |
| Library | `grand/` (`dataio`, `aoi`, `sim`, `geo`, `basis`, `analysis`, `provenance`) |
| Command-line scripts | `scripts/` (15 scripts, e.g. `convert_efield2voltage.py`, `convert_voltage2adc.py`, `convert_efield2efield.py`, `extract_events.py`, `T1_trigger_offline.py`) |
| Simulation conversion | `sim2root/` (ZHAireS and CoREAS converters, `sim2root.py`, `RunSimPipe*.py`) |
| Examples | `examples/` (analysis, event viewer, `examples/old/` excluded) |
| Documentation | `docs/source/` (including the Handbook), READMEs, docstrings |

Out of scope: `granddb/` (database access, needs credentials),
`examples/old/`, and anything needing the computing centre (`/sps`).

## 3. Test environment and data

### Software
- Conda environment from `env/conda/grand-dev.yml`: Python 3.12, ROOT 6.36.04, NumPy 2.
- C extensions (TURTLE, GULL) built by `source env/setup.sh`.
- CI runs the same suite on ROOT 6.36.04 and 6.38.02.

### The GRAND data model (downloaded 2026-10-01)

Most of the simulation chain needs GRAND's data model. It is not in the
repository, because it is about 1 GB.

| | |
|---|---|
| Archive | `grand_model_20250313.tar.gz`, 976,538,181 bytes |
| Source | `https://forge.in2p3.fr/attachments/download/444342/grand_model_20250313.tar.gz` |
| Version pin | `data/model_version.flag` (`444342 20250313`) |
| Fetched by | `data/download_data_grand.py`, called from `env/setup.sh` |

It unpacks into three directories under `data/`:

| Directory | Size | Contents | Needed for |
|---|---|---|---|
| `data/detector/` | 999 MB | Antenna effective length per arm (`Light_GP300Antenna_*_leff.npz`, default, `mat` and `nec` models), RF chain v2 (`RFchain_v2/`: LNA, filter, cable, balun) | Antenna response, RF chain, e-field → voltage, the full pipeline |
| `data/noise/` | 22 MB | Galactic noise tables (`galactic_PL_per_Hz_*`, `Vocmax_*`, `Pocmax_*`, LFmap) | Galactic noise in simulations |
| `data/topography/` | 25 MB | SRTM elevation tile `N41E096.hgt` (GP300 site) | Topography, ground altitude |

Notes:
- **Network access.** The cloud test environment had to be allowed to reach
  `forge.in2p3.fr` (environment settings → Network access → allowed domains).
  Without it, the download fails, and 73 tests fail or error for lack of data.
- **A committed file the archive also ships.** `data/noise/galactic_PL_per_Hz_gp13_GP300.npy`
  is in the repository and in the archive. The download overwrites it with
  bytes that differ but values that are identical (checked: ratio 1
  everywhere). The committed copy is restored with
  `git checkout -- data/noise/galactic_PL_per_Hz_gp13_GP300.npy`.
- **Baseline with the data in place:** the full suite gives 727 passed,
  10 skipped, 11 expected failures, 0 failures. This is the reference every
  fix is checked against.

### Sample data used by the testers
All committed, no download needed:
- `sim2root/ZHAireSRawRoot/`: two ZHAireS showers (events 1618 and 13790), raw and `.rawroot` (regenerated in PR #177);
- `sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/`: the same events converted (efield, voltage, ADC, shower, run);
- `sim2root/Common/LongNoiseTraces/`: measured GP13 noise (2024);
- `examples/analysis/`: ten GP80 cosmic-ray candidates with reconstruction output.

## 4. The testers

About 28 AI agents (Claude Sonnet; the model is set explicitly on each), in
four waves. Each agent plays one kind of user, on a narrow area so it goes
deep rather than broad. Each works in its own isolated copy of the
repository, with the data model linked rather than copied, and **reports
only**: testers do not change code or post on GitHub. Within a wave, 4–5 run
at a time, so findings can be confirmed as they arrive.

Two checks run twice, by agents that do not know about each other (marked
**×2**). If the two runs find different problems, coverage is not yet
saturated and the area gets another pass; if they agree, it is done.

### 4.1 Input validation review (before wave 1, 2026-10-01)

Done by the coordinator before the testers start, to know how much of the
library checks its input at all. Scripts: `scratchpad/scan.py` (static) and
`scratchpad/probe.py` (dynamic), not committed.

**Static count.** Of the 328 public functions and methods in `grand/` that take
arguments, **61 (18 %) check their input explicitly** (a deliberate `raise` of
`ValueError`/`TypeError`/…, or an `isinstance` test). Another 43 rely only on
`assert`, which Python skips under `-O` and which gives no message.

| Package | Public functions | Explicit checks | `assert` only |
|---|---|---|---|
| `grand/analysis` | 39 | 0 (0 %) | 0 |
| `grand/aoi` | 38 | 6 (15 %) | 0 |
| `grand/basis` | 41 | 7 (17 %) | 6 |
| `grand/dataio` | 68 | 12 (17 %) | 0 |
| `grand/geo` | 49 | 19 (38 %) | 0 |
| `grand/sim` | 85 | 16 (18 %) | 37 |
| other | 8 | 1 | 0 |

Also: `exit()` in `grand/aoi/event_list.py` (4 places, kills the caller's
program) and `grand/aoi/event.py:1061`; a plain string raised at
`grand/aoi/event.py:732` and `grand/dataio/data_handling.py:295` (itself a
`TypeError`).

**Dynamic probe.** 60 entry points called with wrong types, signs, ranges,
shapes, NaN, empty arrays, and missing or non-ROOT files:

| Outcome | Count | Examples |
|---|---|---|
| Stopped by a deliberate check, clear message | 7 | `Geodetic(latitude='abc')`; `galactic_noise(f_lst=30)` ("must satisfy 0 <= f_lst < 24"); `galactic_noise(du_type='BAD')`; `TADC.trace_ch = 'abc'` |
| Error, but only by accident (NumPy or ROOT failing inside), message unhelpful | 12 | `xmax_above_ground([1, 2], …)` → "cannot reshape array of size 2"; `PWF_semianalytical` with 2-column positions → `IndexError`; `recons_energy_from_voltage(array)` → "truth value … ambiguous" (the docstring says arrays are accepted) |
| Crash from deep inside, wrong exception type | 13 | `TRun.run_number = -1` → `OverflowError`; `AntennaModel(du_type='BAD')` → `AttributeError: … no attribute 'leff_sn'`; `RFChain(vga_gain='a')` → bare `AssertionError`; `recons_energy_from_voltage(1e3, 0)` → `ZeroDivisionError`; a text file given as ROOT → `OSError` from PyROOT |
| **Accepted silently**, wrong or meaningless result | **27** | `Geodetic(latitude=200)`; `TShower.zenith = 500`; `TShower.energy_primary = -1`; `TRun.t_bin_size = [-2]`; `TRun.du_xyz = [[1, 2]]` (2 coordinates); `TRun.run_number = 1.7` stored as 1; `TRun('missing.root')` **creates an empty file on disk** and reports 0 entries, so a typo in a file name looks like an empty file; `DataDirectory('missing_dir')` and `EventList(123)` succeed; `PWF_semianalytical` with 4 positions and 3 times prints a message and returns `None`; `CRB_PWF` with 2 antennas returns bounds of 10⁵–10⁸ rad instead of saying the fit is underdetermined; `get_peak_time(dt_ns=0)` returns 0; negative frequencies accepted by `galactic_noise` and `RFChain` |
| Kills the caller (`exit()`) | 1 | `EventList('missing')` |
| Hang | 0 | |

**Conclusion.** Validation is thin and inconsistent. The worst cases are the
silent ones: they produce wrong numbers or empty objects that fail much later,
far from the cause. The `grand.geo` coordinates and the galactic-noise entry
point are the models to follow: they check type and range and name the
argument.

**What the fix looks like**, proposed for a dedicated PR before or alongside
wave 1:

1. A small shared helper module (for example `grand/basis/validate.py`) with
   checks that name the argument: `finite`, `positive`/`non_negative`,
   `in_range`, `integer`, `shape`/`ndim`, `same_length`, `one_of`,
   `existing_file`. Each raises `ValueError` or `TypeError`.
2. Use them at the public entry points first: coordinates, the `dataio`
   readers and tree setters (type, shape, unsigned range; physical ranges
   such as zenith and energy as errors on writing), `grand.analysis`,
   `AntennaModel`, `RFChain`, `galactic_noise`, `Handling3dTraces`, the
   `xmax_frame` helpers, and `EventList`.
3. Replace `assert` used for input checks, `exit()`, `print(...); return None`
   and string raises with proper exceptions.
4. Opening a file that does not exist for reading raises
   `FileNotFoundError`, instead of creating or returning an empty object.
5. Each check gets a test; the probe becomes a regression test.

The input-validation auditor (tester 9) and the input fuzzers (4a–c) then
check what remains.

**Result (validation PR, 2026-10-01).** The fix above was done as its own
pull request, before wave 1. Re-running the same 60 probes:

| Outcome | Before | After |
|---|---|---|
| Refused with a clear `GRANDlib:` message | 7 | 50 (2 of them `OSError` for a non-ROOT file) |
| Error by accident, or crash from deep inside | 25 | 0 |
| Accepted with a `GRANDlibWarning` | 0 | 5 (NaN position, twice; zenith 500°; negative energy; negative time bin) |
| Accepted silently, by design | — | 5 (a new file is created by opening a tree on it; an empty `DataDirectory` for writing; zero antennas; a negative energy proxy floored at 0, as documented, twice) |
| `exit()` | 1 | 0 |

Physical ranges in tree fields warn rather than raise: reading a file goes
through the same setters, and a 2024 shower file stores `xmax_grams = -201`.
The probe is kept as `tests/test_input_validation.py`.

### Wave 1: using it (11 agents)

| # | Tester | Mission | Looks for |
|---|---|---|---|
| 1a | **Beginner**, setup and notebooks 01–06 (×2) | Fresh clone; follow only the README and docs: create the environment, run `env/setup.sh`, run notebooks 01→06 in order. Uses no knowledge beyond the docs. | Missing steps, unstated prerequisites, unclear errors, notebooks that depend on earlier state, unexplained jargon |
| 1b | **Beginner**, notebooks 07–12 and examples | As 1a, for notebooks 07→12 and `examples/`. | As 1a |
| 2a | **Expert analyst**, `dataio` | Every tree type, file and directory reader, metadata, writing and rereading. Checks docstrings and type/unit claims against behaviour. | Wrong documentation, surprising behaviour, missing features, inconsistent units or conventions |
| 2b | **Expert analyst**, `aoi`, `analysis`, event viewer | `EventList`/`Event`, reconstruction (`grand.analysis`, Cramér-Rao bounds), `TRecons`, the event viewer. | As 2a |
| 3a | **Pipeline user**, ZHAireS and `sim2root` | ZHAireS → `.rawroot` → `sim2root.py`, every option at least once. | Options that fail or combine badly, outputs that do not feed the next step, silently wrong values |
| 3b | **Pipeline user**, CoREAS and the conversion chain | CoREAS → `.rawroot`; `convert_efield2voltage.py` → `convert_voltage2adc.py` → `convert_efield2efield.py`; `RunSimPipe*.py`. | As 3a |
| 4a | **Input fuzzer**, readers | `dataio` readers, `EventList`, `DataFile`/`DataDirectory` given bad input: missing, empty, truncated or corrupt files; a non-ROOT file; zero events or antennas; paths with spaces, Unicode or `~`. | Brittleness. Each outcome is classified as **clear error**, **crash with traceback**, **hang**, or **silent wrong output** |
| 4b | **Input fuzzer**, conversion scripts (×2) | The same, for every `scripts/convert_*.py` and their options; read-only and existing output directories. | As 4a |
| 4c | **Input fuzzer**, utility scripts and `sim2root` input | The other scripts in `scripts/`, and malformed ZHAireS/CoREAS input to the converters. | As 4a |

### Wave 2: checking it (8 agents)

| # | Tester | Mission | Looks for |
|---|---|---|---|
| 5a | **Physics checker**, geometry | Independently recompute coordinate transforms, the "comes from" angle convention, Xmax frames, `direction`, topography. Round-trip and invariance checks. | Numerical and convention errors that tests written alongside the code can share |
| 5b | **Physics checker**, signal chain | Antenna response, RF chain gain, galactic-noise RMS, ADC conversion, reconstruction and the Cramér-Rao bounds. | As 5a |
| 6a | **Documentation auditor**, `docs/source` and the Handbook | Run every snippet; check every name, signature, option and path mentioned exists. | Docs that drifted from the code |
| 6b | **Documentation auditor**, READMEs and docstrings | As 6a; list public functions with no documentation. | Drift and omissions |
| 7 | **Error handling and API consistency** | `print`/`exit()` inside the library, bare `except`, swallowed errors, `None` returned where an error is expected (or the reverse), inconsistent argument names across similar functions. | Code that fails quietly or kills the caller's program |
| 8 | **Test-suite auditor** | Modules with no tests; tests that cannot fail; skips and expected failures whose reason no longer holds. | Gaps where bugs can hide |
| 9 | **Input validation auditor** | Follows up the validation review (§4.1): every public function's handling of type, sign, range, shape and units. | Missing or inconsistent validation |
| 10 | **Second documentation pass** | Notebooks' prose and outputs against the code they run. | Explanations that no longer match results |

### Wave 3: breaking it (5 agents)

Breakers attack everything other than the front door. A finding counts only
if a plausible user could hit it, and the report says how.

| # | Breaker | Attack | Examples |
|---|---|---|---|
| 11 | **Misuse** | Calls the API the wrong way | Methods in the wrong order; one object reused across files; writing after closing; the same tree opened twice; events that do not exist; trees from different runs mixed; the global `grand_tree_list` |
| 12 | **Numerical edges** | Extreme but legal physics values | Zenith 0° and 90°; azimuth 0/360 and negative; NaN/inf through a whole chain; one-antenna events; float32 overflow; zero-length traces; unusual sampling rates |
| 13 | **Scale and stress** | Size and repetition | Thousands of events or files; very long traces; memory growth over long loops; several processes writing to one output directory |
| 14 | **Environment** | The world around the code | Another working directory; unset environment variables (`GRAND_ROOT`…); missing optional packages (`iminuit`, `psutil`); read-only `data/`; a partial or corrupt data model; old file formats with missing branches |
| 15 | **Unsafe input** | Input that could do harm | File names with quotes, `;` or `$(…)` given to scripts that build shell commands (`RunSimPipe` runs steps through the shell); archive extraction without path checks (`download_data_grand.py`, `extractall` without a filter); `eval`-like parsing |

### Wave 4: regression (about 4 agents)

After the fixes from waves 1–3 are merged, the categories that found the most
issues run again on the fixed `dev-next`, to confirm the fixes and catch
anything they broke.

## 5. How a finding becomes a fix

```
tester ──report──▶ coordinator ──confirm──▶ GitHub issue ──fix──▶ PR into dev-next ──merge──▶ issue closed
                       │                        "dev-next_beta-test: …"
                       └── not reproducible / duplicate / not a bug → logged below, no issue
```

1. **Report.** The tester reports each finding with: severity, a minimal
   reproduction (exact commands or code), expected and actual behaviour,
   and `file:line` where known.
2. **Confirm.** The coordinator (Claude, the main session) reproduces it on
   `dev-next`. Findings that do not reproduce, are duplicates, or are not
   bugs are recorded in [§8](#8-findings-not-logged) with the reason, and no
   issue is opened.
3. **Log.** Each confirmed finding becomes one GitHub issue on
   `grand-mother/grand`, titled **`dev-next_beta-test: <short description>`**
   so the set can be found with a single search. The issue body follows
   [the template](#6-issue-template), and **includes the fix** whenever one
   is clear: the proposed patch, or a description of it.
4. **Fix.** Fixes go into `dev-next` through pull requests, batched by area,
   each with a test that fails before the fix and passes after, and with no
   new failures against the baseline (§3). Fixes that need a physics or
   design decision go to the code's owner instead, and the issue says so.
5. **Close.** When the PR merges, the issue is closed with a link to it, and
   the tracker below is updated.

### Severity

| Level | Meaning |
|---|---|
| **Critical** | Wrong physics results, data loss, or corrupted output, with no warning |
| **High** | A documented feature that crashes or cannot be used; a beginner cannot get past a step |
| **Medium** | Brittle on plausible input; misleading error or documentation; works with a workaround |
| **Low** | Cosmetic, wording, minor inconsistency |

## 6. Issue template

```
Title: dev-next_beta-test: <what goes wrong, in a few words>

**Found by:** beta tester <n> (<role>), <date>
**Severity:** <Critical | High | Medium | Low>
**Area:** <module or script>

**Reproduce** (on dev-next <commit>)
<commands or code>

**Expected**
**Actual**
<exact error or output>

**Cause** <file:line, and why>

**Fix** <the patch, or what it should be; or "needs a decision from <owner>: <question>">
```

## 7. Tracker

All issues: [search `dev-next_beta-test:`](https://github.com/grand-mother/grand/issues?q=is%3Aissue+%22dev-next_beta-test%3A%22).

| Issue | Title | Severity | Found by | Status | Fixed in | How |
|---|---|---|---|---|---|---|
| [#180](https://github.com/grand-mother/grand/issues/180) | `convert_voltage2adc.py` takes a directory, but its help says it takes a file | Medium | coordinator (pre-wave) | open | | |
| [#181](https://github.com/grand-mother/grand/issues/181) | CoREAS converter appends to an existing output and fails with `NotUniqueEvent` | Medium | coordinator (pre-wave) | open | | |
| [#182](https://github.com/grand-mother/grand/issues/182) | `-od/--out_directory` fails when the folder does not exist yet | Medium | coordinator (pre-wave) | open | | |
| [#183](https://github.com/grand-mother/grand/issues/183) | `T1_trigger_offline.py` has no argument parsing; `-h` is opened as a file | Low | coordinator (pre-wave) | open | | |
| [#184](https://github.com/grand-mother/grand/issues/184) | `open_grand_file.py` / `open_grand_directory.py` / `open_grand_analysis_prompt.py` run the file name as Python code | High | coordinator (pre-wave) | open | | |
| [#185](https://github.com/grand-mother/grand/issues/185) | README quickstart fails: `Efield2Voltage` needs a directory, not an efield file | High | 1a-A, 1a-B | open | | |
| [#186](https://github.com/grand-mother/grand/issues/186) | Docs say the declination at Dunhuang is "a few degrees"; it is about 0.3° (and IGRF-13 is outdated after 2020) | Medium | 1a-A, 1a-B | open | | |
| [#187](https://github.com/grand-mother/grand/issues/187) | A file whose name level disagrees with its trees is silently ignored | Medium | 1a-A, 1a-B | open | | |
| [#188](https://github.com/grand-mother/grand/issues/188) | Notebook 06 fixture triggers an unexplained Xmax warning; the warning's angles are ambiguous | Medium | 1a-A, 1a-B | open | | |
| [#189](https://github.com/grand-mother/grand/issues/189) | Notebooks 03, 04 and 06: prose contradicts the outputs | Medium | 1a-A | open | | |
| [#190](https://github.com/grand-mother/grand/issues/190) | Notebook 05 places 1 MHz-spaced noise into 0.977 MHz FFT bins | Medium | 1a-A | open | | |
| [#191](https://github.com/grand-mother/grand/issues/191) | `help()` on a tree field fails: descriptors crash on class access | Low | 1a-B | open | | |
| [#192](https://github.com/grand-mother/grand/issues/192) | Docs lack basic recipes (read one event's shower; geodetic to GRANDCS) | Medium | 1a-B | open | | |
| [#193](https://github.com/grand-mother/grand/issues/193) | Stale or garbled documentation text | Low | 1a-A, 1a-B | open | | |
| [#194](https://github.com/grand-mother/grand/issues/194) | Routine operations print alarming messages | Low | 1a-A, 1a-B | open | | |
| [#195](https://github.com/grand-mother/grand/issues/195) | `DataDirectory` silently drops files: keeps 1 of 10 showers in examples/analysis | High | 2a | open | | |
| [#196](https://github.com/grand-mother/grand/issues/196) | `get_list_of_events()` and `draw()` silently change the loaded entry's values | High | 2a | open | | |
| [#197](https://github.com/grand-mother/grand/issues/197) | `write()` to an existing file replaces its tree even with `overwrite=False` | High | 2a | open | | |
| [#198](https://github.com/grand-mother/grand/issues/198) | `write("other.root")` on a tree that already has a file writes a corrupt copy | High | 2a | open | | |
| [#199](https://github.com/grand-mother/grand/issues/199) | `get_dus_indices_in_run` returns run order, not event order | High | 2a | open | | |
| [#200](https://github.com/grand-mother/grand/issues/200) | `get_traces_lengths` always returns None; `get_list_of_dus` returns the whole tree's units | Medium | 2a | open | | |
| [#201](https://github.com/grand-mother/grand/issues/201) | Vector fields: unsigned char read as characters; `+=` replaces; numpy bool/uint64 assignment fails and empties the field | Medium | 2a | open | | |
| [#202](https://github.com/grand-mother/grand/issues/202) | A misspelt tree field is accepted silently: the typo guard never runs | Medium | 2a | open | | |
| [#203](https://github.com/grand-mother/grand/issues/203) | `creation_datetime` is stored in local time, not UTC | Medium | 2a | open | | |
| [#204](https://github.com/grand-mother/grand/issues/204) | `DataDirectory`: `recursive=True` finds nothing; one oddly named file aborts the directory | Medium | 2a | open | | |
| [#205](https://github.com/grand-mother/grand/issues/205) | Tree wildcards: no match creates a file named with '*'; wildcard chains cannot look up events | Medium | 2a | open | | |
| [#206](https://github.com/grand-mother/grand/issues/206) | Smaller dataio inconsistencies (argument order, silent failed lookups, TRecons units) | Low | 2a | open | | |
| [#207](https://github.com/grand-mother/grand/issues/207) | Regression from #179: numbers given as text refused; ZHAireS one-argument form and `sim2root.py -la/-lo/-al` crash | High | 3a | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | Text that reads as a number converted again, with a warning; callers pass numbers |
| [#209](https://github.com/grand-mother/grand/issues/209) | CoREAS converter mirrors the azimuth on the .inp path (180 − PHIP) | Critical | 2b, 3b | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | azimuth (PHIP − 180) mod 360; committed sample regenerated |
| [#210](https://github.com/grand-mother/grand/issues/210) | Notebook 07 passes an ENU direction to `topography.distance`, which needs ECEF | High | 1b | open | | |
| [#211](https://github.com/grand-mother/grand/issues/211) | Committed `recons_CR_candidates.root` stores raw χ² (notebook 11 calls it reduced); CRB fields 0.0 | High | 2b | open | | |
| [#212](https://github.com/grand-mother/grand/issues/212) | `Event.write`: `overwrite=True` deletes the whole output folder; in-place write crashes | High | 2b | open | | |
| [#213](https://github.com/grand-mother/grand/issues/213) | `EventList` ignores `start_event`, `start_entry`, per-call `tefield_level`; one Event object reused | High | 2b, 4a | open | | |
| [#214](https://github.com/grand-mother/grand/issues/214) | `sin_geomag_angle` returns one number for arrays | Medium | 2b | open | | |
| [#215](https://github.com/grand-mother/grand/issues/215) | Antenna positions from GPS use a hard-coded origin (question for GP80 owners) | Medium | 2b | open | | |
| [#216](https://github.com/grand-mother/grand/issues/216) | Reconstruction edge cases and docstrings | Low | 2b | open | | |
| [#217](https://github.com/grand-mother/grand/issues/217) | Notebook 08's "bit for bit" claim fails here; its control row reads "caught" | Medium | 1b | open | | |
| [#218](https://github.com/grand-mother/grand/issues/218) | examples/: broken, stale or silently misbehaving examples | Medium | 1b | open | | |
| [#219](https://github.com/grand-mother/grand/issues/219) | Notebooks 10–12: miscounts, unstated units, developer jargon | Low | 1b | open | | |
| [#220](https://github.com/grand-mother/grand/issues/220) | sim2root writes wrong `du_geoid` for several events per file; empty with `-ss` | Critical | 3a | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | geoid from unique antennas; `-ss` field name, lengths, first event |
| [#221](https://github.com/grand-mother/grand/issues/221) | README pipeline commands fail: `RunSimPipe.py` and the sim2root example lack `-sl` | High | 3a, 3b | open | | |
| [#222](https://github.com/grand-mother/grand/issues/222) | sim2root: mixed trace windows share one run's t_pre/t_post; window options unchecked | High | 3a | open | | |
| [#223](https://github.com/grand-mother/grand/issues/223) | sim2root: `-ef` bugs; different run numbers silently merged | Medium | 3a | open | | |
| [#224](https://github.com/grand-mother/grand/issues/224) | sim2root and ZHAireS converter: failures leave junk files and often exit 0 | Medium | 3a | open | | |
| [#225](https://github.com/grand-mother/grand/issues/225) | ZHAireS conversion stores sentinels for unknown Xmax and a magic default time | Medium | 3a | open | | |
| [#226](https://github.com/grand-mother/grand/issues/226) | sim2root minor: IllustrateSimPipe `--savefig`; naming and site cosmetics | Low | 3a | open | | |
| [#227](https://github.com/grand-mother/grand/issues/227) | `--rf_chain_nut` / `--rf_chain_gaa` have no effect with `--no_noise --no_rf_chain` | Critical | 3b | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | Nut/GAA chains brought back to the time domain |
| [#228](https://github.com/grand-mother/grand/issues/228) | CoREAS Xmax NaN or ~1 cm: efield2voltage crashes or outputs ~1e-13 µV | Critical | 3b | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | unknown Xmax written as NaN; `Efield2Voltage` refuses an event with no usable Xmax |
| [#229](https://github.com/grand-mother/grand/issues/229) | efield2voltage resampling keeps old `trigger_position` and `t_bin_size`; voltage2adc then uses the wrong rate | Critical | 3b, 4b-B | open (reopened) | [#208](https://github.com/grand-mother/grand/pull/208) | CLI refuses resampling; the Python API (`resample_to_mhz`) still resamples without updating `t_bin_size` (tester 12) |
| [#230](https://github.com/grand-mother/grand/issues/230) | `--seed` does not cover calibration smearing; jitter without a seed crashes; seed 0 unseeded | High | 3b | open | | |
| [#231](https://github.com/grand-mother/grand/issues/231) | Conversion scripts: `-od` writes run files into the input; reruns crash; L0/L1 picked silently | Medium | 3b | open | | |
| [#232](https://github.com/grand-mother/grand/issues/232) | CoREAS converter: magnetic field in three units; run/event swapped; README names | Low | 3b | open | | |
| [#233](https://github.com/grand-mother/grand/issues/233) | efield2efield bare AssertionErrors; plots always drawn; T1 passes no DU on clean sims | Low | 3b | open | | |
| [#234](https://github.com/grand-mother/grand/issues/234) | `Event.close_files()` rewrites input files and can hang; EventList scripts crash at exit | High | 4a | open | | |
| [#235](https://github.com/grand-mother/grand/issues/235) | EventList and DataFile crash deep, with unclear errors, on plausible input | High | 4a | open | | |
| [#236](https://github.com/grand-mother/grand/issues/236) | dataio returns stale or zeroed data: reopened files, missing branches, unrecognised names, use after close | Medium | 4a | open | | |
| [#237](https://github.com/grand-mother/grand/issues/237) | efield2voltage pairs an L0 e-field with an L1 run tree: 2× amplitude, no warning | Critical | 4b-B | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | run tree read at the efield's level |
| [#238](https://github.com/grand-mother/grand/issues/238) | `compute_voltage` for a missing event writes another event with an empty trace | Critical | 4b-B | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | `get_event` checks the pair against the input's events |
| [#239](https://github.com/grand-mother/grand/issues/239) | NaN/inf voltages become INT64_MIN in the ADC; saturation never reported | High | 4b-B | open | | |
| [#240](https://github.com/grand-mother/grand/issues/240) | Re-running conversions: voltage2adc deletes old output then crashes; failed runs leave blocking stubs | High | 4b-B | open | | |
| [#241](https://github.com/grand-mother/grand/issues/241) | Multi-run folders crash; measured noise reused across antennas silently; raw noise errors | Medium | 4b-B | open | | |
| [#242](https://github.com/grand-mother/grand/issues/242) | ZHAireS converter silently accepts damaged simulations (missing antennas, padded/NaN traces, zenith 0, default core) | Critical | 4c | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | traces, antennas, angles, energy, core checked before writing; output removed on failure. Not covered: damaged `.t*` tables (3e) |
| [#243](https://github.com/grand-mother/grand/issues/243) | CoREAS converter: antenna list and trace files not cross-checked; NaN and ragged traces accepted | Critical | 4c | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | list and traces checked before writing |
| [#244](https://github.com/grand-mother/grand/issues/244) | `extract_events.py`: `-ow` deletes everything in the target (even "."); duplicates leave a broken target | High | 4c | open | | |
| [#245](https://github.com/grand-mother/grand/issues/245) | `pipeline/get_files_from_db.py` moves small files while listing | High | 4c | open | | |
| [#246](https://github.com/grand-mother/grand/issues/246) | Utility scripts: no argparse, wrong option lists, files in cwd, version 0.0.0 | Low | 4c | open | | |
| [#247](https://github.com/grand-mother/grand/issues/247) | efield2voltage / efield2efield use the previous event's shower when the shower tree lacks the event | Critical | 4b-A | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | shower and run entries checked after lookup |
| [#248](https://github.com/grand-mother/grand/issues/248) | `convert_efield2efield` writes `du_count` 0, ignores the `-o` folder, crashes on zero-antenna events | High | 4b-A | open | | |
| [#249](https://github.com/grand-mother/grand/issues/249) | Conversion scripts don't check that run, e-field and shower trees match | Medium | 4b-A | open | | |
| [#250](https://github.com/grand-mother/grand/issues/250) | The geoid model (EGM96) is mirrored in latitude: geoid heights tens of metres off | Critical | 5a, 5b | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | `data/egm96.png` rows flipped, grid extents corrected; checked at the poles and the global extremes |
| [#251](https://github.com/grand-mother/grand/issues/251) | Geodetic height NaN west of Greenwich (GEOID); NaN declination there; Horizon objects share location | High | 5a, 5b | open | | |
| [#252](https://github.com/grand-mother/grand/issues/252) | Reconstruction hard-codes ground altitude 1231 m; file antenna z is relative to the site | Medium | 5a | open | | |
| [#253](https://github.com/grand-mother/grand/issues/253) | Antenna effective-length lookup 1° off for every negative azimuth | Medium | 5b | open | | |
| [#254](https://github.com/grand-mother/grand/issues/254) | Physics details: ADF asymmetry B unnormalised, circular convolution at default padding, ADC truncation, refractive index | Medium | 5b | open | | |
| [#255](https://github.com/grand-mother/grand/issues/255) | rf_chain error path crashes (NameError 'Nonec'); duplicate definitions; prints instead of errors | High | 7 | open | | |
| [#256](https://github.com/grand-mother/grand/issues/256) | Silent failures: Newton non-convergence, raise of a string, prints instead of errors in aoi, global warning filter | High | 7 | open | | |
| [#257](https://github.com/grand-mother/grand/issues/257) | Docs give commands that fail (CoREAS, sim2root, Efield2Voltage file input) | High | 6a | open | | |
| [#258](https://github.com/grand-mother/grand/issues/258) | Docs state stale or wrong facts | Medium | 6a | open | | |
| [#259](https://github.com/grand-mother/grand/issues/259) | About 25 user-facing input checks are asserts | Medium | 7 | open | | |
| [#260](https://github.com/grand-mother/grand/issues/260) | READMEs are stale | Medium | 6b | open | | |
| [#261](https://github.com/grand-mother/grand/issues/261) | Docstrings that mislead: missing units and frames, wrong parameters and returns | Medium | 6b | open | | |
| [#262](https://github.com/grand-mother/grand/issues/262) | A NaN or None latitude/longitude segfaults the process (geoid_undulation, Map.elevation) | High | 9 | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | NaN points masked before libturtle (Map, Stack, global and local elevation); NaN out with a warning |
| [#263](https://github.com/grand-mother/grand/issues/263) | `convert_voltage_to_ADC` converts every channel for a boolean mask; slice or int raises (regression from #179) | High | 9 | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | channels indexed as NumPy does; mask length checked |
| [#264](https://github.com/grand-mother/grand/issues/264) | `Handling3dTraces` accepts a NumPy-scalar `f_samp_mhz`, then `apply_bandpass` fails | High | 9 | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | scalar or per-unit rate in any numeric form; length, NaN and bool checked |
| [#265](https://github.com/grand-mother/grand/issues/265) | `Efield2Voltage.params` ignores unknown keys; flags read by truthiness | Medium | 9 | open | | |
| [#266](https://github.com/grand-mother/grand/issues/266) | Values in the wrong unit (Hz/MHz, s/ns, degrees/radians) accepted silently | Medium | 9 | open | | |
| [#267](https://github.com/grand-mother/grand/issues/267) | Modules the validation work did not reach (geo reps, turtle, aoi, ShowerEvent, trigger, basis.signal, du_network) | Medium | 9 | open | | |
| [#268](https://github.com/grand-mother/grand/issues/268) | Notebook 05 sky maps put right ascension 12 h out | High | 10 | fixed | [#208](https://github.com/grand-mother/grand/pull/208) | maps shifted by 12 h, sources marked; noise-table convention still to confirm with owners |
| [#269](https://github.com/grand-mother/grand/issues/269) | Notebook prose, second pass: 14 statements the outputs contradict | Medium | 10 | open | | |
| [#270](https://github.com/grand-mother/grand/issues/270) | Deliberate bugs no test catches (Horizontal azimuth, `get_dus_indices_in_run`, `final_resample`, ADC rounding) | High | 8 | open | | |
| [#271](https://github.com/grand-mother/grand/issues/271) | Tests that cannot fail, non-strict xfails, tests depending on untracked `data/` files | Medium | 8 | open | | |
| [#273](https://github.com/grand-mother/grand/issues/273) | Two tree objects on one file share branch buffers: wrong event numbers written, reads mixed | High | 11 | open | | |
| [#274](https://github.com/grand-mother/grand/issues/274) | Segfault when a tree is used after close_file(), or after another tree closed its file | High | 11 | open | | |
| [#275](https://github.com/grand-mother/grand/issues/275) | Entries filled but not written are discarded silently when a with-block ends or stop_using() is called | High | 11 | open | | |
| [#276](https://github.com/grand-mother/grand/issues/276) | Tree get_entry/get_entry_with_index refuse NumPy integers | Medium | 11 | open | | |
| [#277](https://github.com/grand-mother/grand/issues/277) | Event and Efield2Voltage used in the wrong order or with bad indices give bare errors | Medium | 11 | open | | |
| [#278](https://github.com/grand-mother/grand/issues/278) | A non-editable install or wheel lacks vector_filling.C and rf_chain_config.xml | High | 14 | open | | |
| [#279](https://github.com/grand-mother/grand/issues/279) | Data model integrity: damaged files accepted, unhelpful errors, "up to date" with noise/ missing; undeclared psutil | Medium | 14 | open | | |
| [#280](https://github.com/grand-mother/grand/issues/280) | Environment rough edges: output dirs, missing libraries and optional packages, notebook paths | Low | 14 | open | | |
| [#281](https://github.com/grand-mother/grand/issues/281) | Several processes writing to one output file lose events or corrupt it, while some report success | Critical | 13 | open | | |
| [#282](https://github.com/grand-mother/grand/issues/282) | copy_contents() then fill() empties the source tree's traces | Medium | 13 | open | | |
| [#283](https://github.com/grand-mother/grand/issues/283) | Scaling: per-event appends slow down with file size; DataDirectory superlinear in files | High | 13 | open | | |
| [#284](https://github.com/grand-mother/grand/issues/284) | Memory: trees not freed without stop_using(); residual leak; Efield2Voltage memory far above data size | Medium | 13 | open | | |
| [#285](https://github.com/grand-mother/grand/issues/285) | Antenna response read from the wrong table row below the antenna horizon (θ ≥ 91° wraps to 0°) | Critical | 12 | open | | |
| [#286](https://github.com/grand-mother/grand/issues/286) | recons_ADF returns its starting point when an amplitude is 0 or negative; can run for minutes | High | 12 | open | | |
| [#287](https://github.com/grand-mother/grand/issues/287) | Handling3dTraces crashes on one-antenna events | High | 12 | open | | |
| [#288](https://github.com/grand-mother/grand/issues/288) | Numerical edges: NaN and extreme values pass silently; peak amplitude biased; PWF at zenith 0 | Medium | 12 | open | | |
| [#289](https://github.com/grand-mother/grand/issues/289) | Numerical edges, minor: position guards, ignored sigma, trigger shapes, float32 overflow, ADC helpers | Low | 12 | open | | |

Status values: **open**, **fix in PR**, **fixed** (merged), **with owner** (needs a decision), **won't fix** (with reason).

## 7a. Wave 1 results (2026-10-01)

All 11 wave-1 testers (Claude Sonnet, each in its own copy of the repository) ran on
`dev-next` at `91d30a1b`. The coordinator reproduced every finding marked
"confirmed" in its issue before logging it; findings taken on the tester's word
are marked as such in the issue. Overlapping reports went into one issue, with
the extra detail added as comments.

### By severity

| Severity | Count | Examples |
|---|---|---|
| Critical (wrong physics or data, silently) | 10 | #209 CoREAS converter mirrors the azimuth (older input files); #220 sim2root writes wrong antenna lat/lon (up to ~8 km) for several events per file; #227 `--rf_chain_nut`/`--rf_chain_gaa` ignored without noise or chain; #228 CoREAS events with no usable Xmax give near-zero voltages or crash; #229 resampling in the voltage step leaves the ADC step at the wrong rate; #237 an L0 e-field paired with an L1 run tree doubles amplitudes; #238 a missing event writes another event with an empty trace; #242, #243 ZHAireS and CoREAS converters accept damaged simulations; #247 a missing shower entry reuses the previous event's shower |
| High | 22 | data loss in `write()`, `Event.write(overwrite=True)`, `extract_events -ow` and re-runs; documented commands that crash (README quickstart, `RunSimPipe.py`); `DataDirectory` dropping files |
| Medium | 26 | brittle input handling; misleading errors and documentation; notebooks whose prose contradicts their output |
| Low | 11 | cosmetic; wording; minor inconsistencies |
| **Total** | **69** | #180–#249 (#208 is a PR) |

### By tester

| Tester | Role and area | Issues (first reporter or co-reporter) | Severity |
|---|---|---|---|
| 1a-A | Beginner: setup, notebooks 01–06 | #185 #186 #187 #188 #189 #190 #193 #194 | 1 High, 5 Medium, 2 Low |
| 1a-B | Beginner: setup, notebooks 01–06, three small tasks from the docs | #185 #186 #187 #188 #191 #192 #193 #194 | 1 High, 4 Medium, 3 Low |
| 1b | Beginner: notebooks 07–12, examples/ | #210 #217 #218 #219 | 1 High, 2 Medium, 1 Low |
| 2a | Expert: `grand.dataio` | #195 #196 #197 #198 #199 #200 #201 #202 #203 #204 #205 #206 | 5 High, 6 Medium, 1 Low |
| 2b | Expert: `grand.aoi`, `grand.analysis`, event viewer | #209 #211 #212 #213 #214 #215 #216 | 1 Critical, 3 High, 2 Medium, 1 Low |
| 3a | Pipeline: ZHAireS → `.rawroot` → sim2root | #207 #220 #221 #222 #223 #224 #225 #226 | 1 Critical, 3 High, 3 Medium, 1 Low |
| 3b | Pipeline: CoREAS, conversion scripts | #209 #221 #227 #228 #229 #230 #231 #232 #233 | 4 Critical, 2 High, 1 Medium, 2 Low |
| 4a | Input fuzzer: readers | #213 #234 #235 #236 | 3 High, 1 Medium |
| 4b-A | Input fuzzer: conversion scripts | #247 #248 #249 | 1 Critical, 1 High, 1 Medium |
| 4b-B | Input fuzzer: conversion scripts and simulation classes | #229 #237 #238 #239 #240 #241 | 3 Critical, 2 High, 1 Medium |
| 4c | Input fuzzer: utility scripts, malformed simulation input | #242 #243 #244 #245 #246 | 2 Critical, 2 High, 1 Low |
| coordinator (pre-wave) | Coordinator, while double-checking the validation PR (#179) | #180 #181 #182 #183 #184 | 1 High, 3 Medium, 1 Low |

An issue appears under every tester who reported it. The two pairs that ran the
same mission independently (1a-A/1a-B and 4b-A/4b-B) agreed on their main
findings but each also found problems the other missed, so coverage of those
areas is not yet saturated (§4): the input-fuzzing areas get another pass in
wave 3.

### Notes

- **One regression from this recovery work:** #207 was introduced by the
  validation PR #179 (numbers given as text refused). It is fixed in PR #208.
- **Tester environment:** testers' copies started on the default branch
  (`0cc99804`); every tester switched to `91d30a1b` before testing, and the
  coordinator checked each copy's commit. One tester's example run downloaded two
  terrain tiles into the shared `data/topography/`; they were removed (#218, item 2).
- **Proposed order from here:** fix the 10 Critical issues first, in a few PRs
  grouped by area (CoREAS converter, sim2root, voltage chain, dataio), with
  wave 2 (physics and documentation checks) running in parallel; wave 3 (breakers)
  after the Critical fixes merge; wave 4 (regression) at the end.

## 7b. Open issues (2026-10-01, after PR #272)

98 open issues: the 93 `dev-next_beta-test:` issues below (none closed by PR #272, whose
security fixes were reported privately) and 5 issues kept with their owners
(#104, #139, #140, #141, #142; feature requests and simulation-field questions).

### By area and severity

| Area | Critical | High | Medium | Low | Total |
|---|---|---|---|---|---|
| dataio | 1 | 13 | 11 | 3 | 28 |
| Conversion scripts, voltage chain | 1 | 4 | 8 | 2 | 15 |
| sim2root, ZHAireS, CoREAS |  | 2 | 3 | 2 | 7 |
| Physics, geo, analysis | 1 | 5 | 7 | 2 | 15 |
| Docs, notebooks, examples |  | 4 | 11 | 2 | 17 |
| Tools, install, tests |  | 5 | 4 | 2 | 11 |
| **Total** | **3** | **33** | **44** | **13** | **93** |

### By wave

| Found in | Critical | High | Medium | Low | Total |
|---|---|---|---|---|---|
| Pre-wave (coordinator) |  | 1 | 3 | 1 | 5 |
| Wave 1 | 1 | 20 | 23 | 10 | 54 |
| Wave 2 |  | 5 | 12 |  | 17 |
| Wave 3 | 2 | 7 | 6 | 2 | 17 |

### All open issues

| Severity | Issue | Area | Wave | Title |
|---|---|---|---|---|
| Critical | [#229](https://github.com/grand-mother/grand/issues/229) | Conversion scripts, voltage chain | 1 | efield2voltage resampling keeps old `trigger_position` and `t_bin_size`; voltage2adc then uses the wrong rate |
| Critical | [#281](https://github.com/grand-mother/grand/issues/281) | dataio | 3 | Several processes writing to one output file lose events or corrupt it, while some report success |
| Critical | [#285](https://github.com/grand-mother/grand/issues/285) | Physics, geo, analysis | 3 | Antenna response read from the wrong table row below the antenna horizon (θ ≥ 91° wraps to 0°) |
| High | [#184](https://github.com/grand-mother/grand/issues/184) | Tools, install, tests | pre | `open_grand_file.py` / `open_grand_directory.py` / `open_grand_analysis_prompt.py` run the file name as Python code |
| High | [#185](https://github.com/grand-mother/grand/issues/185) | Docs, notebooks, examples | 1 | README quickstart fails: `Efield2Voltage` needs a directory, not an efield file |
| High | [#195](https://github.com/grand-mother/grand/issues/195) | dataio | 1 | `DataDirectory` silently drops files: keeps 1 of 10 showers in examples/analysis |
| High | [#196](https://github.com/grand-mother/grand/issues/196) | dataio | 1 | `get_list_of_events()` and `draw()` silently change the loaded entry's values |
| High | [#197](https://github.com/grand-mother/grand/issues/197) | dataio | 1 | `write()` to an existing file replaces its tree even with `overwrite=False` |
| High | [#198](https://github.com/grand-mother/grand/issues/198) | dataio | 1 | `write("other.root")` on a tree that already has a file writes a corrupt copy |
| High | [#199](https://github.com/grand-mother/grand/issues/199) | dataio | 1 | `get_dus_indices_in_run` returns run order, not event order |
| High | [#210](https://github.com/grand-mother/grand/issues/210) | Docs, notebooks, examples | 1 | Notebook 07 passes an ENU direction to `topography.distance`, which needs ECEF |
| High | [#211](https://github.com/grand-mother/grand/issues/211) | Docs, notebooks, examples | 1 | Committed `recons_CR_candidates.root` stores raw χ² (notebook 11 calls it reduced); CRB fields 0.0 |
| High | [#212](https://github.com/grand-mother/grand/issues/212) | dataio | 1 | `Event.write`: `overwrite=True` deletes the whole output folder; in-place write crashes |
| High | [#213](https://github.com/grand-mother/grand/issues/213) | dataio | 1 | `EventList` ignores `start_event`, `start_entry`, per-call `tefield_level`; one Event object reused |
| High | [#221](https://github.com/grand-mother/grand/issues/221) | sim2root, ZHAireS, CoREAS | 1 | README pipeline commands fail: `RunSimPipe.py` and the sim2root example lack `-sl` |
| High | [#222](https://github.com/grand-mother/grand/issues/222) | sim2root, ZHAireS, CoREAS | 1 | sim2root: mixed trace windows share one run's t_pre/t_post; window options unchecked |
| High | [#230](https://github.com/grand-mother/grand/issues/230) | Conversion scripts, voltage chain | 1 | `--seed` does not cover calibration smearing; jitter without a seed crashes; seed 0 unseeded |
| High | [#234](https://github.com/grand-mother/grand/issues/234) | dataio | 1 | `Event.close_files()` rewrites input files and can hang; EventList scripts crash at exit |
| High | [#235](https://github.com/grand-mother/grand/issues/235) | dataio | 1 | EventList and DataFile crash deep, with unclear errors, on plausible input |
| High | [#239](https://github.com/grand-mother/grand/issues/239) | Conversion scripts, voltage chain | 1 | NaN/inf voltages become INT64_MIN in the ADC; saturation never reported |
| High | [#240](https://github.com/grand-mother/grand/issues/240) | Conversion scripts, voltage chain | 1 | Re-running conversions: voltage2adc deletes old output then crashes; failed runs leave blocking stubs |
| High | [#244](https://github.com/grand-mother/grand/issues/244) | Tools, install, tests | 1 | `extract_events.py`: `-ow` deletes everything in the target (even "."); duplicates leave a broken target |
| High | [#245](https://github.com/grand-mother/grand/issues/245) | Tools, install, tests | 1 | `pipeline/get_files_from_db.py` moves small files while listing |
| High | [#248](https://github.com/grand-mother/grand/issues/248) | Conversion scripts, voltage chain | 1 | `convert_efield2efield` writes `du_count` 0, ignores the `-o` folder, crashes on zero-antenna events |
| High | [#251](https://github.com/grand-mother/grand/issues/251) | Physics, geo, analysis | 2 | Geodetic height NaN west of Greenwich (GEOID); NaN declination there; Horizon objects share location |
| High | [#255](https://github.com/grand-mother/grand/issues/255) | Physics, geo, analysis | 2 | rf_chain error path crashes (NameError 'Nonec'); duplicate definitions; prints instead of errors |
| High | [#256](https://github.com/grand-mother/grand/issues/256) | Physics, geo, analysis | 2 | Silent failures: Newton non-convergence, raise of a string, prints instead of errors in aoi, global warning filter |
| High | [#257](https://github.com/grand-mother/grand/issues/257) | Docs, notebooks, examples | 2 | Docs give commands that fail (CoREAS, sim2root, Efield2Voltage file input) |
| High | [#270](https://github.com/grand-mother/grand/issues/270) | Tools, install, tests | 2 | Deliberate bugs no test catches (Horizontal azimuth, `get_dus_indices_in_run`, `final_resample`, ADC rounding) |
| High | [#273](https://github.com/grand-mother/grand/issues/273) | dataio | 3 | Two tree objects on one file share branch buffers: wrong event numbers written, reads mixed |
| High | [#274](https://github.com/grand-mother/grand/issues/274) | dataio | 3 | Segfault when a tree is used after close_file(), or after another tree closed its file |
| High | [#275](https://github.com/grand-mother/grand/issues/275) | dataio | 3 | Entries filled but not written are discarded silently when a with-block ends or stop_using() is called |
| High | [#278](https://github.com/grand-mother/grand/issues/278) | Tools, install, tests | 3 | A non-editable install or wheel lacks vector_filling.C and rf_chain_config.xml |
| High | [#283](https://github.com/grand-mother/grand/issues/283) | dataio | 3 | Scaling: per-event appends slow down with file size; DataDirectory superlinear in files |
| High | [#286](https://github.com/grand-mother/grand/issues/286) | Physics, geo, analysis | 3 | recons_ADF returns its starting point when an amplitude is 0 or negative; can run for minutes |
| High | [#287](https://github.com/grand-mother/grand/issues/287) | Physics, geo, analysis | 3 | Handling3dTraces crashes on one-antenna events |
| Medium | [#180](https://github.com/grand-mother/grand/issues/180) | Conversion scripts, voltage chain | pre | `convert_voltage2adc.py` takes a directory, but its help says it takes a file |
| Medium | [#181](https://github.com/grand-mother/grand/issues/181) | Conversion scripts, voltage chain | pre | CoREAS converter appends to an existing output and fails with `NotUniqueEvent` |
| Medium | [#182](https://github.com/grand-mother/grand/issues/182) | Conversion scripts, voltage chain | pre | `-od/--out_directory` fails when the folder does not exist yet |
| Medium | [#186](https://github.com/grand-mother/grand/issues/186) | Docs, notebooks, examples | 1 | Docs say the declination at Dunhuang is "a few degrees"; it is about 0.3° (and IGRF-13 is outdated after 2020) |
| Medium | [#187](https://github.com/grand-mother/grand/issues/187) | dataio | 1 | A file whose name level disagrees with its trees is silently ignored |
| Medium | [#188](https://github.com/grand-mother/grand/issues/188) | Docs, notebooks, examples | 1 | Notebook 06 fixture triggers an unexplained Xmax warning; the warning's angles are ambiguous |
| Medium | [#189](https://github.com/grand-mother/grand/issues/189) | Docs, notebooks, examples | 1 | Notebooks 03, 04 and 06: prose contradicts the outputs |
| Medium | [#190](https://github.com/grand-mother/grand/issues/190) | Docs, notebooks, examples | 1 | Notebook 05 places 1 MHz-spaced noise into 0.977 MHz FFT bins |
| Medium | [#192](https://github.com/grand-mother/grand/issues/192) | Docs, notebooks, examples | 1 | Docs lack basic recipes (read one event's shower; geodetic to GRANDCS) |
| Medium | [#200](https://github.com/grand-mother/grand/issues/200) | dataio | 1 | `get_traces_lengths` always returns None; `get_list_of_dus` returns the whole tree's units |
| Medium | [#201](https://github.com/grand-mother/grand/issues/201) | dataio | 1 | Vector fields: unsigned char read as characters; `+=` replaces; numpy bool/uint64 assignment fails and empties the field |
| Medium | [#202](https://github.com/grand-mother/grand/issues/202) | dataio | 1 | A misspelt tree field is accepted silently: the typo guard never runs |
| Medium | [#203](https://github.com/grand-mother/grand/issues/203) | dataio | 1 | `creation_datetime` is stored in local time, not UTC |
| Medium | [#204](https://github.com/grand-mother/grand/issues/204) | dataio | 1 | `DataDirectory`: `recursive=True` finds nothing; one oddly named file aborts the directory |
| Medium | [#205](https://github.com/grand-mother/grand/issues/205) | dataio | 1 | Tree wildcards: no match creates a file named with '*'; wildcard chains cannot look up events |
| Medium | [#214](https://github.com/grand-mother/grand/issues/214) | Physics, geo, analysis | 1 | `sin_geomag_angle` returns one number for arrays |
| Medium | [#215](https://github.com/grand-mother/grand/issues/215) | Physics, geo, analysis | 1 | Antenna positions from GPS use a hard-coded origin (question for GP80 owners) |
| Medium | [#217](https://github.com/grand-mother/grand/issues/217) | Docs, notebooks, examples | 1 | Notebook 08's "bit for bit" claim fails here; its control row reads "caught" |
| Medium | [#218](https://github.com/grand-mother/grand/issues/218) | Docs, notebooks, examples | 1 | examples/: broken, stale or silently misbehaving examples |
| Medium | [#223](https://github.com/grand-mother/grand/issues/223) | sim2root, ZHAireS, CoREAS | 1 | sim2root: `-ef` bugs; different run numbers silently merged |
| Medium | [#224](https://github.com/grand-mother/grand/issues/224) | sim2root, ZHAireS, CoREAS | 1 | sim2root and ZHAireS converter: failures leave junk files and often exit 0 |
| Medium | [#225](https://github.com/grand-mother/grand/issues/225) | sim2root, ZHAireS, CoREAS | 1 | ZHAireS conversion stores sentinels for unknown Xmax and a magic default time |
| Medium | [#231](https://github.com/grand-mother/grand/issues/231) | Conversion scripts, voltage chain | 1 | Conversion scripts: `-od` writes run files into the input; reruns crash; L0/L1 picked silently |
| Medium | [#236](https://github.com/grand-mother/grand/issues/236) | dataio | 1 | dataio returns stale or zeroed data: reopened files, missing branches, unrecognised names, use after close |
| Medium | [#241](https://github.com/grand-mother/grand/issues/241) | Conversion scripts, voltage chain | 1 | Multi-run folders crash; measured noise reused across antennas silently; raw noise errors |
| Medium | [#249](https://github.com/grand-mother/grand/issues/249) | Conversion scripts, voltage chain | 1 | Conversion scripts don't check that run, e-field and shower trees match |
| Medium | [#252](https://github.com/grand-mother/grand/issues/252) | Physics, geo, analysis | 2 | Reconstruction hard-codes ground altitude 1231 m; file antenna z is relative to the site |
| Medium | [#253](https://github.com/grand-mother/grand/issues/253) | Physics, geo, analysis | 2 | Antenna effective-length lookup 1° off for every negative azimuth |
| Medium | [#254](https://github.com/grand-mother/grand/issues/254) | Physics, geo, analysis | 2 | Physics details: ADF asymmetry B unnormalised, circular convolution at default padding, ADC truncation, refractive index |
| Medium | [#258](https://github.com/grand-mother/grand/issues/258) | Docs, notebooks, examples | 2 | Docs state stale or wrong facts |
| Medium | [#259](https://github.com/grand-mother/grand/issues/259) | Tools, install, tests | 2 | About 25 user-facing input checks are asserts |
| Medium | [#260](https://github.com/grand-mother/grand/issues/260) | Docs, notebooks, examples | 2 | READMEs are stale |
| Medium | [#261](https://github.com/grand-mother/grand/issues/261) | Docs, notebooks, examples | 2 | Docstrings that mislead: missing units and frames, wrong parameters and returns |
| Medium | [#265](https://github.com/grand-mother/grand/issues/265) | Conversion scripts, voltage chain | 2 | `Efield2Voltage.params` ignores unknown keys; flags read by truthiness |
| Medium | [#266](https://github.com/grand-mother/grand/issues/266) | Physics, geo, analysis | 2 | Values in the wrong unit (Hz/MHz, s/ns, degrees/radians) accepted silently |
| Medium | [#267](https://github.com/grand-mother/grand/issues/267) | Tools, install, tests | 2 | Modules the validation work did not reach (geo reps, turtle, aoi, ShowerEvent, trigger, basis.signal, du_network) |
| Medium | [#269](https://github.com/grand-mother/grand/issues/269) | Docs, notebooks, examples | 2 | Notebook prose, second pass: 14 statements the outputs contradict |
| Medium | [#271](https://github.com/grand-mother/grand/issues/271) | Tools, install, tests | 2 | Tests that cannot fail, non-strict xfails, tests depending on untracked `data/` files |
| Medium | [#276](https://github.com/grand-mother/grand/issues/276) | dataio | 3 | Tree get_entry/get_entry_with_index refuse NumPy integers |
| Medium | [#277](https://github.com/grand-mother/grand/issues/277) | Conversion scripts, voltage chain | 3 | Event and Efield2Voltage used in the wrong order or with bad indices give bare errors |
| Medium | [#279](https://github.com/grand-mother/grand/issues/279) | Tools, install, tests | 3 | Data model integrity: damaged files accepted, unhelpful errors, "up to date" with noise/ missing; undeclared psutil |
| Medium | [#282](https://github.com/grand-mother/grand/issues/282) | dataio | 3 | copy_contents() then fill() empties the source tree's traces |
| Medium | [#284](https://github.com/grand-mother/grand/issues/284) | dataio | 3 | Memory: trees not freed without stop_using(); residual leak; Efield2Voltage memory far above data size |
| Medium | [#288](https://github.com/grand-mother/grand/issues/288) | Physics, geo, analysis | 3 | Numerical edges: NaN and extreme values pass silently; peak amplitude biased; PWF at zenith 0 |
| Low | [#183](https://github.com/grand-mother/grand/issues/183) | Conversion scripts, voltage chain | pre | `T1_trigger_offline.py` has no argument parsing; `-h` is opened as a file |
| Low | [#191](https://github.com/grand-mother/grand/issues/191) | dataio | 1 | `help()` on a tree field fails: descriptors crash on class access |
| Low | [#193](https://github.com/grand-mother/grand/issues/193) | Docs, notebooks, examples | 1 | Stale or garbled documentation text |
| Low | [#194](https://github.com/grand-mother/grand/issues/194) | dataio | 1 | Routine operations print alarming messages |
| Low | [#206](https://github.com/grand-mother/grand/issues/206) | dataio | 1 | Smaller dataio inconsistencies (argument order, silent failed lookups, TRecons units) |
| Low | [#216](https://github.com/grand-mother/grand/issues/216) | Physics, geo, analysis | 1 | Reconstruction edge cases and docstrings |
| Low | [#219](https://github.com/grand-mother/grand/issues/219) | Docs, notebooks, examples | 1 | Notebooks 10–12: miscounts, unstated units, developer jargon |
| Low | [#226](https://github.com/grand-mother/grand/issues/226) | sim2root, ZHAireS, CoREAS | 1 | sim2root minor: IllustrateSimPipe `--savefig`; naming and site cosmetics |
| Low | [#232](https://github.com/grand-mother/grand/issues/232) | sim2root, ZHAireS, CoREAS | 1 | CoREAS converter: magnetic field in three units; run/event swapped; README names |
| Low | [#233](https://github.com/grand-mother/grand/issues/233) | Conversion scripts, voltage chain | 1 | efield2efield bare AssertionErrors; plots always drawn; T1 passes no DU on clean sims |
| Low | [#246](https://github.com/grand-mother/grand/issues/246) | Tools, install, tests | 1 | Utility scripts: no argparse, wrong option lists, files in cwd, version 0.0.0 |
| Low | [#280](https://github.com/grand-mother/grand/issues/280) | Tools, install, tests | 3 | Environment rough edges: output dirs, missing libraries and optional packages, notebook paths |
| Low | [#289](https://github.com/grand-mother/grand/issues/289) | Physics, geo, analysis | 3 | Numerical edges, minor: position guards, ignored sigma, trigger shapes, float32 overflow, ADC helpers |

**Next:** the 3 Critical (#229 Python-API resampling, #281 concurrent writers,
#285 antenna response below the horizon), one commit and one test each, in one PR;
then the High issues by area, starting with `dataio` (data loss and wrong values).

## 8. Findings not logged

Reports that were not confirmed, with the reason.

| Tester | Report | Why not logged |
|---|---|---|
| 9 | `Efield2Voltage.get_event(event_number=99, run_number=0)` silently reuses the previous event | Duplicate of #238, already fixed in PR #208 (the pair is now checked against `events_list`) |
| 9 | Further instances of bare `assert` input checks | Added to #259 as a comment |

## 9. Progress log

| Date | Event |
|---|---|
| 2026-10-01 | PR #272 merged into `dev-next` (7e5c753d): fixes for the privately reported security findings (archive extraction, external commands, production pipeline, granddb remote access), with tests. Open-issue summary added (§7b): 98 open, 3 Critical, 33 High, 44 Medium, 13 Low, 5 with owners. |
| 2026-10-01 | **Wave 3 complete.** Testers 12 (numerical edges) and 13 (scale and stress) in: logged #281–#289 (2 Critical: #281 concurrent writers lose events, #285 antenna response wrapped below the horizon; 3 High, 3 Medium, 1 Low); #229 reopened (Python API still resamples); #223 extended. Wave 3 total: #273–#289 (2 Critical, 7 High, 6 Medium, 2 Low) plus tester 15's security findings, fixed in PR #272. No regression of the #208 fixes. |
| 2026-10-01 | Wave 3: testers 11 (misuse) and 14 (environment) in, logged #273–#280 (4 High, 3 Medium, 1 Low). Security fixes for tester 15 in PR #272. Still running: 12, 13. |
| 2026-10-01 | Tester 15 (unsafe input) in: security findings in three areas (archive handling, simulation scripts, production pipeline), confirmed by the coordinator. Reported privately as GitHub security advisories, not as public issues; they will be summarised here once fixed. |
| 2026-10-01 | **Wave 3 started** on `dev-next` at `89622823` (PR #208 merged): breakers 11 (misuse), 12 (numerical edges), 13 (scale and stress, capped at 3 GB disk and ~4 GB RAM), 14 (environment, on private copies of the data model), 15 (unsafe input, harmless marker-file proofs only). |
| 2026-10-01 | PR #208 merged into `dev-next` (8962282). Its 16 issues closed: all 11 Critical (#209, #220, #227, #228, #229, #237, #238, #242, #243, #247, #250) and 5 High (#207, #262, #263, #264, #268). Follow-ups recorded on #229 (TVoltage sampling rate), #242 (damaged `.t*` tables), #250 (productions with geoid heights), #268 (noise-table RA convention). Next: wave 3. |
| 2026-10-01 | Critical fixes in PR #208, continued: #209, #228, #243 (CoREAS), #242 (ZHAireS), #250 (geoid). All 10 wave-1 Criticals and #250 now have a fix in PR #208. **Wave 2 complete:** testers 8, 9 and 10 in, logged #262–#271 (5 High, 5 Medium; #263 is a regression from #179). Tester 8: 15 deliberate bugs, 10 caught. |
| 2026-10-01 | Critical fixes in PR #208 (one commit each, with a failing-first test): #227, #238, #247, #237, #229, #220. Wave 2 batch 1 in (5a, 5b, 6a, 6b, 7): logged #250–#261, one more Critical (#250 geoid mirrored, confirmed at the poles and Tokyo). Wave 2 batch 2 running: 8, 9, 10. |
| 2026-10-01 | **Wave 1 complete.** 4b-A in: logged #247–#249 (1 more Critical). Totals: 69 issues (#180–#249; #208 is a PR): 10 Critical, 22 High, 26 Medium, 11 Low. The two ×2 pairs (1a-A/1a-B, 4b-A/4b-B) overlapped heavily but each found things the other missed, so the input-fuzzing area gets another pass in wave 3. |
| 2026-10-01 | 4c in: logged #242–#246 (2 more Critical: converters accept damaged simulations); #184 extended to `open_grand_analysis_prompt.py` and raised to High. Still running: 4b-A. |
| 2026-10-01 | 4b-B in: logged #237–#241 (2 more Critical: #237 L0/L1 mix doubles amplitudes, #238 missing event writes the wrong one); #229 raised to Critical. Still running: 4b-A, 4c. |
| 2026-10-01 | Reports in from 3a, 1b, 2b, 4a, 3b: confirmed and logged as #207–#236 (4 Critical: #209 CoREAS azimuth mirrored, #220 sim2root `du_geoid`, #227 nut/GAA chains ignored, #228 CoREAS Xmax → zero voltage). #207 is a regression from #179 (validation), fixed in PR #208. Batch 3 running: 4b ×2, 4c. |
| 2026-10-01 | Batch 1 reports in from 1a-A, 1a-B and 2a: all findings confirmed by the coordinator (re-run where marked) and logged as #185–#206; five High (silent data loss or wrong values in `grand.dataio`). Batch 2 started: 1b, 2b, 3b. |
| 2026-10-01 | Validation merged (PR #179). Wave 1 started on `dev-next` at `91d30a1b`: batch 1 (1a ×2, 2a, 3a, 4a). Five problems found by the coordinator during the pre-PR check logged as #180–#184. |
| 2026-10-01 | Validation work double-checked before its PR, against `dev-next`: full pipeline (sim2root, the three converters, T1, CoREAS) gives identical output (676 branches in 21 files); the 12 notebooks give the same results; reading and writing speed unchanged. Found and fixed 3 checks that refused input which used to work: a covariance matrix as `sigma`, `channels` as a slice, a `(1, 3)` source position (and `[[x, y, z]]` for 3-value tree fields). |
| 2026-10-01 | Validation PR: 50 of 60 bad inputs now refused with a `GRANDlib:` message, 0 crashes, 0 `exit()` (§4.1). |
| 2026-10-01 | Input validation review: 18 % of public functions check input; 27 of 60 bad inputs accepted silently (§4.1). |
| 2026-10-01 | Data model downloaded into the test environment; full suite 727 passed, 10 skipped, 11 xfailed, 0 failed. Plan written. |
