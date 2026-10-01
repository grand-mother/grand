# `dev-next` beta test: plan and tracker

**Status:** planned; validation review before wave 1 · **Last updated:** 2026-10-01 · **Branch under test:** `dev-next` at `b25d0521`

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
| — | *none yet* | | | | | |

Status values: **open**, **fix in PR**, **fixed** (merged), **with owner** (needs a decision), **won't fix** (with reason).

## 8. Findings not logged

Reports that were not confirmed, with the reason.

| Tester | Report | Why not logged |
|---|---|---|
| — | *none yet* | |

## 9. Progress log

| Date | Event |
|---|---|
| 2026-10-01 | Input validation review: 18 % of public functions check input; 27 of 60 bad inputs accepted silently (§4.1). |
| 2026-10-01 | Data model downloaded into the test environment; full suite 727 passed, 10 skipped, 11 xfailed, 0 failed. Plan written. |
