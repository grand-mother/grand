# `dev-next` beta test: plan and tracker

**Status:** planned, not started · **Last updated:** 2026-10-01 · **Branch under test:** `dev-next` at `b25d0521`

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

Eight AI agents (Claude Sonnet), each playing one kind of user, run in two
waves so that the first wave's findings can steer the second. Each works in
its own isolated copy of the repository and **reports only**: testers do not
change code or post on GitHub.

### Wave 1: using it

| # | Tester | Mission | Looks for |
|---|---|---|---|
| 1 | **Beginner** | Fresh clone; follow only the README and docs: create the environment, run `env/setup.sh`, run notebooks 01→12 in order, then the "getting started" examples. Uses no knowledge beyond the docs. | Missing steps, unstated prerequisites, unclear errors, notebooks that depend on earlier state, jargon without explanation |
| 2 | **Expert analyst** | Realistic tasks through the API: read every tree type, iterate events (`EventList`), reconstruct directions and energy (`grand.analysis`), write and reread a `TRecons` file, use the event viewer. Checks docstrings and type/unit claims against behaviour. | Documentation that is wrong, surprising behaviour, missing features an expert expects, inconsistent units or conventions |
| 3 | **Simulation pipeline user** | The full chain on the samples: ZHAireS/CoREAS → `.rawroot` → `sim2root.py` → `convert_efield2voltage.py` → `convert_voltage2adc.py` → `convert_efield2efield.py`, and `RunSimPipe*.py`; every command-line option at least once. | Options that do not work or combine badly, outputs that do not feed the next step, silently wrong values, unhelpful messages |
| 4 | **Input fuzzer** | Every script and public reader given bad input: missing, empty, truncated or corrupt files; a non-ROOT file; zero events or antennas; NaN/inf; wrong types; paths with spaces, Unicode or `~`; relative paths; read-only and full output directories; existing outputs. | Brittleness. Each outcome is classified as **clear error**, **crash with traceback**, **hang**, or **silent wrong output** |

### Wave 2: checking it

| # | Tester | Mission | Looks for |
|---|---|---|---|
| 5 | **Physics checker** | Independently recompute the key results: coordinate transforms and the "comes from" angle convention, Xmax frames, galactic-noise RMS, antenna response, RF chain gain, ADC conversion, the Cramér-Rao bounds. Round-trip and invariance checks (transform and back, units, energy scaling). | Numerical and convention errors that tests written alongside the code can share |
| 6 | **Documentation auditor** | Run every code snippet in `docs/source/`, the Handbook, READMEs and docstrings; check that every name, signature, option and file path mentioned exists; list public functions with no documentation. | Docs that drifted from the code, and omissions |
| 7 | **Error handling and API consistency** | Look for `print`/`exit()` inside the library, bare `except`, swallowed errors, `None` returned where an error is expected (or the reverse), inconsistent argument names across similar functions. | Code that fails quietly, or kills the caller's program |
| 8 | **Test-suite auditor** | Map modules with no tests; find tests that cannot fail; check every skip and expected failure still has a valid reason. | Gaps where bugs can hide |

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
| 2026-10-01 | Data model downloaded into the test environment; full suite 727 passed, 10 skipped, 11 xfailed, 0 failed. Plan written. |
