# API audit of dev-next against `dev`

Does dev-next still do what `dev` did, for code and files written for `dev`? @lwpiotr asked this after reviewing dev-next in October 2026. `dev` is the version frozen as `archive/dev-2026-09`; dev-next was audited at 86680cb0. The discussion and every finding are in [#293](https://github.com/grand-mother/grand/issues/293).

This folder is the working record of the audit, kept for reference. Discussion and decisions happen in the issues; an issue number such as #294 refers to `https://github.com/grand-mother/grand/issues/294`.

## How

1. **Lech's points**, each checked by running the old and the new code: #294 to #311.
2. **`grand.dataio` and `grand.aoi`**: every public function whose code changed, old and new side by side, with an example run on both versions: [`dataio_aoi.md`](dataio_aoi.md), #312 to #325.
3. **`grand.sim`, `grand.geo`, `grand.basis` and the command-line scripts**, the same way: [`sim_geo_basis_scripts.md`](sim_geo_basis_scripts.md), #326 to #332.
4. **`dev`'s own tests, example scripts and Lech's notebooks**, run on both versions: results in #310, and #333.

Each change that needs a decision has its own issue, with the example, the commands and the output of both versions, and how to fix it when that is known.

## Results

A function counts in every row that applies.

| | `grand.dataio` | `grand.aoi` | `grand.sim` | `grand.geo` | `grand.basis` | Scripts | Total |
|---|---|---|---|---|---|---|---|
| Changed functions or scripts | 73 | 26 | 72 | 32 | 21 | 22 | **246** |
| (1) Incompatible with `dev` | 48 | 18 | 52 | 17 | 9 | 16 | **160** |
| of which also fix a bug | 44 | 18 | 47 | 14 | 5 | 11 | **139** |
| (2) Improvements | 36 | 18 | 67 | 28 | 15 | 15 | **179** |
| (3) Non-critical fixes | 36 | 11 | 45 | 12 | 5 | 14 | **123** |
| Serious fixes | 21 | 14 | 8 | 6 | 1 | 5 | **55** |
| No visible change | 10 | 2 | 2 | 3 | 4 | 0 | **21** |

- **(1) Incompatible**: for input that worked in `dev`, the function now raises, returns something else or writes something else. 139 of these 160 also fix a bug, so most of the question is how each change should behave, not whether to undo it.
- **(2) Improvements**: new checks on input `dev` handled badly, units and limits, new options, clearer messages.
- **(3) Non-critical fixes**: wrong results or failures on unusual input.

**Bugs found in dev-next:** #312 (a crash), #326 (the pipeline scripts), #327 (two noise plots), #333 (a zenith out of range). #306 (ROOT 6.40) predates dev-next. #325 is a bug in both versions.

**What to decide:** the other issues of #293. A draft compatibility policy that would settle most of them is in #311.

## Files

- `dataio_aoi.md` and `sim_geo_basis_scripts.md`: one row per changed function or script, with its classes, what a caller sees and its issue.
- `examples/`: the scripts shown in the issues, each with the output of both versions (`.out`). The setup to run them is in #293.
