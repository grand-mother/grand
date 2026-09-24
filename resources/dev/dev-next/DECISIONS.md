# Three branches waiting on the collaboration

`dev_marion`, `grandio_light` and `snonis_sim2root_test_merge` can't be settled
by review. Each one asks the collaboration a question, and merging the branch
would answer it by default. This page gives each question the evidence it
needs.

Everything below was measured on 2026-09-24 against `dev-next` at `9605981b`,
on full git history. Where code is said to fail, it was run.

---

## 1. `dev_marion` — does reconstruction live in GRANDlib?

> **Decided 2026-09-24: yes, in `grand/analysis/`; merged.** The branch was
> merged as it stood, so the history is Marion Guelfand's, and each
> prerequisite below followed as its own commit: `TRecons` pinned in the
> schema snapshot; `iminuit` declared (conda environment, and an optional
> `analysis` extra in `pyproject.toml`); the example data moved to
> `examples/analysis/`; round-trip tests added before any cleanup; the code
> brought within the lint gate; and the empty `grand/recon/` removed. Two
> bugs found on the way were fixed: the Cherenkov solver used `np.infty`,
> which NumPy 2 removed, and `grand/analysis/coords/` would not have been in
> a built package. Not yet tested: whether the fits agree with simulated or
> measured showers.

**What it is.** Marion Guelfand's reconstruction package, January–February
2026, seven commits. It adds `grand/analysis/` (38 files):

- plane-wave, spherical-wave and ADF fits of arrival direction and Xmax;
- an electromagnetic-energy estimate from voltages;
- signal extraction, footprint geometry and a Cherenkov-angle model;
- worked examples on GP13 data.

**What merging it would also do.**

- **Add a new part of the data format.** `TRecons`, a ROOT tree of 27 fields
  (fit results, per-antenna peaks, angles, energy). Once files are written with
  it, the fields are a contract. Until 2026-09-24 the schema test would not
  have noticed; it now fails on any new tree, and names `TRecons`.
- **Add a dependency nobody has declared.** Every module in `grand/analysis/`
  imports `iminuit`, which is in neither `env/conda/grand-dev.yml` nor
  `pyproject.toml`. Merged onto `dev-next`, none of them can be imported in the
  project's own environment. `import grand` itself still works.
- **Ship 0.9 MB of example data inside the Python package**
  (`grand/analysis/example/*.root`), so every install carries it.
- **Add no tests** for the new code.

It merges without a single conflict, and the existing data-layer tests still
pass (317) with it merged.

**The question for the collaboration.** Does event reconstruction belong in
GRANDlib, beside simulation and I/O, or in its own package that depends on
GRANDlib? If it belongs here, three things are needed before merging:
approve `TRecons` as a format, declare `iminuit`, and move the example data
out of the package. The repository already has an empty `grand/recon/` (two
stub files), which suggests someone once planned for it to live here.

---

## 2. `grandio_light` — should GRANDlib ship an I/O-only version?

> **Decided 2026-09-24: lazy imports, branch not merged.** `grand/__init__.py`
> now loads its public names on first use. Measured afterwards,
> `import grand.dataio` loads only the data layer and works without the
> compiled core, and every public name still resolves; the full suite gives
> identical results before and after. Pinned by `tests/test_lazy_imports.py`.
> Whether to also publish a separate `grandlib-io` package is still open.

**What it is.** luckyjim's proposal, December 2024 – August 2025, for a
"light" GRANDlib that can read and write GRAND's ROOT files without the
physics. The branch does this by **deleting** the physics from the
repository: it keeps `grand/basis/`, `grand/dataio/` and logging, and
removes `geo`, `sim`, `aoi`, `recon` and the database code. Merged as it
stands, it would remove most of GRANDlib. Against `dev-next` it conflicts
in 45 files.

**Why someone wants it.** It's a real problem, measured: on `dev-next`,
`import grand.dataio` loads **13 `grand.sim` modules, 6 `grand.geo`
modules and the compiled C core**, and so fails without TURTLE and GULL
built. That's because `grand/__init__.py` imports the physics eagerly. It's
how this session's first attempt to run the converter failed
(`No module named 'grand._core'`).

**What the measurement shows.** The I/O code doesn't need the physics. In
`grand/dataio/` and `grand/basis/` there is exactly one import of another
part of GRANDlib (`basis/type_trace.py` uses `geo.coordinates`), plus one
lazy import that only runs when used. So an I/O-only GRANDlib doesn't need a
separate branch or deleted code. It needs `grand/__init__.py` to stop
importing everything eagerly, and possibly a second, smaller package
definition built from the same tree.

**The question for the collaboration.** Is an I/O-only install wanted? If
yes, make the imports lazy on `dev-next` (small, reversible, and it fixes
the import failure above for everyone), then decide whether to publish a
separate `grandlib-io`. The branch can then be closed without being merged.
If no, close it.

---

## 3. `snonis_sim2root_test_merge` — which way do GRAND's angles point?

> **Decided 2026-09-24: "comes from" stays; branch not merged.** The
> convention is now stated in the coordinates documentation and pinned by
> `tests/geo/test_angle_convention.py`, which checks the core transform
> against both committed ZHAireS summaries. With the branch's lines applied,
> all three of its tests fail.

**What it is.** snonis's branch, January–April 2024. Half of it,
propagating `du_type` through `Efield2Voltage`, is already on the trunk.
The other half changes the angle convention in six core functions of
`grand/geo/coordinates.py`: zenith becomes 180° − zenith and azimuth becomes
azimuth + 180°. Every angle GRANDlib computes would then describe the
direction the shower **travels** rather than the direction it **comes from**.

**What the data says.** The files GRANDlib reads already use "comes from".
For event 1618, today's converter puts Xmax at (−4050.6, 3988.9, 4499.4) m
from the core, and the file stores zenith 51.64°, azimuth 135.44°. Running
each version of `_cartesian_to_spherical` on that point:

| Version | θ | φ | On arrays |
|---|---|---|---|
| `dev-next` | **51.64°** | **135.44°** | works |
| this branch | 128.36° | 315.44° | raises `ValueError` |

`dev-next` reproduces the stored angles exactly. The branch returns the
opposite direction, and it can't process more than one value at a time,
which is how it's normally called: `if phi==360` is ambiguous on an array.

**The question for the collaboration.** Is there a reason to switch GRAND to
"travels towards"? Both conventions are legitimate. But switching would
mean rewriting the stored angles in every existing file, or converting at
every read, and updating the paper's conventions (Appendix A, now in the
coordinates documentation). Without such a reason, keep "comes from" and
close the branch. Whatever snonis needed the flip for is better done where
it's needed, with a named function such as `propagation_direction()`, than
by changing the core transforms.

---

## For all three

None of these should be merged as a routine code change: each would decide
a question by accident. Recording the answers in `docs/dev/branch_facts.py`
under `DECIDED` updates `BRANCHES.md` and the diagrams the next time the
generators run.
