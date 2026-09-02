# Contributing to GRANDlib

The full guide is in the documentation, at
[Contributing](https://grand-mother.github.io/grand-docs/contributing.html)
(or `docs/source/contributing.rst` in this repository). This file is the short
version, and the parts you need before your first commit.

## Setting up

```bash
conda env create -f env/conda/grand-dev.yml --solver=libmamba
conda activate grand-dev
source env/setup.sh
pip install -e . --no-deps --no-build-isolation
```

`env/setup.sh` compiles the TURTLE and GULL C extensions and downloads about a
gigabyte of models, so it is neither optional nor fast the first time. It needs
`make`, which the environment file does not declare — install it from your
distribution if the build stops immediately.

## Before you open a pull request

Everything CI runs, you can run:

```bash
python -m pytest tests/ -q
ruff check grand/ tests/ quality/ notebooks/ docs/dev/
cd docs && make html
python notebooks/make_notebooks.py
```

The documentation must build with **zero warnings**. Do not add `-W` to that
build; see the CI page for why.

## Five things that are convention, not configuration

**The lint list is a ratchet.** `pyproject.toml` carries a `per-file-ignores`
table recording lint debt in code that had never been checked. It **may shrink
and must never grow**. If your change makes a listed file clean of a listed
rule, delete that rule in the same commit. New code is written clean.

**Docstrings are numpydoc, on every function**, private ones included, and
summaries are third person — "Returns the effective length", not "Return the
effective length". `D401` is disabled for that reason; leave it disabled. Do
not mix in the legacy `:param:` fields.

**Tests build their fixtures.** `data/` is gitignored, so a test that reads a
checked-in ROOT file cannot run in CI. Construct the input from the tree
classes instead — `tests/sim/test_pipeline_end_to_end.py` is the pattern.

**Seed every random draw, through a local generator** — `np.random.default_rng(0)`,
not `np.random.seed(0)`, so a test does not disturb state another test depends
on. An unseeded draw here failed about one run in six.

**The notebooks are generated.** Edit `notebooks/make_notebooks.py`, never the
`.ipynb`. So are the diagrams (`docs/dev/make_*_diagram.py`) and the handbook
pages (`docs/dev/build_handbook.py`).

## Merging

Work goes to `dev-next`, then `dev`, then `master`. A clean textual merge is
not a compatible merge — run the pre-merge check:

```bash
python quality/premerge_check.py <branch>
```

It catches two names for one quantity, and two implementations of one thing in
different files. The third failure mode, a change of meaning under an
unchanged name, only running the code detects.

## Reporting something

`docs/source/known_issues.rst` lists what is already known, with what was
measured and what would settle it. If your problem is not there, it is worth
reporting — and worth a test that reproduces it, because most entries on that
page began as one.
