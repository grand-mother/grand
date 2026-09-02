## What this changes

<!-- One or two sentences. If it fixes something, say what was wrong, not just
     which file you touched. -->

## Why

<!-- What made this necessary. If it corrects an earlier claim -- in a commit
     message, a docstring, the Handbook -- say so and give the number. -->

## Checks

- [ ] `python -m pytest tests/ -q` passes
- [ ] `ruff check grand/ tests/ quality/ notebooks/ docs/dev/` is clean
- [ ] `cd docs && make html` builds with no warnings
- [ ] `python notebooks/make_notebooks.py` succeeds, if a notebook or the
      package's public behaviour changed

## Conventions

- [ ] New functions carry numpydoc docstrings with `Parameters` and `Returns`
- [ ] No new entries added to the `per-file-ignores` ratchet in `pyproject.toml`
      (removing one is welcome)
- [ ] Any random draw in a test is seeded through a local generator
- [ ] Generated files were regenerated rather than hand-edited: notebooks via
      `notebooks/make_notebooks.py`, diagrams via `docs/dev/make_*_diagram.py`,
      handbook pages via `docs/dev/build_handbook.py`

## If this changes a physical result

<!-- State the measurement, not the impression: what the number was before,
     what it is now, and how you measured it. Results that depend on the
     Galactic-noise normalisation or on du_type should say which was used --
     see docs/source/known_issues.rst. -->
