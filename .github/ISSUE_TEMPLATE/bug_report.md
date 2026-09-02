---
name: Bug report
about: Something behaves differently from what the documentation says
labels: bug
---

## What happened

<!-- Include the actual output. If the result was `nan` rather than an
     exception, say so -- that is this library's most common failure and it is
     easy to describe as "it didn't work". -->

## What you expected, and where that expectation came from

<!-- A docstring, a documentation page, the paper, the Handbook. If they
     disagree with each other, that is itself worth reporting. -->

## To reproduce

```python
# The shortest script that shows it.
```

## Environment

- GRANDlib revision (`git describe --tags --always --dirty`):
- Installed how (conda environment file, or something else):
- ROOT version (`python -c "import ROOT; print(ROOT.gROOT.GetVersion())"`):
- Operating system:

## Checked already

- [ ] It is not in `docs/source/known_issues.rst`
- [ ] It is not in the Troubleshooting page
- [ ] If it involves an absolute Galactic-noise level, I have stated the
      `du_type` — the three values differ by up to a factor of two
