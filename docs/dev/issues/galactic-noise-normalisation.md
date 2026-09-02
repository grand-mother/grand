<!-- Ready to post as a GitHub issue on grand-mother/grand. Not yet posted.
     Title:  Galactic-noise normalisation does not match the tabulated model
     Labels: bug, physics, blocking -->

## Summary

The RMS of the simulated Galactic noise does not equal the value Parseval's
theorem gives for the table it is built from. The open PR that changes this
constant (#153) improves the agreement but does not achieve it, so the choice
between `size_out/2` and `size_out/sqrt(2)` is not the whole question.

## Measurement

Made 2026-08-30 against `data/noise/Vocmax_30-250MHz_uVperMHz_hfss.npy` at
LST 18 h, 600 antennas, 221 bins of 1 MHz embedded in a 2048-point transform.
Reproduce with `python tests/sim/test_galactic_noise_normalisation.py`.

| Normalisation | simulated / model |
|---|---|
| `size_out / 2` (current `dev`) | **0.33** |
| `size_out / sqrt(2)` (#153) | **0.47** |

Neither is 1. A factor of roughly 2 is unaccounted for.

## What is not in doubt

The implementation is internally consistent: the time series obtained from the
returned spectrum carries that spectrum's energy to one part in 10^9. The
transform is right; only its scaling is in question.

## The question that settles it

**Is `Vocmax_30-250MHz_uVperMHz` an RMS voltage spectral density, or a
maximum?**

The filename says *max*. If it is a peak quantity, the reference used above is
wrong by a known factor and the comparison shifts — possibly onto one of the
two candidates. If it is an RMS, neither candidate is correct and the
normalisation chain needs a closer look.

This is a question for the authors of the table rather than something that can
be settled by reading the code.

## A second, separate discrepancy

Section 8.2 of [arXiv:2408.10926](https://arxiv.org/abs/2408.10926) states that
the module randomises the **phase** of the sky-averaged noise. The code also
randomises the modulus, drawing from a normal distribution and taking the
absolute value. The two agree in mean power and differ in their fluctuations,
so this may not be a defect — but the published description matches neither
candidate implementation, and one of the two should be corrected.

## Why it matters

A factor of 1.41 in noise amplitude is not cosmetic in an experiment whose
sensitivity is set by a trigger threshold. Every trigger study, sensitivity
estimate and Data Challenge output downstream of this depends on it.

Until it is resolved, results that depend on the noise level should record
which version of `grand/sim/noise/galaxy.py` produced them; `TRun.software_version`
exists for that.

## Blocks

- #153 (`dev_snonis`) — the normalisation change itself
- #146 (`refact_galaxy`) — the parallel rewrite; both cannot land

## Tests

`tests/sim/test_galactic_noise_normalisation.py` asserts what holds under any
convention (internal Parseval consistency, seed reproducibility, no silently
zero antenna arm) and records the contested comparison as a skipped test
carrying the measurement, so the number is visible without presuming the
target.
