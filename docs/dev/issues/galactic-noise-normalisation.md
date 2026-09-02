<!-- Ready to post as a GitHub issue on grand-mother/grand. Not yet posted.
     Title:  Galactic-noise normalisation: one definition decides between two constants
     Labels: bug, physics, blocking -->

## Summary

PR #153 changes the Galactic-noise normalisation from `size_out / 2` to
`size_out / sqrt(2)`, scaling every simulated noise voltage by 1.41. Which is
right turns on a single question about the shipped tables, and that question
has never been written down.

Measuring against the table each `du_type` actually reads settles everything
except that one definition.

## Measurement

Made 2026-09-02 at LST 18 h, 221 bins of 1 MHz embedded in a transform of
length `size_out`. Reproduce with
`python tests/sim/test_galactic_noise_normalisation.py`.

| `du_type` | simulated RMS / tabulated | vs 1/√2 = 0.7071 |
|---|---|---|
| `GP300` (default) | 0.7050 | −0.3 % |
| `GP300_nec` | 0.7049 | −0.3 % |
| `GP300_mat` | 0.7049 | −0.3 % |

The ratio is 1/√2 to a fraction of a percent. It does not move with
`size_out` (1024, 2048 and 4096 give the same value to five digits) or with
the number of antennas. So the implementation produces an RMS that is exactly
1/√2 of the tabulated value — the RMS-to-peak relation for a sinusoid.

## The question that settles it

**Is `Vocmax_30-250MHz_uVperMHz` an RMS voltage spectral density, or a
maximum?**

| If the table is a… | then the right constant is |
|---|---|
| **maximum** | `size_out / 2` — the current code, and PR #153 would break it |
| **RMS** | `size_out / sqrt(2)` — PR #153 is right |

The filename says *max*. Nothing else in the repository states which is meant.
This is a question for whoever produced the tables, not something the code can
answer.

## Correction to an earlier report

An earlier measurement in this repository gave 0.33 and concluded that neither
candidate constant reaches 1, with a factor of roughly 2 unaccounted for. That
comparison ran the default `GP300` simulation against
`Vocmax_..._hfss.npy` — a table `GP300` never reads. The missing factor was
the normalisation difference between those two tables, not a defect in the
transform. **Compared like with like there is no unexplained factor**, and the
`√2` framing is exactly right after all.

## What is not in doubt

The implementation is internally consistent: the time series obtained from the
returned spectrum carries that spectrum's energy to one part in 10⁹. The
transform is right; only its scaling is in question.

## Three separate problems found alongside

Filed here because they were found in the same investigation, but they are
independent of the constant above and could be split off.

1. **`Vocmax_..._nec.npy` and `Vocmax_..._mat.npy` are byte-identical**, as
   are the `Pocmax_...` and `Voutmax_...` pairs. The docstring offers
   `GP300_nec` and `GP300_mat` as the NEC and MATLAB variants; they select the
   same numbers. One file was presumably copied over the other, and which is
   the survivor is not recoverable from the repository.

2. **The default is not the model the docstring names.** `du_type='GP300'`
   reads no `Vocmax_*.npy` table; it recomputes `V_oc² = 4·P·R_ant` from
   `PG_ALL_jifen.mat`. The result is 0.463× the `hfss` table, flat across the
   band to 2.4 % — a normalisation difference, not a different model (the
   ratio to the `nec` table varies by 11 %, which is what a different model
   looks like). So `GP300` and `hfss` are one model at two normalisations
   differing by ≈2.16.

3. **The `*_hfss.npy` tables ship but no `du_type` opens them.**

Together these mean the three nominal `du_type` values resolve to two distinct
sets of numbers whose band-integrated levels differ by a factor of 1.6, and a
fourth, unreachable set 2.1× the default:

| Selector | Reads | Band RMS at LST 18 h (X, Y, Z) |
|---|---|---|
| `GP300` (default) | `PG_ALL_jifen.mat` | 27.8, 35.6, 31.4 µV |
| `GP300_nec` | `Vocmax_..._nec.npy` | 44.5, 55.6, 53.4 µV |
| `GP300_mat` | the same file as `nec` | 44.5, 55.6, 53.4 µV |
| *(unreachable)* | `Vocmax_..._hfss.npy` | 59.6, 75.5, 66.9 µV |

A noise level quoted without its `du_type` is ambiguous by a factor of two.

## A fourth, documentation-level discrepancy

Section 8.2 of [arXiv:2408.10926](https://arxiv.org/abs/2408.10926) says the
module randomises the *phase* of the sky-averaged noise. The code also
randomises the modulus, drawing from a normal distribution and taking the
absolute value. The two agree in mean power and differ in their fluctuations.
Not necessarily a defect, but the paper and the code should agree.

## Tests

`tests/sim/test_galactic_noise_normalisation.py` (14 tests, all passing)
asserts the 1/√2 relation for every `du_type`, its independence from
`size_out`, the internal Parseval consistency, and the three data-file facts
above. If someone applies PR #153 without settling the definitional question,
the suite says which constant changed.
