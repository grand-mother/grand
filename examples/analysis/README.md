# Analysis examples

- `main_AOI.py`, `main_DOI.py`: reconstruct events (plane wave, spherical
  wave, ADF) and write a `TRecons` file. They need the GP80 data and the
  antenna positions named in `config.py`.
- `display.py`: plots the ADF fit of each event in a `TRecons` file.

## `recons_CR_candidates.root`

Ten GP13 cosmic-ray candidates, reconstructed with an earlier version of
`main_AOI.py`; which version is not recorded. As `TRecons` documents:

- `chi2_pwf`, `chi2_swf` and `chi2_adf` are the **raw** χ². Divide by the
  degrees of freedom (`du_count - 2` for the plane wave, `du_count - 4` for
  the spherical wave and the ADF) for the reduced χ².
- The file predates the `crb_*` (Cramér-Rao bound) fields, so it has no
  bounds. `TRecons` reads them as NaN; before #211 they read 0.0, which
  looked like "no uncertainty".

See grand-mother/grand#211.
