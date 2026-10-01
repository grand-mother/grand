# GRANDlib status — GRAND Weekly Meeting, 1 October 2026

`2026-10-01-Bustamante-GRAND_Weekly_Meeting.odp` (and its `.pdf`): 19 slides
on the `dev-next` recovery: the cleanup, with the recovery plan's own figures;
input validation (PR #179); the beta test in three waves (issues #180–#271,
fixes in PR #208); the one request (access to the SPS data for the aoi code);
and a timeline.

## Rebuilding

    python make_figures_grandlib.py      # severity, validation and timeline plots
    DECK_TEMPLATE=<carrier.odp> python deck_grandlib.py

| File | |
|---|---|
| `deck_grandlib.py` | the content: one call per slide, speaker notes inline |
| `make_figures_grandlib.py` | the three plots, in the style of `make_figures.py` |
| `build_deck.py`, `make_figures.py` | the deck machinery and figure helpers of the GNN Board talk of 28 September 2026 |
| `figures/` | the plots; `recovery/` holds the plan's figures, rendered from `resources/dev/dev-next/*.svg` with `rsvg-convert -z 2.5` |

**One change to `build_deck.py`.** `write_content()` now drops the carrier's
`x…` automatic styles. A carrier that was itself built by this script (the GNN
Board deck, used here) carries styles named `xT24` and so on; left in, they
came first and won over the new deck's styles of the same name, which gave
mixed font sizes and overlapping text.

The carrier (master page, background, logos) is not in the repository; set
`DECK_TEMPLATE` to `2026-09-28-Bustamante-GNN_Board.odp` or to the
3 September weekly-meeting copy. Text widths are measured with Palatino
Linotype, or TeX Gyre Pagella where it is missing; without either, the
builder guesses and text overlaps.

Numbers are from `resources/dev/dev-next/RECOVERY_PLAN.md`, `BRANCHES.md` and
`beta-tests/TEST_PLAN.md` on `dev-next-ipfxhh`, 1 October 2026. The PDF was
rendered with LibreOffice 24.2 and TeX Gyre Pagella standing in for Palatino.
