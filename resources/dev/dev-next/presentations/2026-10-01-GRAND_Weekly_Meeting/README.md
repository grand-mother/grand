# GRANDlib status — GRAND Weekly Meeting, 1 October 2026

`2026-10-01-Bustamante-GRAND_Weekly_Meeting.odp`: 14 slides on the `dev-next`
recovery: branches merged, old pull requests and issues cleaned up, input
validation (PR #179), the beta test (issues #180–#271, fixes in PR #208), and
what is needed from the collaboration.

## Rebuilding

    python make_figures_grandlib.py      # the two plots, into figures/
    DECK_TEMPLATE=<carrier.odp> python deck_grandlib.py

| File | |
|---|---|
| `deck_grandlib.py` | the content: one call per slide, speaker notes inline |
| `make_figures_grandlib.py` | the severity and input-validation plots |
| `build_deck.py`, `make_figures.py` | the deck machinery and figure helpers (faces, palette), unchanged from the GNN Board talk of 28 September 2026 |
| `figures/` | the rendered plots |

The carrier (master page, background, logos) is not in the repository. The
deck was built on `2026-09-28-Bustamante-GNN_Board.odp`; `build_deck.py`
otherwise looks for the 3 September weekly-meeting copy on the author's
laptop. Set `DECK_TEMPLATE` to point at one.

Numbers are from `resources/dev/dev-next/RECOVERY_PLAN.md`, `BRANCHES.md`
and `beta-tests/TEST_PLAN.md` on `dev-next-ipfxhh`, 1 October 2026, 11:40.
The `.odp` was not rendered before it was committed; check it in Impress.
