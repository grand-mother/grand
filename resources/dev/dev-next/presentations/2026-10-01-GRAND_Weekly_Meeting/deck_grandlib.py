#!/usr/bin/env python
"""GRANDlib status: content. GRAND Weekly Meeting, 1 October 2026.

Same machinery as the GNN Board deck (build_deck.py); only the output name and
title differ. One text size per role, as in that deck: titles 28 pt, body 18 pt,
tables 15 pt, sources 11 pt.

    python make_figures_grandlib.py
    DECK_TEMPLATE=<carrier.odp> python deck_grandlib.py

Numbers: resources/dev/dev-next/RECOVERY_PLAN.md, BRANCHES.md and
beta-tests/TEST_PLAN.md on dev-next-ipfxhh (PR #208), 1 October 2026. The
recovery-plan figures are the plan's own (docs/dev/make_*_diagram.py), rendered
from resources/dev/dev-next/*.svg into figures/recovery/.
"""

import build_deck as bd
from build_deck import (
    FIGS, MARGIN, W, BODY_W,
    build, fit, full_figure, picture, section, slide, table, textbox,
    title_slide,
)
from make_figures_grandlib import FIG_SEVERITY, FIG_VALIDATION, FIG_TIMELINE

bd.OUT = bd.HERE / "2026-10-01-Bustamante-GRAND_Weekly_Meeting.odp"
bd.TITLE = "GRANDlib: merged, cleaned, validated, beta-tested"

UCPH = "1000000100000AC8000004049F3C1DF5.png"
VILLUM = "10000001000003280000010A710211B6.png"
RECOVERY = FIGS / "recovery"

BODY = 18          # body text
TABLE = 15         # every table
PLAN = "Recovery plan: resources/dev/dev-next/ on dev-next"
TRACKER = "Beta test: resources/dev/dev-next/beta-tests/TEST_PLAN.md"


def fig(name, size, x, y):
    href, iw, ih = fit(FIGS / f"{name}.png", *size)
    return picture(href, x, y, iw, ih)


def body(y, lines, w=BODY_W, x=MARGIN):
    frame, h = textbox(x, y, w, lines, size=BODY, line_height=125,
                       space_after=0.3)
    return frame, h


def plan_figure(name, lead, notes=""):
    href, iw, ih = fit(RECOVERY / f"{name}.png", 25.4, 11.9, max_px=3000)
    full_figure(href, iw, ih, lead=lead, ref=PLAN, notes=notes)


def deck():
    # ------------------------------------------------------------------ 1
    title_slide(
        ["GRANDlib:", "merged, cleaned, validated, beta-tested"],
        "Mauricio Bustamante",
        "Niels Bohr Institute, University of Copenhagen",
        ["GRAND Weekly Meeting", "October 1, 2026"],
        [(UCPH, 21.137, 10.32, 6.286, 2.326),
         (VILLUM, 20.779, 12.628, 7.567, 2.413)],
        notes=(
            "About 15 minutes. Four parts: the cleanup, input validation,\n"
            "the beta test in three waves, and the timeline. One request:\n"
            "access to the SPS data to test the aoi part of the code."),
    )

    # ------------------------------------------------------------------ 2
    rows = [
        ["", "What", "Status"],
        ["**Branches**", "All 40 branches decided; dev-next holds everything "
         "kept", "done"],
        ["**Old PRs and issues**", "8 pull requests closed, 1 ported; 12 issues "
         "closed in triage, 7 fixed", "done"],
        ["**Input validation**", "Bad input refused with a clear message: 50 of "
         "60 cases, up from 7", "merged, PR #179"],
        ["**Beta test**", "Two waves, 19 testers, 91 issues; all 11 Critical "
         "issues fixed", "PR #208, CI green"],
    ]
    parts, _ = table(MARGIN, 2.6, [0.0, 6.2, 20.4], rows, size=TABLE,
                     row_h="auto", width=BODY_W,
                     aligns=["left", "left", "center"])
    slide(lead="Since the last update", extras=parts,
          notes="One minute. Each row is a part of the talk.")

    # ================================================================== A
    section(["Cleaning up"])

    plan_figure("recovery", "The recovery plan: 65 of 90 items done",
                notes="Phases 2, 4 and 7 are done. Phase 5 waits on two "
                      "collaboration decisions: Docker, and reprocessing.")
    plan_figure("branches", "Every branch decided: 40 branches",
                notes="Green: in dev-next. Red: decided against. Purple: "
                      "content taken. Nothing deleted yet: archiving to tags "
                      "waits for the software team.")
    plan_figure("history", "Seven years of branches, 2019 to 2026",
                notes="Context only: most of these no longer exist.")

    rows = [
        ["", "Count", "What happened"],
        ["**Pull requests**", "8 + 1", "Closed: #9, #49, #52, #146, #149, #151, "
         "#153, #154. Ported by hand: #150 (Cramér–Rao bounds)"],
        ["**Old issues, triaged**", "12", "Fixed (#95, #122, #123, #136); "
         "duplicate (#80, #84, #94, #99); not reproducible (#90, #92, #156); "
         "obsolete (#47)"],
        ["**Old issues, fixed**", "7", "#71, #89, #91, #104 (Xmax, direction), "
         "#137, #139 (T1 trigger), #140 (reader)"],
        ["**Left for owners**", "4", "#85, #121, #141, #142"],
    ]
    parts, _ = table(MARGIN, 2.6, [0.0, 6.2, 8.6], rows, size=TABLE,
                     row_h="auto", width=BODY_W,
                     aligns=["left", "center", "left"])
    slide(lead="Old pull requests and issues", extras=parts, ref=PLAN)

    # ================================================================== B
    section(["Input validation"])

    t, h = body(2.6, ["60 bad inputs given to public functions: wrong type, "
                      "range, shape or file. ~Before:~ 27 gave a wrong result "
                      "silently. ~After:~ none do."])
    slide(lead="Bad input now stops with a clear message",
          extras=[t, fig("grandlib_validation", FIG_VALIDATION, MARGIN,
                         2.6 + h + 0.3)],
          ref="PR #179; " + TRACKER + ", §4.1")

    # ================================================================== C
    section(["{The} beta test"])

    t, h = body(2.6, [
        "Testers use GRANDlib as a newcomer, an expert or a pipeline user "
        "would, and report only.",
        "Every report is reproduced before it becomes a GitHub issue, titled "
        "*dev-next_beta-test: …*, with its severity.",
        "Fixes go into one pull request, one commit and one test each. A test "
        "must fail without its fix.",
    ])
    rows = [
        ["Wave", "Testers", "Looks for", "Issues"],
        ["**1. Using it**", "11", "Missing steps, wrong docs, brittle input",
         "69"],
        ["**2. Checking it**", "8", "Physics, docs, error handling, tests", "22"],
        ["**3. Breaking it**", "5", "Misuse, extremes, stress, unsafe input",
         "next"],
    ]
    parts, _ = table(MARGIN, 2.6 + h + 0.4, [0.0, 6.2, 9.0, 22.4], rows,
                     size=TABLE, row_h="auto", width=BODY_W,
                     aligns=["left", "center", "left", "center"])
    slide(lead="The beta test: three waves", extras=[t] + parts, ref=TRACKER,
          notes="Testers are AI agents, each in its own copy of the "
                "repository, on dev-next at 91d30a1b.")

    rows = [
        ["Tester", "Area", "Issues", "Critical", "High"],
        ["**Beginners** (×3)", "Setup, the 12 notebooks, examples", "14",
         "–", "2"],
        ["**Expert, dataio**", "Every tree, reader and writer", "12", "–",
         "5"],
        ["**Expert, analysis**", "aoi, reconstruction, event viewer", "7",
         "1", "3"],
        ["**Pipeline, ZHAireS**", "ZHAireS → rawroot → sim2root", "8", "1",
         "3"],
        ["**Pipeline, CoREAS**", "CoREAS, e-field → voltage → ADC", "9", "4",
         "2"],
        ["**Input fuzzers** (×4)", "Readers, scripts, damaged simulations",
         "18", "6", "8"],
    ]
    parts, _ = table(MARGIN, 2.6, [0.0, 6.4, 18.0, 20.5, 23.0], rows,
                     size=TABLE, row_h="auto", width=BODY_W,
                     aligns=["left", "left", "center", "center", "center"])
    t, _ = body(10.6, ["69 issues: !!10 Critical!!, 22 High, 26 Medium, 11 Low. "
                       "Most Critical ones were in the conversion chain."])
    slide(lead="Wave 1: using it", extras=parts + [t], ref=TRACKER,
          notes="Counts include 5 issues found before the wave, while "
                "double-checking the validation PR. An issue reported by two "
                "testers counts once.")

    rows = [
        ["Tester", "Area", "Issues", "Critical", "High"],
        ["**Physics** (×2)", "Coordinates, Xmax, signal chain, noise", "7",
         "1", "2"],
        ["**Documentation** (×2)", "Docs, Handbook, READMEs, docstrings", "4",
         "–", "1"],
        ["**Error handling**", "Silent failures, prints, asserts", "3", "–",
         "2"],
        ["**Test suite**", "Coverage, deliberate bugs, weak tests", "2", "–",
         "1"],
        ["**Input validation**", "Type, range, shape, units", "6", "–", "3"],
        ["**Notebook prose**", "Every claim against its output", "2", "–",
         "1"],
    ]
    parts, _ = table(MARGIN, 2.6, [0.0, 6.4, 18.0, 20.5, 23.0], rows,
                     size=TABLE, row_h="auto", width=BODY_W,
                     aligns=["left", "left", "center", "center", "center"])
    t, _ = body(10.6, ["22 issues: !!1 Critical!!, the EGM96 geoid upside "
                       "down (−7.75 m at GP300 instead of −61.0 m). The test "
                       "suite let 5 of 15 deliberate bugs through."])
    slide(lead="Wave 2: checking it", extras=parts + [t], ref=TRACKER)

    rows = [
        ["Breaker", "Attack"],
        ["**Misuse**", "Methods in the wrong order; one object across files; "
         "events that do not exist"],
        ["**Numerical edges**", "Zenith 0° and 90°; NaN through a whole chain; "
         "one-antenna events"],
        ["**Scale and stress**", "Thousands of files; long traces; memory "
         "over long loops"],
        ["**Environment**", "Unset variables; missing packages; read-only or "
         "partial data"],
        ["**Unsafe input**", "File names that reach a shell; archive "
         "extraction"],
    ]
    parts, _ = table(MARGIN, 2.6, [0.0, 6.4], rows, size=TABLE,
                     row_h="auto", width=BODY_W)
    t, _ = body(10.6, ["Starts once PR #208 is merged. A regression wave "
                       "then re-runs the busiest testers on the fixed code."])
    slide(lead="Wave 3: breaking it", extras=parts + [t], ref=TRACKER)

    rows = [
        ["Severity", "Issues", "Fixed", "Open"],
        ["!!Critical!!", "11", "11", "0"],
        ["**High**", "31", "5", "26"],
        ["**Medium**", "38", "0", "38"],
        ["**Low**", "11", "0", "11"],
        ["**Total**", "91", "16", "75"],
    ]
    parts, _ = table(16.2, 2.9, [0.0, 3.4, 5.6, 7.6], rows, size=TABLE,
                     row_h=0.9, width=9.6,
                     aligns=["left", "center", "center", "center"])
    t, _ = body(9.3, ["!!Critical:!! wrong physics or data, silently.",
                      "**High:** crashes, data loss, documented commands "
                      "that fail."], w=10.4, x=16.2)
    slide(lead="91 issues; every Critical one fixed",
          extras=[fig("grandlib_severity", FIG_SEVERITY, MARGIN, 2.5)] + parts
          + [t],
          ref=TRACKER + ". Fixed: in PR #208, not yet merged")

    rows = [
        ["Issue", "Was wrong", "Effect"],
        ["#209", "CoREAS azimuth mirrored", "Wrong arrival direction"],
        ["#220", "sim2root antenna latitude and longitude", "Off by up to 8 km"],
        ["#227, #229", "RF-chain options ignored; resampled rate lost",
         "Wrong voltages and ADC rate"],
        ["#228", "Unknown CoREAS Xmax read as −1 cm",
         "Voltages near 10{{−13}} µV"],
        ["#237", "L0 e-field paired with an L1 run tree", "Amplitudes doubled"],
        ["#238, #247", "A missing event or shower", "Another event's data"],
        ["#242, #243", "Damaged simulations accepted", "Valid-looking files"],
        ["#250", "EGM96 geoid upside down", "Heights off by 53 m at GP300"],
    ]
    parts, _ = table(MARGIN, 2.6, [0.0, 3.8, 16.4], rows, size=TABLE,
                     row_h="auto", width=BODY_W)
    slide(lead="The 11 Critical bugs, now fixed", extras=parts,
          ref="PR #208: one commit and one test each; CI green on ROOT 6.36 "
              "and 6.38",
          notes="Each gave a plausible wrong answer with exit code 0.")

    rows = [
        ["Area", "Open High issues"],
        ["**Data loss**", "write() replaces trees; Event.write(overwrite=True); "
         "extract_events -ow; re-runs; get_files_from_db.py"],
        ["**Silent wrong results**", "DataDirectory drops files; event order; "
         "NaN in the ADC; geodetic heights west of Greenwich"],
        ["**Commands that fail**", "README quickstart; pipeline commands; "
         "docs examples; helper scripts"],
        ["**Crashes and hangs**", "close_files(); EventList options; the "
         "rf_chain error path"],
    ]
    parts, _ = table(MARGIN, 2.6, [0.0, 6.4], rows, size=TABLE,
                     row_h="auto", width=BODY_W)
    t, _ = body(10.6, ["Next: the 26 High issues, data loss first. The 38 "
                       "Medium and 11 Low are mostly docs and messages."])
    slide(lead="Still to fix", extras=parts + [t], ref=TRACKER)

    # ================================================================== D
    t, _ = body(2.6, [
        "~Access to the GRAND data at CC-IN2P3 (SPS)~, to run the aoi part of "
        "GRANDlib on real events.",
        "The aoi readers and the reconstruction scripts expect the data under "
        "/sps/grand/data/. So far they have been tested only on simulations "
        "and on the small files committed to the repository.",
        "An account, or a few event files copied out, would be enough.",
    ])
    slide(lead="What we need from you", extras=[t],
          notes="This is the one request.")

    # ------------------------------------------------------------------ end
    href, iw, ih = fit(FIGS / "grandlib_timeline.png", *FIG_TIMELINE)
    full_figure(href, iw, ih, lead="Timeline",
                ref="Planned dates to confirm; the release waits for the "
                    "software team's green light",
                notes="Leave this up during questions.")


if __name__ == "__main__":
    build(deck)
