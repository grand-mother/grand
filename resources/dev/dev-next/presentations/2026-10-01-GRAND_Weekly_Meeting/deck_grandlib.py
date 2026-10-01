#!/usr/bin/env python
"""GRANDlib status: content. GRAND Weekly Meeting, 1 October 2026.

Same machinery and carrier as the GNN Board deck (build_deck.py, unchanged);
only the output name and title differ.

    python make_figures_grandlib.py
    python deck_grandlib.py

Numbers: resources/dev/dev-next/RECOVERY_PLAN.md, BRANCHES.md and
beta-tests/TEST_PLAN.md on dev-next-ipfxhh (PR #208), 1 October 2026, 11:40.
"""

import build_deck as bd
from build_deck import (
    FIGS, MARGIN, BLUE, DARKRED, GREY, GREEN,
    build, fit, picture, section, slide, table, textbox, title_slide,
)
from make_figures_grandlib import FIG_SEVERITY, FIG_VALIDATION

bd.OUT = bd.HERE / "2026-10-01-Bustamante-GRAND_Weekly_Meeting.odp"
bd.TITLE = "GRANDlib: merged, cleaned, validated, beta-tested"

UCPH = "1000000100000AC8000004049F3C1DF5.png"
VILLUM = "10000001000003280000010A710211B6.png"


def fig(name, size, x, y):
    href, iw, ih = fit(FIGS / f"{name}.png", *size)
    return picture(href, x, y, iw, ih)


def note(x, y, w, lines, size=16, color=None):
    frame, _ = textbox(x, y, w, lines, size=size, color=color or bd.BLACK,
                       line_height=125, space_after=0.25)
    return frame


def deck():
    # ------------------------------------------------------------------ 1
    title_slide(
        ["GRANDlib:", "merged, cleaned, validated, beta-tested"],
        "++Mauricio Bustamante++",
        "Niels Bohr Institute, University of Copenhagen",
        ["GRAND Weekly Meeting", "October 1, 2026"],
        [(UCPH, 21.137, 10.32, 6.286, 2.326),
         (VILLUM, 20.779, 12.628, 7.567, 2.413)],
        notes=(
            "About 12 minutes. Four parts: branches and issues cleaned up,\n"
            "input validation, the beta test, and what we need from you.\n"
            "The message: dev-next is close to becoming the default branch;\n"
            "every Critical bug found is fixed; the owners' answers are now\n"
            "the bottleneck."),
    )

    # ------------------------------------------------------------------ 2
    rows = [
        ["", "Done", "Status"],
        ["**Branches**", "All 39 branches decided: 24 merged into `dev-next`, "
         "13 decided against or content taken", "%%done%%"],
        ["**Old PRs and issues**", "8 PRs closed or ported; 12 issues closed in "
         "triage, 7 more fixed in one PR", "%%done%%"],
        ["**Input validation**", "Bad input now refused with a clear "
         "`GRANDlib:` message: 50 of 60, up from 7", "%%merged (#179)%%"],
        ["**Beta test**", "13 testers in 2 waves; 91 issues filed; all 11 "
         "Critical ones fixed", "^^PR #208, CI green^^"],
    ]
    parts, _ = table(MARGIN, 2.45, [0.0, 6.0, 20.6], rows, size=16,
                     row_h=[0.86] + [1.55] * 4, width=25.45,
                     aligns=["left", "left", "center"])
    slide(lead="Since the last update", extras=parts,
          notes="One minute. Each row is a section of the talk.")

    # ================================================================== A
    section(["Cleaning up"])

    # ------------------------------------------------------------------ 4
    rows = [
        ["", "Count", "What happened"],
        ["**Branches**", "39", "24 contained in `dev-next`; 10 decided against; "
         "3 whose content was taken. Nothing deleted yet: archiving to tags "
         "waits for the software team's green light"],
        ["**Pull requests**", "8 + 1", "Closed: #9, #49, #52, #146, #149, #151, "
         "#153, #154. Ported by hand: #150 (Cramér–Rao bounds)"],
        ["**Old issues, triaged**", "12", "Fixed (#95, #122, #123, #136); "
         "duplicate (#80, #84, #94, #99); not reproducible (#90, #92, #156); "
         "obsolete (#47)"],
        ["**Old issues, fixed**", "7", "#71, #89, #91, #104 (Xmax, direction), "
         "#137, #139 (T1 trigger), #140 (reader half)"],
        ["**Left for owners**", "4", "#85, #121, #141, #142 (mjtueros)"],
    ]
    parts, _ = table(MARGIN, 2.45, [0.0, 6.0, 8.6], rows, size=15,
                     row_h="auto", width=25.45, aligns=["left", "center", "left"])
    slide(lead="Branches, pull requests and old issues", extras=parts,
          ref="resources/dev/dev-next/BRANCHES.md and RECOVERY_PLAN.md on dev-next",
          notes="The branch map is in BRANCHES.md, read from git.")

    # ================================================================== B
    section(["Input validation"])

    # ------------------------------------------------------------------ 6
    slide(
        lead="Bad input now stops with a clear message",
        body=["60 bad inputs given to public functions (wrong type, wrong "
              "range, missing file). ^^Before:^^ 27 gave a wrong result "
              "silently. ^^After:^^ none do."],
        body_size=18,
        extras=[fig("grandlib_validation", FIG_VALIDATION, MARGIN, 4.55),
                note(MARGIN, 12.95, 25.4,
                     ["Messages name the function and the problem: "
                      "@@GRANDlib: TShower.zenith: should be in [0, 180] deg, "
                      "got 500@@"], size=15)],
        ref="PR #179; TEST_PLAN.md §4.1",
        notes=(
            "Merged as PR #179. One regression found by the beta test\n"
            "(#207, numbers given as text) and one more by tester 9 (#263,\n"
            "channel masks); both fixed in PR #208."),
    )

    # ================================================================== C
    section(["{The} beta test"])

    # ------------------------------------------------------------------ 8
    rows = [
        ["Wave", "Testers", "Role", "Issues"],
        ["**1 — using it**", "11", "Beginners (setup, notebooks), experts "
         "(dataio, aoi, analysis), pipeline (ZHAireS, CoREAS), input fuzzers",
         "69"],
        ["**2 — checking it**", "8", "Physics and geometry, documentation, "
         "error handling, test-suite audit, input validation, notebook prose",
         "22"],
        ["**3 — breaking it**", "5", "After PR #208 merges", "—"],
        ["**4 — regression**", "~4", "Before `dev-next` becomes the default", "—"],
    ]
    parts, _ = table(MARGIN, 2.45, [0.0, 5.6, 8.4, 23.0], rows, size=15,
                     row_h="auto", width=25.45,
                     aligns=["left", "center", "left", "center"])
    slide(
        lead="How the beta test runs",
        extras=parts + [note(MARGIN, 10.6, 25.4, [
            "Each tester gets one mission and reports only. Every finding is "
            "reproduced before it becomes an issue, titled "
            "~dev-next_beta-test: …~",
            "Testers are AI agents, each in its own copy of the repository, on "
            "`dev-next` at `91d30a1b`."], size=15)],
        ref="resources/dev/dev-next/beta-tests/TEST_PLAN.md",
        notes="Issues #180–#271 (#208 is the PR).")

    # ------------------------------------------------------------------ 9
    rows = [
        ["Severity", "Issues", "Fixed", "Open"],
        ["!!Critical!!", "11", "%%11%%", "0"],
        ["**High**", "31", "%%5%%", "26"],
        ["**Medium**", "38", "0", "38"],
        ["**Low**", "11", "0", "11"],
        ["**Total**", "**91**", "%%16%%", "**75**"],
    ]
    parts, _ = table(15.9, 3.0, [0.0, 3.6, 5.8, 7.9], rows, size=17,
                     row_h=0.95, width=10.0,
                     aligns=["left", "center", "center", "center"])
    slide(
        lead="91 issues; every Critical one fixed",
        extras=[fig("grandlib_severity", FIG_SEVERITY, MARGIN, 2.3)] + parts + [
            note(15.9, 9.3, 10.6, [
                "!!Critical!!: wrong physics or data, silently.",
                "**High**: crashes, data loss, documented commands that fail.",
                "Fixes: PR #208, one commit and one test each."], size=14)],
        ref="TEST_PLAN.md §7 and §7a, 1 Oct. 2026",
        notes="Fixed means 'fix in PR #208', not yet merged.")

    # ------------------------------------------------------------------ 10
    rows = [
        ["Issue", "Was wrong", "Effect"],
        ["#209", "CoREAS azimuth mirrored (.inp path)", "wrong arrival direction"],
        ["#220", "sim2root antenna lat/lon", "off by up to ~8 km"],
        ["#227, #229", "RF-chain flags ignored; resampled rate lost",
         "wrong voltages, wrong ADC rate"],
        ["#228", "CoREAS unknown Xmax read as −1 cm", "voltages ~10{{−13}} µV"],
        ["#237", "L0 e-field paired with L1 run tree", "amplitudes doubled"],
        ["#238, #247", "Missing event or shower", "another event's data written"],
        ["#242, #243", "ZHAireS, CoREAS: damaged simulations accepted",
         "valid-looking files"],
        ["#250", "EGM96 geoid upside down", "GP300: −7.75 m → −61.0 m"],
    ]
    parts, _ = table(MARGIN, 2.45, [0.0, 3.6, 15.0], rows, size=15,
                     row_h=0.98, width=25.45)
    slide(lead="The 11 Critical bugs, now fixed", extras=parts,
          ref="All in PR #208; CI green on both ROOT versions",
          notes=(
              "Every one gave a plausible-looking wrong answer with exit code 0.\n"
              "#250 changes absolute heights by ~53 m at the GP300 site."))

    # ------------------------------------------------------------------ 11
    rows = [
        ["Area", "Open High issues"],
        ["**Data loss**", "`write()` replaces trees (#197, #198); "
         "`Event.write(overwrite=True)` (#212); `extract_events -ow` (#244); "
         "re-runs (#240); `get_files_from_db.py` (#245)"],
        ["**Silent wrong results**", "`DataDirectory` drops files (#195); "
         "event order (#199); NaN to ADC (#239); geodetic west of Greenwich "
         "(#251); no-test bugs (#270)"],
        ["**Documented commands fail**", "README quickstart (#185), pipeline "
         "(#221), docs (#257), scripts (#184, #248)"],
        ["**Crashes and hangs**", "`close_files()` (#234); `EventList` (#213, "
         "#235); rf_chain (#255); silent failures (#256)"],
    ]
    parts, _ = table(MARGIN, 2.45, [0.0, 6.6], rows, size=15, row_h="auto",
                     width=25.45)
    slide(lead="Still to fix: 26 High, 38 Medium, 11 Low", extras=parts + [
        note(MARGIN, 12.6, 25.4, ["Next: the High issues, then wave 3. "
                                  "Medium and Low are mostly docs and messages."],
             size=15, color=BLUE)],
          notes="Order: data loss first, then silent wrong results.")

    # ================================================================== D
    section(["What {we} need", "{from} you"])

    # ------------------------------------------------------------------ 13
    rows = [
        ["Need", "From", "Issue"],
        ["**Were productions made with geoid heights?** They move ~53 m at GP300",
         "simulation", "#250"],
        ["**Store the sampling rate in TVoltage?** A data-format change",
         "data model", "#229"],
        ["**Which RA convention do the galactic-noise tables use?**",
         "noise model", "#268"],
        ["**Units of the T1 parameters** `tcmax_ch`, `tprev_ch`, `tper_ch`",
         "trigger group", "#139"],
        ["**Origin of GP80 GPS positions**", "GP80", "#215"],
        ["**Regenerate files** made with the buggy converters",
         "productions", "#209 #220 #237"],
        ["**Small real-data samples** to commit as test fixtures",
         "data", "#271"],
        ["**Green light** to merge PR #208, archive branches, tag", "software team",
         ""],
    ]
    parts, _ = table(MARGIN, 2.45, [0.0, 17.4, 22.0], rows, size=15,
                     row_h="auto", width=25.45,
                     aligns=["left", "center", "center"])
    slide(lead="Decisions and data only you can give", extras=parts,
          notes="The rest of the work does not need anyone else.")

    # ------------------------------------------------------------------ 14
    slide(
        lead="Next",
        body=[
            ("1.", "Merge PR #208: 11 Critical + 5 High fixes, each with a test "
                   "that fails without it."),
            ("2.", "Fix the open High issues, data loss first."),
            ("3.", "Wave 3 (breakers), then wave 4 (regression)."),
            ("4.", "Make `dev-next` the default branch; archive old branches "
                   "to tags; tag a release. ^^After the software team's green "
                   "light.^^"),
        ],
        body_size=20,
        notes="Leave this up during questions.")


if __name__ == "__main__":
    build(deck)
