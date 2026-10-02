# GRANDlib recovery: summary

*Status on 2026-10-01. Owner: Mauricio Bustamante. The details are in
[RECOVERY_PLAN.md](https://github.com/grand-mother/grand/blob/dev-next/resources/dev/dev-next/RECOVERY_PLAN.md), [BRANCHES.md](https://github.com/grand-mother/grand/blob/dev-next/resources/dev/dev-next/BRANCHES.md) and the
[beta-test plan](https://github.com/grand-mother/grand/blob/dev-next/resources/dev/dev-next/beta-tests/TEST_PLAN.md).*

## Where we started

- **No working trunk.**
  - `master`, the GitHub default, was 1163 commits behind `dev`, the real trunk.
  - A second trunk, `main`, had been abandoned 355 commits behind.
- **No working CI.** The last 36 workflow runs had all been cancelled, and the
  test workflow had never completed a run. No recent merge had been checked by
  anything.
- **A backlog:** 36 branches and 10 open pull requests.

## What was done

All work lands on one integration branch, `dev-next`, cut from `dev`. Nothing
has been deleted, so rollback stays trivial.

| Area | Result |
|---|---|
| **Branches** | All 40 branches reviewed and decided. Everything worth keeping is in `dev-next`. Archiving the rest waits for the software team's green light |
| **Old pull requests and issues** | 8 pull requests closed and 1 ported by hand. 12 old issues triaged, 7 fixed, 5 left with their owners |
| **CI** | Lint, tests on two ROOT versions, notebooks and the documentation all run and are green. A gate stops a skipped test job from reading as a pass. `dev-next` is protected against force-push and deletion |
| **Tests** | 841 passed, 0 failed (it was 640 on 2026-09-28) |
| **Code coverage** | Measured in CI on every run, for `grand/` and `granddb/` separately: 73 % and 20 % (63 % together) when last measured, on 2026-09-08. `granddb/` was not measured at all before |
| **Docstrings** | Numpydoc docstrings across `grand/`: 657 of its 666 functions (99 %) now have one, 61 % document their parameters and 58 % their returns. A lint ratchet enforces them: the list of exempt modules may shrink but never grow |
| **Documentation** | Published at https://grand-mother.github.io/grand/. 23 written pages, the API reference, a Handbook (also as a PDF) and 12 tutorial notebooks. The docs build with zero warnings |
| **Input validation** | Bad input is now refused with a clear `GRANDlib:` message in 50 of 60 test cases, up from 7. No crashes, and no silent `exit()` |
| **Beta test** | 3 waves (using, checking, breaking), 24 independent testers. Every finding was reproduced before it was logged |
| **Security** | Hardening of archive handling, external commands, the production pipeline and remote data access. Reported privately as GitHub security advisories; the fixes are merged |

## Beta test in numbers

Every Critical issue is fixed: no Critical issue is open.

| | Filed | Critical | Fixed so far |
|---|---|---|---|
| Wave 1: using it | 69 | 10 | all 10 Critical and 1 High |
| Wave 2: checking it | 22 | 1 | the Critical (geoid upside down) and 4 High |
| Wave 3: breaking it | 17 | 2 | both Critical |
| **Total** | **108** | **13** | **18 closed** |

Examples of what the Critical fixes corrected:
- the CoREAS converter mirrored the azimuth;
- antenna positions were wrong by up to about 8 km;
- the EGM96 geoid was upside down;
- the antenna response was wrapped below the horizon;
- events were lost when several jobs wrote one file;
- damaged simulations were accepted silently.

**Open now: 95 issues.**

| Severity | Open |
|---|---|
| High | 33 (the largest group, 13, is in `dataio`: data loss and wrong values) |
| Medium | 44 |
| Low | 13 |
| With their owners | 5 (feature requests and data-model questions) |

## What remains

1. **Fix the 33 High issues**, starting with `dataio`. Then run wave 4 of the
   beta test, which checks for regressions.
2. **Audit the documentation.**
3. **A decision for the collaboration:** whether to provide a Docker image.
4. **Access to SPS data** to test the `aoi` part of the code.
5. **Promote `dev-next` to `main`.** One exit criterion remains: a
   clean-machine install.
6. **Archive the retired branches and tag the release**, once the software team
   gives the green light.
