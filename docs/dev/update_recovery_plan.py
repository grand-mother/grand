# -*- coding: utf-8 -*-
r"""Keeps the measurable claims in the recovery plan true.

    python docs/dev/update_recovery_plan.py            # measure and rewrite
    python docs/dev/update_recovery_plan.py --check     # fail if out of date

The plan is prose, and prose is the point: the phases, the decisions and the
corrections are judgements somebody has to write. But it also carries counts --
how many tests pass, how many branches are still out, how many issues are
documented -- and those rot. On 2026-09-24 the plan still said "593 passed" and
"15 carry genuinely unmerged patches" when the numbers were 601 and 6, because
the last person to change the repository was not the last person to edit the
sentence about it.

So the counts are generated and the prose is not. This script owns the regions
between ``<!-- measured: begin -->`` and ``<!-- measured: end -->`` and touches
nothing else. Run it after anything that moves a number, alongside the four
diagram and inventory generators.

``--check`` rewrites nothing and exits non-zero if the file would change, which
is what a CI job would call.

``--tests-summary`` and ``--tests-source`` take the test result from somewhere
else instead of running pytest here -- for a machine that cannot run the whole
suite, such as one without the antenna data model::

    python docs/dev/update_recovery_plan.py \
        --tests-summary "616 passed, 10 skipped, 11 xfailed" \
        --tests-source "CI run 36017933785 on c3b2caa5"

The source is printed beside the numbers, so the plan never presents a count
from elsewhere as one measured where the rest of the block was.
"""
import argparse
import datetime
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import branch_facts as facts                                  # noqa: E402

PLAN = facts.ROOT / "resources" / "dev" / "dev-next" / "RECOVERY_PLAN.md"

BEGIN, END = "<!-- measured: begin -->", "<!-- measured: end -->"

#: The lint command CI runs. Kept identical on purpose: a plan that reports a
#: different scope from the gate is reporting about something else.
LINT = ["ruff", "check", "grand/", "tests/", "quality/", "notebooks/",
        "docs/dev/", "granddb/"]


def run(args, **kwargs):
    r"""Runs a command in the repository root and returns its stdout."""
    return subprocess.run(args, capture_output=True, text=True,
                          cwd=facts.ROOT, **kwargs).stdout


def measure_tests():
    r"""Returns pytest's own summary line, or None if it could not be run.

    The line is taken verbatim rather than recomposed from counts, so that a
    failure or an error appears in the plan as pytest phrased it instead of
    being flattened into a number that looks like a pass.
    """
    out = run([sys.executable, "-m", "pytest", "tests/", "-q"])
    for line in reversed(out.strip().split("\n")):
        if re.search(r"\d+ (passed|failed|error)", line):
            return re.sub(r"\s+in [\d.]+s$", "", line.strip())
    return None


def measure_lint():
    r"""Returns the lint verdict as a phrase."""
    out = run(LINT)
    if "All checks passed!" in out:
        return "clean over `grand/ tests/ quality/ notebooks/ docs/dev/ granddb/`"
    found = re.search(r"Found (\d+) error", out)
    return ("**%s findings** over the gated tree" % found.group(1) if found
            else "could not be measured")


def count_issues():
    r"""Counts the entries in ``known_issues.rst`` by their ``:Status:`` line.

    The statuses are prose -- "open, not blocking", "**resolved 2026-09-07.**
    Merged from ``dev_snonis``" -- so only the first word is classified, and
    anything that is neither open nor settled is counted separately and
    reported. A generated number that quietly misfiles an entry is worse than
    one that admits it could not tell.

    Returns
    -------
    tuple of int
        ``(total, open, settled, unclassified)``.
    """
    path = facts.ROOT / "docs" / "source" / "known_issues.rst"
    if not path.exists():
        return 0, 0, 0, 0
    text = path.read_text(encoding="utf-8")
    total = len(re.findall(r"^\.\. _issue-", text, re.M))

    opened = settled = unclear = 0
    for status in re.findall(r"^:Status:\s*(.+)$", text, re.M):
        word = re.sub(r"[^a-z]", "", status.replace("*", "").split()[0].lower())
        if word == "open":
            opened += 1
        elif word in ("fixed", "resolved", "withdrawn", "closed"):
            settled += 1
        else:
            unclear += 1
    return total, opened, settled, unclear


def measure(tests_summary=None, tests_source=None):
    r"""Returns the rows of the generated block.

    Parameters
    ----------
    tests_summary : str, optional
        A pytest summary line to report instead of running the suite here.
    tests_source : str, optional
        Where that summary came from; required with it, and shown beside it.

    Returns
    -------
    list of tuple
        ``(label, value)`` pairs, in the order they appear.
    """
    info = facts.collect(include_historical=False)
    state = [facts.display_state(n, e) for n, e in info.items()]

    n_issues, n_open, n_settled, n_unclear = count_issues()

    notebooks = sorted((facts.ROOT / "notebooks").glob("*.ipynb"))
    tag = run(["git", "describe", "--tags", "--abbrev=0"]).strip() or "none"
    head = run(["git", "rev-parse", "--short", "HEAD"]).strip()

    if tests_summary:
        tests_cell = "**%s** — reported by %s, not run here" % (
            tests_summary.strip(), tests_source.strip())
    else:
        tests = measure_tests()
        tests_cell = "**%s**" % tests if tests else "could not be run here"
    rows = [
        ("Test suite", tests_cell),
        ("Lint", measure_lint()),
        ("Branches", "%d contained in `dev-next`, **%d still out**, %d decided "
                     "against, %d absorbed"
                     % (state.count("merged"), state.count("unmerged"),
                        state.count("retired"), state.count("absorbed"))),
        ("Known issues", "%d documented in `known_issues.rst`: %d open, %d "
                         "settled%s"
                         % (n_issues, n_open, n_settled,
                            ", %d not classifiable from its :Status: line"
                            % n_unclear if n_unclear else "")),
        ("Notebooks", "%d, generated by `notebooks/make_notebooks.py`"
                      % len(notebooks)),
        ("Tag", "`%s`, at `%s`" % (tag, head)),
    ]
    return rows


def render(rows):
    r"""Returns the generated block, markers included."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    out = [BEGIN,
           "",
           "*Measured %s by `docs/dev/update_recovery_plan.py`. Everything in "
           "this block is generated; edit the script, not the table.*" % stamp,
           "",
           "| | |",
           "|---|---|"]
    out += ["| %s | %s |" % (label, value) for label, value in rows]
    out += ["", END]
    return "\n".join(out)


def main():
    r"""Rewrites the generated block, or checks that it is current."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="rewrite nothing; exit 1 if the plan is stale")
    parser.add_argument("--tests-summary",
                        help="a pytest summary line to report instead of "
                             "running the suite here")
    parser.add_argument("--tests-source",
                        help="where --tests-summary came from, e.g. a CI run")
    args = parser.parse_args()
    if bool(args.tests_summary) != bool(args.tests_source):
        parser.error("--tests-summary and --tests-source go together: a count "
                     "without its source cannot be checked")
    if args.tests_summary and not re.search(r"\d+ passed", args.tests_summary):
        parser.error("--tests-summary does not look like a pytest summary line")

    if not PLAN.exists():
        raise SystemExit("cannot find the plan at %s" % PLAN)
    text = PLAN.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(
            "the plan carries no measured block. Add\n\n  %s\n  %s\n\nwhere "
            "the generated table should go." % (BEGIN, END))

    block = render(measure(args.tests_summary, args.tests_source))
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    updated = pattern.sub(lambda _: block, text, count=1)

    # The timestamp changes on every run, so compare everything but that.
    def without_stamp(s):
        return re.sub(r"\*Measured [^*]*\*", "", s)

    if args.check:
        if without_stamp(updated) != without_stamp(text):
            print("the recovery plan's measured block is out of date; run "
                  "python docs/dev/update_recovery_plan.py", file=sys.stderr)
            raise SystemExit(1)
        print("the recovery plan's measured block is current")
        return

    PLAN.write_text(updated, encoding="utf-8")
    print("updated %s" % PLAN, file=sys.stderr)


if __name__ == "__main__":
    main()
