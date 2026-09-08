# -*- coding: utf-8 -*-
r"""Writes the branch-by-branch reference for the recovery.

    python docs/dev/make_branch_inventory.py

Writes ``resources/dev/dev-next/BRANCHES.md``: every branch that exists in the
repository today, with who made it, where it came from, what came out of it,
and whether it should be merged.

The companion to ``make_branch_diagram.py``. The diagram shows the shape; this
shows the detail the shape cannot hold. Both read ``branch_facts.py``, so a
branch described or judged once is described and judged in both.

**Live branches only.** ``branch_facts.collect`` can also recover the branches
that were merged and then deleted, from the second parent of the merges that
took them, and ``make_history_diagram.py`` draws them. They are left out here
because this is a working document: it exists to be acted on, and a branch that
no longer exists cannot be acted on.

Green is in the trunk, amber is still out -- the diagram's colours. Markdown
has no colour of its own, so they are the coloured circles; they survive
copy-paste, unlike HTML, and render on GitHub, in VS Code and through Sphinx.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import branch_facts as facts                                  # noqa: E402

#: Verdict codes to the words used in the document.
ACTIONS = {
    "merge": ("Merge", "test-merges clean"),
    "merge-hand": ("Merge by hand", "conflicts, but only on context"),
    "cherry-pick": ("Cherry-pick", "take part of it, never the branch"),
    "no": ("Do not merge", "nothing here applies"),
    "decide": ("Decision needed", "a question, not a merge"),
    "ask": ("Ask the author", "needs its author before anything else"),
}


def cell(text):
    r"""Escapes the pipe, which would otherwise end a table cell."""
    return str(text).replace("|", "\\|")


def summary_row(name, entry):
    r"""Returns one row of the big table.

    Parameters
    ----------
    name : str
        Branch name.
    entry : dict
        Its record from :func:`branch_facts.collect`.

    Returns
    -------
    str
        A markdown table row.
    """
    if entry["state"] == "trunk":
        mark, merged = "\U0001F535", "n/a -- this is the trunk"
    elif entry["state"] == "merged":
        where = entry["merged_on"] or "?"
        if where == "ancestor":
            merged = "yes, no merge names it"
        elif entry["merge_named"]:
            merged = "yes, %s (`%s`)" % (where, entry["merged_by"])
        else:
            # Contained by this date, but merged by fast-forward, squash or
            # rebase: no merge commit names the branch, so nothing in the
            # history says which commits were its own.
            merged = "yes, in trunk by %s" % where
        mark = "\U0001F7E2"
    elif facts.display_state(name, entry) == "retired":
        mark = "\U0001F534"
        merged = ("**no, and never will** (decided %s)"
                  % facts.DECIDED[name][0])
    else:
        mark, merged = "\U0001F7E1", "**no**"

    action = facts.VERDICTS.get(name, ("", ""))[0]
    verdict = ACTIONS[action][0] if action else (
        "--" if entry["state"] != "unmerged" else "not yet reviewed")
    if name in facts.DECIDED:
        verdict = "**%s** — decided" % verdict

    children = entry["children"]
    feeds = ", ".join("`%s`" % c for c in children[:3])
    if len(children) > 3:
        feeds += " +%d more" % (len(children) - 3)

    # "Own" is git cherry: patches that are in no other branch. It is the
    # number a merge decision turns on, and for an old branch it is nothing
    # like the commit count -- snonis_sim2root_test_merge has 279 commits
    # since it diverged and two patches of its own.
    own = str(entry["ahead"]) if entry["state"] == "unmerged" else "--"

    return "| " + " | ".join([
        mark,
        "`%s`" % cell(name),
        cell(facts.DESCRIPTIONS.get(name, "")),
        entry["created"] or "--",
        cell(entry["creator"] or "--"),
        "`%s`" % entry["parent"] if entry["parent"] else "--",
        feeds or "--",
        str(entry["commits"]) if entry["commits"] else "--",
        own,
        entry["last"] or "--",
        cell(entry["author"] or "--"),
        merged,
        verdict,
    ]) + " |"


def build(info):
    r"""Returns the finished markdown.

    Parameters
    ----------
    info : dict
        The output of :func:`branch_facts.collect`.

    Returns
    -------
    str
        The document.
    """
    stamp, commit = facts.provenance("docs/dev/make_branch_inventory.py")

    unmerged = sorted((n for n, e in info.items() if e["state"] == "unmerged"),
                      key=lambda n: info[n]["last"], reverse=True)
    merged = sorted((n for n, e in info.items() if e["state"] == "merged"),
                    key=lambda n: info[n]["last"], reverse=True)
    trunk = [n for n, e in info.items() if e["state"] == "trunk"]

    out = [
        "# Every branch in grand, and what should happen to it",
        "",
        "Generated %s by `docs/dev/make_branch_inventory.py`, at commit `%s`."
        % (stamp, commit),
        "",
        "Do not edit this file. Edit the generator, or the descriptions and",
        "verdicts in `docs/dev/branch_facts.py`, and run it again.",
        "",
        "%d branches exist in the repository today: %d still carrying patches "
        "of their own, %d contained in `dev-next`, and the trunk itself. "
        "Everything except the descriptions and the verdicts is read from git, "
        "so it cannot go stale."
        % (len(info), len(unmerged), len(merged)),
        "",
        "\U0001F7E2 in the trunk  ·  \U0001F7E1 still out, undecided  ·  "
        "\U0001F534 decided against, will never be merged  ·  "
        "\U0001F535 the trunk itself",
        "",
        "A verdict is a recommendation until it is agreed. The red ones have "
        "been agreed and are settled; the amber ones are still open questions, "
        "which is why they do not share a colour.",
        "",
        "*Created* is the first commit that was the branch's own. Branches that "
        "were merged and then deleted are not listed: this is a document to "
        "act on, and they cannot be acted on. They are drawn in "
        "`history.svg`, which covers the repository's whole history.",
        "",
        "*Commits* counts everything on the branch since it diverged. *Own* "
        "counts the patches that are in no other branch, which is what a "
        "merge decision turns on and is sometimes a much smaller number: "
        "`snonis_sim2root_test_merge` has 279 commits and two patches of its "
        "own. Reading the first number as the second is the mistake this "
        "column exists to prevent.",
        "",
        "The verdicts were last reviewed end to end on %s."
        % facts.VERDICTS_REVIEWED,
        "",
        "---",
        "",
    ]

    # The decision log. Generated from DECIDED, so it cannot fall out of step
    # with the colours -- a hand-kept list beside a generated one is how the
    # old branch table in the plan came to be wrong.
    if facts.DECIDED:
        taken = sorted(facts.DECIDED.items(), key=lambda kv: kv[1][0],
                       reverse=True)
        out += ["## Decisions taken (%d)" % len(taken),
                "",
                "Dispositions that have been agreed. Everything not listed "
                "here is still an open question, whatever this document "
                "recommends.",
                "",
                "| Decided | Branch | Disposition | Why, in short |",
                "|---|---|---|---|"]
        for name, (when, why) in taken:
            action = facts.VERDICTS.get(name, ("", ""))[0]
            label = ACTIONS.get(action, ("--", ""))[0]
            mark = "\U0001F534" if action == "no" else "\U0001F7E1"
            out.append("| %s | %s `%s` | %s | %s. |"
                       % (when, mark, cell(name), label, cell(why)))
        out += ["", "---", ""]

    out += [
        "## Still out (%d)" % len(unmerged),
        "",
        "These carry patches that are in no other branch. Reasons are below "
        "the table.",
        "",
    ]

    header = ("|  | Branch | What it is | Created | By | Off | Feeds |"
              " Commits | Own | Last commit | By | Merged | Verdict |")
    rule = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"

    out += [header, rule]
    out += [summary_row(n, info[n]) for n in unmerged]
    out += ["", "### Why, one by one", ""]

    for name in unmerged:
        action, reason = facts.VERDICTS.get(
            name, ("", "Not yet reviewed."))
        label, gloss = ACTIONS.get(action, ("Not yet reviewed", ""))
        decided = facts.DECIDED.get(name, (None, None))[0]
        out += ["#### %s `%s` — %s"
                % ("\U0001F534" if decided and action == "no"
                   else "\U0001F7E1", name, label),
                "",
                "*%s*%s" % (facts.DESCRIPTIONS.get(name, "no description"),
                            " — %s" % gloss if gloss else ""),
                "",
                reason,
                ""]
        if decided:
            out += ["**Decided %s.** This is settled, not a proposal." % decided,
                    ""]

    out += ["---", "", "## The trunk", "", header, rule]
    out += [summary_row(n, info[n]) for n in trunk]

    out += ["", "---", "",
            "## Already in `dev-next` (%d)" % len(merged), "",
            "Nothing to do with these. They are here because a branch that "
            "was merged years ago and then deleted is otherwise invisible, "
            "and the question \"was that ever merged?\" has been asked often "
            "enough during this recovery to be worth answering in a table.",
            "", header, rule]
    out += [summary_row(n, info[n]) for n in merged]
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    target = facts.ROOT / "resources" / "dev" / "dev-next" / "BRANCHES.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(facts.collect(include_historical=False)),
                      encoding="utf-8")
    print("wrote %s" % target, file=sys.stderr)
