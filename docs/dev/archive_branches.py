# -*- coding: utf-8 -*-
r"""Archives settled branches to tags, then (separately) deletes them.

    python docs/dev/archive_branches.py                 # plan: what would happen
    python docs/dev/archive_branches.py --tag           # create the tags locally
    python docs/dev/archive_branches.py --verify        # check tags against branches
    python docs/dev/archive_branches.py --push          # push the tags
    python docs/dev/archive_branches.py --delete        # delete the archived branches

Each step is its own command, and nothing is deleted by default.  Run them in
that order, looking at the output of each.

**Why tags.**  Git discards a commit only when nothing refers to it.  A tag on
a branch's last commit keeps that commit and its whole history, so deleting the
branch afterwards removes only the name: every commit stays in the repository,
browsable on GitHub under Tags, and any clone can bring the branch back with::

    git branch <name> archive/<name>-<YYYY-MM>

**The convention** is the repository's own, from ``archive/master-2025-03`` and
``archive/dev-2026-09``: ``archive/<branch>-<month archived>``, an annotated
tag whose message says what the branch was and why it was retired.  The
messages here are written from ``branch_facts.py``, so they carry the same
decision that ``BRANCHES.md`` shows.

**Which branches.**  Every branch the inventory knows except those in
``KEEP``: the trunk, the working branch, ``ci/docker-test`` (pushing to it is
how the Docker workflow is started), and ``master``, ``dev`` and ``main``,
which Phases 9 and 10 retire deliberately.  A branch that is still undecided
is refused rather than archived.

**Safety.**  ``--delete`` deletes a branch only when the tag on the remote
points at the branch's current tip, and it deletes with ``--force-with-lease``
on that tip, so a branch that received a commit since it was archived is left
alone.  Deleting a branch closes any pull request open from it.
"""

import argparse
import datetime
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import branch_facts as facts  # noqa: E402

#: Not archived here, each for a reason.
KEEP = {
    facts.TRUNK: "the trunk",
    "dev-next-ipfxhh": "the working branch the recovery's pull requests come from",
    "ci/docker-test": "pushing to it is how the Docker workflow is started",
    "master": "retired to archive/master-2025-03 in Phase 10",
    "dev": "retired in Phase 10; archive/dev-2026-09 predates 7cd02097, so it needs a new tag then",
    "main": "archived in Phase 9, which frees the name",
}

#: What each state means in a tag message.
STATE_TEXT = {
    "merged": "Everything on it is in dev-next.",
    "retired": "Decided against: it will not be merged.",
    "absorbed": "Its content was taken into dev-next another way; its own patches were not merged.",
}


def _git(*args, check=True):
    r"""Runs git in the repository root and returns its stripped stdout."""
    done = subprocess.run(["git"] + list(args), capture_output=True, text=True,
                          cwd=facts.ROOT)
    if check and done.returncode != 0:
        raise SystemExit("git %s failed:\n%s" % (" ".join(args), done.stderr))
    return done.stdout.strip()


def plan(month):
    r"""Returns what would be archived, and what is refused.

    Parameters
    ----------
    month : str
        ``YYYY-MM``, the suffix of the tag names.

    Returns
    -------
    list of dict
        One per candidate branch: ``name``, ``tip``, ``state``, ``tag``,
        ``message``, and ``refused`` (a reason, or empty).
    """
    info = facts.collect()
    rows = []
    for name in sorted(info):
        entry = info[name]
        if name in KEEP or entry.get("deleted"):
            continue
        state = facts.display_state(name, entry)
        tip = _git("rev-parse", "origin/%s" % name)
        refused = ""
        if state not in STATE_TEXT:
            refused = "state is %r: only settled branches are archived" % state
        rows.append(dict(name=name, tip=tip, state=state,
                         tag="archive/%s-%s" % (name, month),
                         message=_message(name, entry, state),
                         refused=refused))
    return rows


def _message(name, entry, state):
    r"""Writes an archive tag's message from what the inventory knows."""
    description = facts.DESCRIPTIONS.get(name, "")
    lines = ["%s as it stood when archived%s" % (
        name, " -- %s" % description if description else ""), ""]
    lines.append(STATE_TEXT.get(state, ""))
    if name in facts.DECIDED:
        when, why = facts.DECIDED[name]
        lines.append("Decided %s: %s." % (when, why))
    elif entry.get("merged_on") and entry["merged_on"] != "ancestor":
        lines.append("In dev-next since %s%s." % (
            entry["merged_on"],
            " (merge %s)" % entry["merged_by"] if entry.get("merged_by") else ""))
    lines += ["",
              "Last commit %s by %s." % (entry.get("last") or "?",
                                         entry.get("author") or "?"),
              "Details: resources/dev/dev-next/BRANCHES.md.",
              "Restore with:  git branch %s <this tag>" % name]
    return "\n".join(lines)


def _show(rows):
    r"""Prints the plan as a table."""
    for row in rows:
        print("  %-9s %-58s %s%s" % (
            row["state"], row["name"], row["tag"],
            "   REFUSED: %s" % row["refused"] if row["refused"] else ""))
    kept = ", ".join("%s (%s)" % item for item in KEEP.items())
    print("\n  %d to archive, %d refused. Kept: %s"
          % (sum(not r["refused"] for r in rows),
             sum(bool(r["refused"]) for r in rows), kept))


def tag(rows):
    r"""Creates the annotated tags locally; never moves an existing one."""
    for row in rows:
        if row["refused"]:
            continue
        existing = _git("rev-parse", "-q", "--verify",
                        "refs/tags/%s^{commit}" % row["tag"], check=False)
        if existing:
            if existing != row["tip"]:
                raise SystemExit("%s already exists at %s, not at the branch "
                                 "tip %s; not moving it"
                                 % (row["tag"], existing[:8], row["tip"][:8]))
            continue
        _git("tag", "-a", row["tag"], row["tip"], "-m", row["message"])
        print("  tagged %s -> %s" % (row["tag"], row["tip"][:8]))


def verify(rows, remote=False):
    r"""Checks every tag holds exactly the branch's commits.

    Parameters
    ----------
    rows : list of dict
        From :func:`plan`.
    remote : bool, optional
        Check the tags on origin rather than the local ones.

    Returns
    -------
    bool
        True when every archivable branch is held by its tag.
    """
    remote_tags = {}
    if remote:
        for line in _git("ls-remote", "--tags", "origin",
                         "refs/tags/archive/*").split("\n"):
            if line.endswith("^{}"):
                sha, ref = line.split()
                remote_tags[ref[len("refs/tags/"):-3]] = sha
    good = True
    for row in rows:
        if row["refused"]:
            continue
        if remote:
            held = remote_tags.get(row["tag"], "")
        else:
            held = _git("rev-parse", "-q", "--verify",
                        "refs/tags/%s^{commit}" % row["tag"], check=False)
        # Same commit, so the same history: nothing on the branch that the
        # tag does not reach.
        missing = (_git("rev-list", "--count", "%s..%s" % (held, row["tip"]))
                   if held else "all")
        ok = held == row["tip"] and missing == "0"
        good &= ok
        print("  %s %-58s %s" % ("ok  " if ok else "FAIL", row["name"],
                                 "" if ok else "tag %s, %s commits not held"
                                 % (held[:8] or "missing", missing)))
    return good


def push(rows):
    r"""Pushes the archive tags to origin."""
    tags = ["refs/tags/%s" % r["tag"] for r in rows if not r["refused"]]
    _git("push", "origin", *tags)
    print("  pushed %d tags" % len(tags))


def delete(rows):
    r"""Deletes each branch whose remote tag holds its current tip."""
    if not verify(rows, remote=True):
        raise SystemExit("not deleting: some branches are not held by a tag "
                         "on origin. Push the tags first.")
    for row in rows:
        if row["refused"]:
            continue
        _git("push", "--force-with-lease=refs/heads/%s:%s" % (row["name"], row["tip"]),
             "origin", ":refs/heads/%s" % row["name"])
        print("  deleted %s (kept in %s)" % (row["name"], row["tag"]))


def main():
    r"""Parses the command line and runs one step."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--tag", action="store_true", help="create the tags locally")
    step.add_argument("--verify", action="store_true",
                      help="check the local tags hold the branches")
    step.add_argument("--push", action="store_true", help="push the tags")
    step.add_argument("--delete", action="store_true",
                      help="delete the branches whose tags are on origin")
    parser.add_argument("--month", default=datetime.date.today().strftime("%Y-%m"),
                        help="tag suffix, YYYY-MM (default: this month)")
    parser.add_argument("--show-messages", action="store_true",
                        help="print each tag message with the plan")
    args = parser.parse_args()

    # Branches with --prune, so a deleted branch is seen as deleted; tags
    # without it, which would otherwise discard tags created by --tag and not
    # yet pushed.  Remote tags never overwrite local ones here: --tag refuses
    # to move a tag, and a conflict should stop the run, not be resolved.
    _git("fetch", "--prune", "origin")
    _git("fetch", "origin", "refs/tags/archive/*:refs/tags/archive/*")
    rows = plan(args.month)
    if args.tag:
        tag(rows)
    elif args.verify:
        raise SystemExit(0 if verify(rows) else 1)
    elif args.push:
        push(rows)
    elif args.delete:
        delete(rows)
    else:
        _show(rows)
        if args.show_messages:
            for row in rows:
                print("\n--- %s\n%s" % (row["tag"], row["message"]))


if __name__ == "__main__":
    main()
