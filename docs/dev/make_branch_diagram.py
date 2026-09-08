# -*- coding: utf-8 -*-
r"""Draws every branch in the repository as a family tree against the trunk.

The recovery diagram (``make_recovery_diagram.py``) shows the *merge queue* --
the handful of branches somebody curated. That is useful and, alone,
misleading: an empty queue reads as "everything is merged" while 38 branches
exist and fifteen still carry patches of their own.

This one shows all of them, and how they are related. Columns are generations:
a branch sits one column right of its nearest ancestor, so the tree reads left
to right from the oldest roots to ``dev-next``.

    python docs/dev/make_branch_diagram.py

Writes ``resources/dev/dev-next/branches.svg``, and a copy at
``docs/source/branches.svg``. Two copies because the recovery plan embeds this
picture and is read in two places: GitHub resolves the image beside the plan,
Sphinx relative to ``docs/source/``, and no single relative path satisfies
both. Everything except the
descriptions is read from git at build time, so the picture cannot drift from
the repository the way a hand-maintained list does.

On "nearest ancestor": the parent of a branch A is the branch B whose tip is an
ancestor of A and whose own last commit is the most recent among such branches.
For a branch nobody has merged, that is where it forked from. For one that has
been merged it can instead reflect the merge -- ``dev-next`` descends from
``dev_snonis`` in this sense because ``dev_snonis`` was merged into it. The line
means "contains, most recently", which is a fact; "was branched from" is an
inference that happens to be right most of the time.
"""
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import branch_facts as facts                                  # noqa: E402
from branch_facts import DESCRIPTIONS, SHORT, collect, git    # noqa: E402,F401

WORK = [("conda env", "done"), ("setup.sh", "done"), ("pyproject", "done"),
        ("Sphinx docs", "done"), ("schema test", "done"), ("CI green", "done"),
        ("593 tests", "done"), ("cov 73%", "done"), ("interface", "todo")]

FILL = {"trunk": "#D5E8F5", "merged": "#E1F1EA", "unmerged": "#F6EDDA",
        "retired": "#F7E6E7", "absorbed": "#EFE9F5",
        "done": "#E1F1EA", "todo": "#EDF1F3"}
EDGE = {"trunk": "#1F5C82", "merged": "#1D7A57", "unmerged": "#8A6210",
        "retired": "#A9484E", "absorbed": "#6B4E8E",
        "done": "#1D7A57", "todo": "#BCC7CE"}
TEXT = {"trunk": "#1F5C82", "merged": "#1D7A57", "unmerged": "#8A6210",
        "retired": "#A9484E", "absorbed": "#6B4E8E",
        "done": "#1D7A57", "todo": "#7A8994"}

MONO = "IBM Plex Mono, monospace"
SANS = "IBM Plex Sans, Helvetica, Arial, sans-serif"
BOX_W, BOX_H = 232, 46
COL, ROW = 292, 54
X0, TOP = 40, 168
CHAR_W = 5.45


def esc(text):
    r"""Escapes the five XML characters."""
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
                 ('"', "&quot;"), ("'", "&apos;")):
        text = text.replace(a, b)
    return text


def build(info):
    r"""Returns the finished SVG.

    Raises
    ------
    SystemExit
        If a label will not fit its box.  Refusing to build beats clipping a
        branch name and hoping nobody looks closely -- the recovery diagram
        drew overlapping boxes for some time before anyone noticed.
    """
    columns = collections.defaultdict(list)
    for name, entry in info.items():
        columns[entry["generation"]].append(name)
    for gen in columns:
        columns[gen].sort(key=lambda n: info[n]["last"], reverse=True)

    ngen = max(columns) + 1
    width = X0 * 2 + (ngen - 1) * COL + BOX_W
    height = TOP + max(len(v) for v in columns.values()) * ROW + 62

    at = {}
    for gen, names in columns.items():
        for i, name in enumerate(names):
            at[name] = (X0 + gen * COL, TOP + i * ROW)

    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'width="%d" height="%d" font-family="%s">'
           % (width, height, width, height, SANS),
           '<rect width="%d" height="%d" fill="#FFFFFF"/>' % (width, height),
           '<text x="%d" y="34" font-size="15" font-weight="600" fill="#22313A">'
           'Every branch in grand and its status</text>' % X0,
           '<text x="%d" y="52" font-size="10" fill="#5A6A73">%d branches, joined '
           'to their nearest ancestor. Columns are generations, oldest on the '
           'left.</text>' % (X0, len(info))]

    # A worked example of a box, in grey, so the caption does not have to
    # describe one in words.
    lx, ly = width - X0 - BOX_W, 34
    out.append('<text x="%.1f" y="%.1f" font-size="9" font-weight="600" '
               'fill="#7A8994">WHAT EACH BOX SHOWS</text>' % (lx, ly - 8))
    out.append('<rect x="%.1f" y="%.1f" width="%d" height="%d" rx="4" '
               'fill="#F4F6F7" stroke="#B7C4CC" stroke-width="1.1" '
               'stroke-dasharray="4 3"/>' % (lx, ly, BOX_W, BOX_H))
    out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="9" '
               'font-weight="600" fill="#5A6A73">branch-name</text>'
               % (lx + 8, ly + 14, MONO))
    out.append('<text x="%.1f" y="%.1f" font-size="8" fill="#7A8994">'
               'what it is, in a few words</text>' % (lx + 8, ly + 26))
    out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7" '
               'fill="#5A6A73">status vs trunk</text>' % (lx + 8, ly + 39, MONO))
    out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7" '
               'fill="#7A8994" text-anchor="end">last commit, author</text>'
               % (lx + BOX_W - 8, ly + 39, MONO))
    out.append('<text x="%.1f" y="%.1f" font-size="8" fill="#7A8994">'
               'green: in dev-next · amber: still out · red: decided against'
               '</text>' % (lx, ly + BOX_H + 13))
    out.append('<text x="%.1f" y="%.1f" font-size="8" fill="#7A8994">'
               'purple: content taken, patch not merged · blue: the trunk'
               '</text>' % (lx, ly + BOX_H + 24))

    out.append('<text x="%d" y="88" font-size="9" font-weight="600" '
               'fill="#5A6A73">INFRASTRUCTURE</text>' % X0)
    iw, gap = 108, 10
    for i, (label, state) in enumerate(WORK):
        x = X0 + i * (iw + gap)
        out.append('<rect x="%.1f" y="98" width="%d" height="22" rx="3" fill="%s" '
                   'stroke="%s" stroke-width="1"/><text x="%.1f" y="113" '
                   'font-family="%s" font-size="9" fill="%s">%s %s</text>'
                   % (x, iw, FILL[state], EDGE[state], x + 7, MONO,
                      TEXT[state], "✓" if state == "done" else "○",
                      esc(label)))

    # The span of last-commit dates in each column, so a reader can see the
    # generations are also eras: the roots stopped in 2023, the rightmost
    # column is this month.
    for gen, names in sorted(columns.items()):
        dates = sorted(info[n]["last"] for n in names)
        span = ("%s/%s" % (dates[0][2:4], dates[0][5:7]) if dates[0] == dates[-1]
                else "%s/%s - %s/%s" % (dates[0][2:4], dates[0][5:7],
                                        dates[-1][2:4], dates[-1][5:7]))
        out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="9" '
                   'font-weight="600" fill="#5A6A73" text-anchor="middle">%s</text>'
                   % (X0 + gen * COL + BOX_W / 2.0, TOP - 14, MONO, esc(span)))

    for name, entry in info.items():
        parent = entry["parent"]
        if not parent or parent not in at:
            continue
        px, py = at[parent]
        cx, cy = at[name]
        # One colour throughout. The line says who came from whom; whether a
        # branch is in or out is the box's colour, and encoding it twice only
        # makes the two compete.
        mid = px + BOX_W + (cx - px - BOX_W) / 2.0
        out.append('<path d="M %.1f %.1f H %.1f V %.1f H %.1f" fill="none" '
                   'stroke="#C3CDD4" stroke-width="1"/>'
                   % (px + BOX_W, py + BOX_H / 2.0, mid, cy + BOX_H / 2.0, cx))

    for name, entry in info.items():
        x, y = at[name]
        label = SHORT.get(name, name)
        if len(label) * CHAR_W > BOX_W - 14:
            raise SystemExit('branch name will not fit its box: %r -- add a '
                             'shorter form to SHORT' % name)
        state = facts.display_state(name, entry)
        out.append('<rect x="%.1f" y="%.1f" width="%d" height="%d" rx="4" '
                   'fill="%s" stroke="%s" stroke-width="%s"/>'
                   % (x, y, BOX_W, BOX_H, FILL[state], EDGE[state],
                      "2" if state == "trunk" else "1.1"))
        out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="9" '
                   'font-weight="600" fill="%s">%s</text>'
                   % (x + 8, y + 14, MONO, TEXT[state], esc(label)))
        out.append('<text x="%.1f" y="%.1f" font-size="8" fill="#5A6A73">%s</text>'
                   % (x + 8, y + 26, esc(DESCRIPTIONS.get(name, ""))))

        if state == "absorbed":
            # Its content is in, its patch is not. Both halves matter.
            note = "content in %s" % facts.DECIDED[name][0][2:]
        elif state == "retired":
            # Decided against, not merely unmerged. The date is the decision's,
            # not a commit's.
            note = "not merging %s" % facts.DECIDED[name][0][2:]
        elif state == "merged":
            # "merged <date> <sha>" only when a merge commit names the branch.
            # Otherwise it is contained but nothing took it by name -- a
            # fast-forward, a squash, a rebase -- and naming the first merge
            # that happens to contain it would credit another branch's merge.
            if entry["merged_on"] == "ancestor":
                note = "merged: ancestor"
            elif entry["merge_named"]:
                note = "merged %s %s" % (entry["merged_on"][2:],
                                         entry["merged_by"])
            else:
                note = "in trunk by %s" % entry["merged_on"][2:]
        elif state == "unmerged":
            note = "%d patch%s out" % (entry["ahead"],
                                       "" if entry["ahead"] == 1 else "es")
        else:
            note = "the trunk"
        out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7" '
                   'fill="%s">%s</text>'
                   % (x + 8, y + 39, MONO, TEXT[state], esc(note)))

        out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7" '
                   'fill="#7A8994" text-anchor="end">%s %s</text>'
                   % (x + BOX_W - 8, y + 39, MONO, esc(entry["last"][2:]),
                      esc(entry["author"][:15])))

    # Where this picture came from. A diagram with stored contents is a
    # measurement, and a measurement without a timestamp is an anecdote.
    script = "docs/dev/make_branch_diagram.py"
    stamp, commit = facts.provenance(script)
    out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7" '
               'fill="#9AA7AF" text-anchor="end">%s</text>'
               % (width - X0, height - 26, MONO,
                  esc("generated %s by %s" % (stamp, script))))
    out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7" '
               'fill="#9AA7AF" text-anchor="end">%s</text>'
               % (width - X0, height - 15, MONO,
                  esc("script at commit %s" % commit)))

    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    svg = build(collect())
    # The diagram lives with the rest of the dev-next paperwork. The copy
    # under docs/ is what Sphinx and the roadmap page include; one script
    # writes both, so the two cannot drift.
    for target in (facts.ROOT / "resources" / "dev" / "dev-next" / "branches.svg",
                   facts.ROOT / "docs" / "source" / "branches.svg"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(svg, encoding="utf-8")
        print("wrote %s" % target, file=sys.stderr)
