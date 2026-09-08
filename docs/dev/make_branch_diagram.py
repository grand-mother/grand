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

Writes ``docs/source/_static/branches.svg``. Everything except the one-line
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
import subprocess
import sys

#: Two-to-four words per branch. The only hand-maintained data here; git knows
#: the rest. A branch with no entry is drawn with an empty description rather
#: than omitted, so a new one appears as soon as it is pushed.
DESCRIPTIONS = {
    "dev-next": "the new trunk",
    "dev": "the old trunk",
    "master": "old default branch",
    "main": "abandoned 2023 trunk",
    "event-viewer": "event viewer",
    "dev_fix_root_warnings_lwp": "ROOT 6.38 warnings",
    "dev_fix_root_warnings_lwp_new_fields": "NUTRIG name clash",
    "dev_fix_root_warnings_aoi_levels_lwp": "levels, +40% speed",
    "dev_nutrig_fields": "NUTRIG fields in TADC",
    "dev_reprocessing": "Snakemake pipeline",
    "dev_Event_write": "tshower writing",
    "dev_aoi_unittest": "aoi unit tests",
    "dev_snonis": "noise sqrt2 fix",
    "dev_database": "data catalogue",
    "dev_io_root": "ROOT I/O layer",
    "dev_io_root_testmerges": "I/O test merges",
    "dev_sim2root": "sim2root converters",
    "dev_sim2root_merge": "sim2root merge",
    "dev_sim2root_merge__merge_with_dev": "sim2root into dev",
    "dev_sim2root_merge__merge_with_dev_fix_fields": "sim2root field fixes",
    "dev_imports": "import cleanup",
    "ci/docker-test": "docker CI trial",
    "copilot/add-color-coded-diagram": "diagram experiment",
    "radio": "2020 lib/ work",
    "refact_galaxy": "rival galaxy refactor",
    "dev_marion": "reconstruction package",
    "grandio_light": "the package split",
    "dev_downsample_and_ADCconversion_Jelena": "ADC conversion",
    "masterkastner": "docstrings, old docs",
    "beta_dc1": "DC1 analysis scripts",
    "dc2_debug_xmax": "DC2 xmax debugging",
    "dev_leisos": "recursive coreas pipeline",
    "dev_event_viewer": "event viewer examples",
    "snonis_sim2root_test_merge": "galaxy test notebook",
    "147-add-option-to-read-in-non-parallel-coreas-sims-in-sim2root":
        "non-parallel CoREAS",
    "tian-conda-arm": "ARM install notes",
    "no-astropy": "drop astropy",
    "dependabot/pip/binder/pillow-9.3.0": "abandoned auto-PR",
}

#: Names too long for a box. Display only -- git is always asked about the real
#: name, so an abbreviation here cannot break a lookup, which is a mistake the
#: first version of this script made.
SHORT = {
    "dev_fix_root_warnings_lwp_new_fields": "dev_fix_root_..._new_fields",
    "dev_fix_root_warnings_aoi_levels_lwp": "dev_fix_root_..._aoi_levels",
    "dev_sim2root_merge__merge_with_dev": "dev_sim2root_..._with_dev",
    "dev_sim2root_merge__merge_with_dev_fix_fields": "dev_sim2root_..._fix_flds",
    "dev_downsample_and_ADCconversion_Jelena": "dev_downsample_..._Jelena",
    "147-add-option-to-read-in-non-parallel-coreas-sims-in-sim2root":
        "147-non-parallel-coreas",
    "dependabot/pip/binder/pillow-9.3.0": "dependabot/...pillow",
    "copilot/add-color-coded-diagram": "copilot/add-color-coded",
}

WORK = [("conda env", "done"), ("setup.sh", "done"), ("pyproject", "done"),
        ("Sphinx docs", "done"), ("schema test", "done"), ("CI green", "done"),
        ("593 tests", "done"), ("cov 73%", "done"), ("interface", "todo")]

FILL = {"trunk": "#D5E8F5", "merged": "#E1F1EA", "unmerged": "#F6EDDA",
        "done": "#E1F1EA", "todo": "#EDF1F3"}
EDGE = {"trunk": "#1F5C82", "merged": "#1D7A57", "unmerged": "#8A6210",
        "done": "#1D7A57", "todo": "#BCC7CE"}
TEXT = {"trunk": "#1F5C82", "merged": "#1D7A57", "unmerged": "#8A6210",
        "done": "#1D7A57", "todo": "#7A8994"}

MONO = "IBM Plex Mono, monospace"
SANS = "IBM Plex Sans, Helvetica, Arial, sans-serif"
BOX_W, BOX_H = 232, 46
COL, ROW = 292, 54
X0, TOP = 40, 168
CHAR_W = 5.45


def git(*args):
    r"""Returns the stdout of a git command, stripped."""
    return subprocess.run(["git"] + list(args), capture_output=True,
                          text=True).stdout.strip()


def esc(text):
    r"""Escapes the five XML characters."""
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
                 ('"', "&quot;"), ("'", "&apos;")):
        text = text.replace(a, b)
    return text


def collect():
    r"""Reads every branch from git and works out how they are related.

    Returns
    -------
    dict
        Branch name to a dict carrying ``last``, ``author``, ``parent``,
        ``generation``, ``state``, ``ahead``, ``merged_on`` and ``merged_by``.
    """
    heads = [h for h in git("for-each-ref", "--format=%(refname:short)",
                            "refs/remotes/origin").split()
             if h not in ("origin/HEAD", "origin")]

    info, tip = {}, {}
    for head in heads:
        name = head.replace("origin/", "")
        tip[name] = git("rev-parse", head)
        last, author = (git("log", "-1", "--format=%ad|%an", "--date=short",
                            head).split("|") + ["?"])[:2]
        ahead = sum(1 for line in git("cherry", "dev-next", head).split("\n")
                    if line.startswith("+"))
        state = ("trunk" if name == "dev-next"
                 else "unmerged" if ahead else "merged")
        merged_on = merged_by = ""
        if state == "merged":
            path = [x for x in git("log", "--ancestry-path", "--merges",
                                   "--reverse", "--format=%h|%ad", "--date=short",
                                   "%s..dev-next" % head).split("\n") if x.strip()]
            if path:
                merged_by, merged_on = path[0].split("|")
            else:
                merged_on = "ancestor"
        info[name] = dict(last=last, author=author, state=state, ahead=ahead,
                          merged_on=merged_on, merged_by=merged_by)

    by_sha = {sha: name for name, sha in tip.items()}
    for name in info:
        contained = [by_sha[s] for s in set(git("rev-list", tip[name]).split())
                     & set(by_sha) if by_sha[s] != name]
        info[name]["parent"] = (max(contained, key=lambda n: info[n]["last"])
                                if contained else None)

    depth = {}

    def generation(name, guard=0):
        if name in depth:
            return depth[name]
        parent = info[name]["parent"]
        depth[name] = (0 if parent is None or guard > 40
                       else generation(parent, guard + 1) + 1)
        return depth[name]

    for name in info:
        info[name]["generation"] = generation(name)
    return info


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
    height = TOP + max(len(v) for v in columns.values()) * ROW + 44

    at = {}
    for gen, names in columns.items():
        for i, name in enumerate(names):
            at[name] = (X0 + gen * COL, TOP + i * ROW)

    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'width="%d" height="%d" font-family="%s">'
           % (width, height, width, height, SANS),
           '<rect width="%d" height="%d" fill="#FFFFFF"/>' % (width, height),
           '<text x="%d" y="34" font-size="15" font-weight="600" fill="#22313A">'
           'Every branch, and where it came from</text>' % X0,
           '<text x="%d" y="52" font-size="10" fill="#5A6A73">%d branches. Each '
           'is joined to its nearest ancestor, so columns are generations, oldest '
           'on the left. Green is contained in dev-next, amber still carries '
           'patches, blue is the trunk. Bottom right of each box: last commit and '
           'author.</text>' % (X0, len(info))]

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

    for name, entry in info.items():
        parent = entry["parent"]
        if not parent or parent not in at:
            continue
        px, py = at[parent]
        cx, cy = at[name]
        colour = "#9FCBB6" if entry["state"] != "unmerged" else "#D8C9A6"
        mid = px + BOX_W + (cx - px - BOX_W) / 2.0
        out.append('<path d="M %.1f %.1f H %.1f V %.1f H %.1f" fill="none" '
                   'stroke="%s" stroke-width="1"/>'
                   % (px + BOX_W, py + BOX_H / 2.0, mid, cy + BOX_H / 2.0,
                      cx, colour))

    for name, entry in info.items():
        x, y = at[name]
        label = SHORT.get(name, name)
        if len(label) * CHAR_W > BOX_W - 14:
            raise SystemExit('branch name will not fit its box: %r -- add a '
                             'shorter form to SHORT' % name)
        state = entry["state"]
        out.append('<rect x="%.1f" y="%.1f" width="%d" height="%d" rx="4" '
                   'fill="%s" stroke="%s" stroke-width="%s"/>'
                   % (x, y, BOX_W, BOX_H, FILL[state], EDGE[state],
                      "2" if state == "trunk" else "1.1"))
        out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="9" '
                   'font-weight="600" fill="%s">%s</text>'
                   % (x + 8, y + 14, MONO, TEXT[state], esc(label)))
        out.append('<text x="%.1f" y="%.1f" font-size="8" fill="#5A6A73">%s</text>'
                   % (x + 8, y + 26, esc(DESCRIPTIONS.get(name, ""))))

        if state == "merged":
            note = ("merged: ancestor" if entry["merged_on"] == "ancestor"
                    else "merged %s %s" % (entry["merged_on"][2:], entry["merged_by"]))
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

    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[2]
    target = root / "docs" / "source" / "_static" / "branches.svg"
    target.write_text(build(collect()), encoding="utf-8")
    print("wrote %s" % target, file=sys.stderr)
