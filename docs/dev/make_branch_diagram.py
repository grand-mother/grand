# -*- coding: utf-8 -*-
r"""Draws every branch in the repository against the trunk.

The recovery diagram (``make_recovery_diagram.py``) shows the *merge queue* --
the handful of branches somebody curated for merging.  That is a useful picture
and a misleading one on its own: the queue being empty reads as "everything is
merged", when 38 branches exist and a third of them still carry work.

This one shows all of them.  Left of the spine, contained in ``dev-next``, with
a line saying when and by which merge.  Right of it, still carrying patches of
their own.  Nothing is summarised away.

    python docs/dev/make_branch_diagram.py

Writes ``docs/source/_static/branches.svg``.  Regenerate it after any merge; the
data below is hand-maintained, because a branch's one-line description is a
judgement rather than something git knows.
"""
import pathlib
import subprocess
import sys

# (ref, display name, 2-4 word description).  `ref` is what git knows; the
# display name is shortened where the real one will not fit a box, and the two
# are kept apart so that abbreviating a label cannot break the merge lookup.
#
# `dev-next` itself is deliberately absent: it is the spine, not a box.
MERGED = [
    ("dev_fix_root_warnings_lwp", "dev_fix_root_warnings_lwp", "ROOT 6.38 warnings"),
    ("dev_nutrig_fields", "dev_nutrig_fields", "NUTRIG fields in TADC"),
    ("dev_reprocessing", "dev_reprocessing", "Snakemake pipeline"),
    ("dev_Event_write", "dev_Event_write", "tshower writing"),
    ("dev_aoi_unittest", "dev_aoi_unittest", "aoi unit tests"),
    ("dev_snonis", "dev_snonis", "noise √2 fix"),
    ("dev_fix_root_warnings_lwp_new_fields", "dev_fix_root_warnings_..._new_fields", "NUTRIG name clash"),
    ("dev_fix_root_warnings_aoi_levels_lwp", "dev_fix_root_warnings_aoi_levels_lwp", "levels, +40% speed"),
    ("dev_database", "dev_database", "data catalogue"),
    ("dev_io_root", "dev_io_root", "ROOT I/O layer"),
    ("dev_io_root_testmerges", "dev_io_root_testmerges", "I/O test merges"),
    ("dev_sim2root", "dev_sim2root", "sim2root converters"),
    ("dev_sim2root_merge", "dev_sim2root_merge", "sim2root merge"),
    ("dev_sim2root_merge__merge_with_dev", "dev_sim2root_..._with_dev", "sim2root into dev"),
    ("dev_sim2root_merge__merge_with_dev_fix_fields", "dev_sim2root_..._fix_fields", "sim2root field fixes"),
    ("dev_imports", "dev_imports", "import cleanup"),
    ("ci/docker-test", "ci/docker-test", "docker CI trial"),
    ("event-viewer", "event-viewer", "event viewer"),
    ("copilot/add-color-coded-diagram", "copilot/add-color-coded-diagram", "diagram experiment"),
    ("dev", "dev", "the old trunk"),
    ("master", "master", "old default branch"),
    ("main", "main", "abandoned 2023 trunk"),
]

UNMERGED = [
    ("radio", "radio", "2020 lib/ work"),
    ("refact_galaxy", "refact_galaxy", "rival galaxy refactor"),
    ("dev_marion", "dev_marion", "reconstruction package"),
    ("grandio_light", "grandio_light", "the package split"),
    ("dev_downsample_and_ADCconversion_Jelena", "dev_downsample_..._Jelena", "ADC conversion"),
    ("masterkastner", "masterkastner", "docstrings, old docs"),
    ("beta_dc1", "beta_dc1", "DC1 analysis scripts"),
    ("dc2_debug_xmax", "dc2_debug_xmax", "DC2 xmax debugging"),
    ("dev_leisos", "dev_leisos", "recursive coreas pipeline"),
    ("dev_event_viewer", "dev_event_viewer", "event viewer examples"),
    ("snonis_sim2root_test_merge", "snonis_sim2root_test_merge", "galaxy test notebook"),
    ("147-add-option-to-read-in-non-parallel-coreas-sims-in-sim2root", "147-add-option-...-sim2root", "non-parallel CoREAS"),
    ("tian-conda-arm", "tian-conda-arm", "ARM install notes"),
    ("no-astropy", "no-astropy", "drop astropy"),
    ("dependabot/pip/binder/pillow-9.3.0", "dependabot/...pillow-9.3.0", "abandoned auto-PR"),
]

# The infrastructure track, carried over from the recovery diagram.
WORK = [
    ("conda env", "done"), ("setup.sh", "done"), ("pyproject", "done"),
    ("Sphinx docs", "done"), ("schema test", "done"), ("CI green", "done"),
    ("593 tests", "done"), ("cov 73%", "done"), ("interface", "todo"),
]

FILL = {"merged": "#E1F1EA", "unmerged": "#F6EDDA",
        "done": "#E1F1EA", "todo": "#EDF1F3"}
EDGE = {"merged": "#1D7A57", "unmerged": "#8A6210",
        "done": "#1D7A57", "todo": "#BCC7CE"}
TEXT = {"merged": "#1D7A57", "unmerged": "#8A6210",
        "done": "#1D7A57", "todo": "#7A8994"}

MONO = "IBM Plex Mono, monospace"
SANS = "IBM Plex Sans, Helvetica, Arial, sans-serif"

W = 1180
BOX_W, BOX_H, ROW = 250, 30, 34
SPINE_X = 590
LEAD = 150                      # space between a box and the spine
TOP = 150                       # first branch row
INFRA_Y = 86
CHAR_W = 5.45                   # measured for IBM Plex Mono at 9px


def esc(text):
    r"""Escapes the five XML characters."""
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
                 ('"', "&quot;"), ("'", "&apos;")):
        text = text.replace(a, b)
    return text


def fits(text, width, size=9.0):
    r"""Returns whether `text` fits in `width` px at that monospace size."""
    return len(text) * CHAR_W * (size / 9.0) <= width - 12


def merge_info(branch):
    r"""Returns ``(date, sha)`` for the merge that brought `branch` into the trunk.

    Parameters
    ----------
    branch : str
        Short branch name as it appears on ``origin``.

    Returns
    -------
    tuple of str
        Date as ``YYYY-MM-DD`` and the abbreviated merge commit, or
        ``("ancestor", "")`` when the branch is simply an ancestor with no merge
        of its own on the path, or ``("?", "")`` when it cannot be resolved --
        the display names above are abbreviated, so a lookup can miss.
    """
    ref = "origin/%s" % branch
    probe = subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        return "?", ""
    out = subprocess.run(
        ["git", "log", "--ancestry-path", "--merges", "--reverse",
         "--format=%h|%ad", "--date=short", "%s..dev-next" % ref],
        capture_output=True, text=True).stdout.strip().split("\n")
    out = [line for line in out if line.strip()]
    if not out:
        return "ancestor", ""
    sha, date = out[0].split("|")
    return date, sha


def box(x, y, w, h, state, name, desc):
    r"""Returns the SVG for one branch box."""
    return (
        '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="4" '
        'fill="%s" stroke="%s" stroke-width="1.1"/>'
        '<text x="%.1f" y="%.1f" font-family="%s" font-size="9" '
        'font-weight="600" fill="%s">%s</text>'
        '<text x="%.1f" y="%.1f" font-family="%s" font-size="8" '
        'fill="#5A6A73">%s</text>'
        % (x, y, w, h, FILL[state], EDGE[state],
           x + 8, y + 13, MONO, TEXT[state], esc(name),
           x + 8, y + 24, SANS, esc(desc)))


def build():
    r"""Returns the finished SVG as a string.

    Raises
    ------
    SystemExit
        If any label would overflow its box.  A diagram that silently clips a
        branch name is worse than one that refuses to build.
    """
    height = TOP + max(len(MERGED), len(UNMERGED)) * ROW + 70
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'width="%d" height="%d" font-family="%s">' % (W, height, W, height, SANS),
           '<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, height)]

    out.append('<text x="%d" y="34" font-size="15" font-weight="600" '
               'fill="#22313A">Every branch, against dev-next</text>' % 40)
    out.append('<text x="%d" y="52" font-size="10" fill="#5A6A73">'
               '%d branches. Left: contained in dev-next, with the merge that '
               'brought it in. Right: still carrying patches of its own.'
               '</text>' % (40, len(MERGED) + len(UNMERGED)))

    # --- infrastructure track ------------------------------------------
    out.append('<text x="40" y="%d" font-size="9" font-weight="600" '
               'fill="#5A6A73">INFRASTRUCTURE</text>' % (INFRA_Y - 10))
    iw, gap = 108, 10
    for i, (label, state) in enumerate(WORK):
        x = 40 + i * (iw + gap)
        if not fits(label, iw):
            raise SystemExit('infrastructure label does not fit: %r' % label)
        out.append('<rect x="%.1f" y="%d" width="%d" height="22" rx="3" '
                   'fill="%s" stroke="%s" stroke-width="1"/>'
                   '<text x="%.1f" y="%d" font-family="%s" font-size="9" '
                   'fill="%s">%s %s</text>'
                   % (x, INFRA_Y, iw, FILL[state], EDGE[state],
                      x + 7, INFRA_Y + 15, MONO, TEXT[state],
                      "✓" if state == "done" else "○", esc(label)))

    # --- the spine ------------------------------------------------------
    spine_top = TOP - 22
    spine_bottom = TOP + max(len(MERGED), len(UNMERGED)) * ROW + 4
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#22313A" '
               'stroke-width="2.5"/>' % (SPINE_X, spine_top, SPINE_X, spine_bottom))
    out.append('<text x="%d" y="%d" font-family="%s" font-size="10" '
               'font-weight="600" fill="#22313A" text-anchor="middle">dev-next</text>'
               % (SPINE_X, spine_top - 8, MONO))

    # --- merged, on the left, each joined to the spine -------------------
    for i, (ref, name, desc) in enumerate(MERGED):
        y = TOP + i * ROW
        x = SPINE_X - LEAD - BOX_W
        if not fits(name, BOX_W):
            raise SystemExit('branch name does not fit its box: %r' % name)
        out.append(box(x, y, BOX_W, BOX_H, "merged", name, desc))
        cy = y + BOX_H / 2
        out.append('<line x1="%.1f" y1="%.1f" x2="%d" y2="%.1f" stroke="#1D7A57" '
                   'stroke-width="1.2"/>' % (x + BOX_W, cy, SPINE_X, cy))
        out.append('<circle cx="%d" cy="%.1f" r="3" fill="#1D7A57"/>' % (SPINE_X, cy))
        date, sha = merge_info(ref)
        label = date if not sha else "%s · %s" % (date, sha)
        out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7.5" '
                   'fill="#1D7A57" text-anchor="middle">%s</text>'
                   % (x + BOX_W + LEAD / 2, cy - 4, MONO, esc(label)))

    # --- unmerged, on the right, unattached ------------------------------
    for i, (ref, name, desc) in enumerate(UNMERGED):
        y = TOP + i * ROW
        x = SPINE_X + LEAD
        if not fits(name, BOX_W):
            raise SystemExit('branch name does not fit its box: %r' % name)
        out.append(box(x, y, BOX_W, BOX_H, "unmerged", name, desc))
        cy = y + BOX_H / 2
        out.append('<line x1="%d" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#C9B48A" '
                   'stroke-width="1" stroke-dasharray="3 3"/>'
                   % (SPINE_X, cy, x, cy))

    out.append('<text x="%d" y="%d" font-size="9" fill="#5A6A73">'
               'A dashed line is a branch that has never been merged; the solid '
               'green ones carry the date and commit that brought the branch in.'
               '</text>' % (40, height - 26))
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parents[2]
    target = here / "docs" / "source" / "_static" / "branches.svg"
    target.write_text(build(), encoding="utf-8")
    print("wrote %s" % target, file=sys.stderr)
