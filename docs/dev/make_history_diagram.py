# -*- coding: utf-8 -*-
r"""Draws the whole history of the repository: every branch, every merge.

    python docs/dev/make_history_diagram.py

Writes ``resources/dev/dev-next/history.svg``.

This one is **context, not a working document**. ``branches.svg`` and
``BRANCHES.md`` describe the branches that exist now, which is what the
recovery acts on; this describes how the repository got to where it is, going
back to the first commit in June 2019. Most of what it draws cannot be acted
on, because most of it no longer exists.

Time runs left to right, one row per branch. A bar spans a branch's life, from
its first commit to its last. A thin line entering a bar from above or below is
where the branch forked; one leaving it is a merge, drawn to the branch it
merged into.

Colour says only one thing, and only about branches that still exist: green if
`dev-next` contains it, amber if it does not, blue for `dev-next` itself.
Everything grey was merged and deleted, and asking whether it is merged is not
a useful question about it.

Merge targets are read from merge subjects, which is all a deleted branch
leaves behind. Where a subject does not name one -- a pull request records only
the head branch -- the merge is drawn into whichever trunk was current at the
time.
"""
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import branch_facts as facts                                  # noqa: E402

#: The trunks, oldest first. Each gets a lane of its own at the top.
TRUNKS = ["master", "dev", "main", "dev-next"]

#: Which trunk each was cut from, where there is an answer. This is stated
#: rather than measured because git cannot supply it: ``master`` and ``dev``
#: have merged into each other since 2019, so their merge-base is 2025-03-04
#: and says nothing about when ``dev`` began. ``main`` and ``dev-next`` really
#: were cut from ``dev``, and for those the merge-base is the birthday.
#: A trunk with no entry starts at its oldest commit.
CUT_FROM = {"main": "dev", "dev-next": "dev"}

FILL = {"trunk": "#D5E8F5", "merged": "#E1F1EA", "unmerged": "#F6EDDA",
        "retired": "#F7E6E7", "absorbed": "#EFE9F5", "gone": "#ECEFF1"}
EDGE = {"trunk": "#1F5C82", "merged": "#1D7A57", "unmerged": "#8A6210",
        "retired": "#A9484E", "absorbed": "#6B4E8E", "gone": "#AEBAC2"}
TEXT = {"trunk": "#1F5C82", "merged": "#1D7A57", "unmerged": "#8A6210",
        "retired": "#A9484E", "absorbed": "#6B4E8E", "gone": "#8B99A3"}

MONO = "IBM Plex Mono, monospace"
SANS = "IBM Plex Sans, Helvetica, Arial, sans-serif"

PX_MONTH = 15.5           # horizontal scale
LANE_H = 21               # one branch per lane
TRUNK_H = 26
BAR_H = 9
X0 = 58                   # left edge of the time area
LABEL_PAD = 7
CHAR_W = 4.35             # advance of the 7.2px mono label
PANEL_W = 358             # reserved at the right for the legend panel

ONE_DAY = datetime.timedelta(days=1)


def day(text):
    r"""Returns a ``YYYY-MM-DD`` string as a date, or None."""
    try:
        return datetime.date(*[int(p) for p in text.split("-")])
    except (ValueError, AttributeError):
        return None


def esc(text):
    r"""Escapes the five XML characters."""
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
                 ('"', "&quot;"), ("'", "&apos;")):
        text = text.replace(a, b)
    return text


def trunk_spans():
    r"""Returns ``{name: (start, end)}`` for the trunks that exist.

    A trunk's bar has to cover every merge drawn into it, or the lines land in
    blank space beside it. ``master`` and ``dev`` carried the repository from
    the beginning and span it; ``main`` and ``dev-next`` start where
    :data:`CUT_FROM` says they were cut.

    Returns
    -------
    dict
        Branch name to ``(start, end)`` dates.
    """
    spans = {}
    for name in TRUNKS:
        ref = "origin/%s" % name
        if not facts.git("rev-parse", "-q", "--verify", ref):
            continue
        start = day(facts.git("log", "--reverse", "--format=%ad",
                              "--date=short", ref).split("\n")[0].strip())
        last = day(facts.git("log", "-1", "--format=%ad", "--date=short", ref))
        parent = CUT_FROM.get(name)
        if parent and facts.git("rev-parse", "-q", "--verify",
                                "origin/%s" % parent):
            base = facts.git("merge-base", "origin/%s" % parent, ref)
            cut = day(facts.git("log", "-1", "--format=%ad", "--date=short",
                                base)) if base else None
            if cut:
                start = cut
        if start and last:
            spans[name] = (start, max(last, start))
    return spans


def state_of(name, entry):
    r"""Returns the colour key for a branch.

    Delegates to :func:`branch_facts.display_state` so that this diagram,
    ``branches.svg`` and ``BRANCHES.md`` cannot disagree about what colour a
    branch is.
    """
    return facts.display_state(name, entry)


def layout(info, spans, outgoing):
    r"""Assigns every branch a lane and a horizontal extent.

    Trunks take the top lanes in order. Everything else is packed: a branch
    reuses a lane once the previous occupant's bar *and label* have ended, so
    seventy branches fit in far fewer than seventy rows without any label
    landing on top of another.

    Returns
    -------
    tuple
        ``(rows, lanes)`` -- a list of dicts with ``name``, ``lane``,
        ``start``, ``end`` and ``state``, and the number of packed lanes.
    """
    rows, packed = [], []
    for i, name in enumerate([n for n in TRUNKS if n in spans]):
        start, end = spans[name]
        rows.append(dict(name=name, lane=-len([n for n in TRUNKS
                                               if n in spans]) + i,
                         start=start, end=end, state=state_of(name, info[name]),
                         trunk=True))

    others = []
    for name, entry in info.items():
        if name in spans:
            continue
        start, end = day(entry["created"]), day(entry["last"])
        # A branch lives from its first commit until its last merge, and the
        # recorded dates can fall inside that: a branch merged repeatedly over
        # years has its commits attributed from the last merge alone.
        merges = outgoing.get(name)
        if merges:
            first, last_merge = merges
            if not start or first < start:
                start = first
            if not end or last_merge > end:
                end = last_merge
        if not start:
            # Contained in the trunk with no merge naming it, so nothing says
            # when it began. Drawn as a point at its last commit rather than
            # dropped: the branch existed, and a picture of the history that
            # silently omits thirteen branches is worse than one that admits
            # it cannot date them.
            start = end
        if not start or not end:
            continue
        others.append((start, end, name, entry))
    others.sort(key=lambda t: (t[0], t[1]))

    for start, end, name, entry in others:
        label = facts.SHORT.get(name, name)
        width = max((end - start).days / 30.4 * PX_MONTH, 5)
        needed = width + LABEL_PAD + len(label) * CHAR_W + 14
        for lane, used in enumerate(packed):
            if used <= start_px(start) - 6:
                packed[lane] = start_px(start) + needed
                break
        else:
            lane = len(packed)
            packed.append(start_px(start) + needed)
        rows.append(dict(name=name, lane=lane, start=start, end=end,
                         state=state_of(name, entry), trunk=False,
                         dated=bool(entry["created"])))

    # The packer is supposed to make this impossible. Checking it anyway,
    # because a label sitting on top of another is exactly the kind of defect
    # that survives review: it looks like a font problem, not a bug.
    seen = {}
    for row in rows:
        if row["trunk"]:
            continue
        label = facts.SHORT.get(row["name"], row["name"])
        x1 = start_px(row["start"])
        x2 = max(start_px(row["end"]), x1 + 5) + LABEL_PAD + len(label) * CHAR_W
        for other, (ox1, ox2) in seen.get(row["lane"], {}).items():
            if x1 < ox2 and ox1 < x2:
                raise SystemExit(
                    "%r and %r overlap in lane %d (%.0f-%.0f vs %.0f-%.0f); "
                    "the lane packer is wrong"
                    % (row["name"], other, row["lane"], x1, x2, ox1, ox2))
        seen.setdefault(row["lane"], {})[row["name"]] = (x1, x2)
    return rows, len(packed)


#: Filled in by :func:`build` before layout runs; ``start_px`` needs the
#: earliest date and the packer needs ``start_px``.
ORIGIN = None


def start_px(date):
    r"""Returns the x for a date."""
    return X0 + (date - ORIGIN).days / 30.4 * PX_MONTH


def build(info):
    r"""Returns the finished SVG."""
    global ORIGIN

    # A merge whose *source* is a trunk is the trunk being pulled into a
    # branch to catch it up. It is not the branch coming back, and drawing it
    # produced 39 lines running the wrong way. Only merges out of a branch are
    # part of the shape of the history.
    events = [e for e in facts.merge_events() if e["source"] not in TRUNKS]
    # First and last merge out of each branch. A branch demonstrably existed
    # at both, so its bar has to reach them: the merge line leaves from the
    # bar, and a line leaving from blank space beside it reads as a fork that
    # went nowhere.
    outgoing = {}
    for event in events:
        when = day(event["date"])
        if not when:
            continue
        first, last = outgoing.get(event["source"], (when, when))
        outgoing[event["source"]] = (min(first, when), max(last, when))

    spans = trunk_spans()
    ORIGIN = min(s for s, _ in spans.values())
    today = max(e for _, e in spans.values())

    rows, lanes = layout(info, spans, outgoing)
    n_trunk = sum(1 for r in rows if r["trunk"])

    top = 128
    trunk_top = top
    packed_top = trunk_top + n_trunk * TRUNK_H + 16
    height = int(packed_top + lanes * LANE_H + 46)
    width = int(start_px(today) + PANEL_W)

    def lane_y(row):
        if row["trunk"]:
            return trunk_top + (row["lane"] + n_trunk) * TRUNK_H + TRUNK_H / 2
        return packed_top + row["lane"] * LANE_H + LANE_H / 2

    at = {r["name"]: (lane_y(r), r) for r in rows}

    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'width="%d" height="%d" font-family="%s">'
           % (width, height, width, height, SANS),
           '<rect width="%d" height="%d" fill="#FFFFFF"/>' % (width, height),
           '<text x="%d" y="34" font-size="15" font-weight="600" fill="#22313A">'
           'Every branch and merge in grand, %d to %d</text>'
           % (X0, ORIGIN.year, today.year),
           '<text x="%d" y="52" font-size="10" fill="#5A6A73">%d branches over '
           '%d years, oldest at the top. Context for the recovery rather than '
           'part of it: most of these no longer exist.</text>'
           % (X0, len(rows), today.year - ORIGIN.year)]

    # --- year grid --------------------------------------------------------
    for year in range(ORIGIN.year, today.year + 1):
        x = start_px(datetime.date(year, 1, 1)) if year > ORIGIN.year else X0
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#EEF1F3" '
                   'stroke-width="1"/>' % (x, top - 16, x, height - 34))
        out.append('<text x="%.1f" y="%d" font-family="%s" font-size="9" '
                   'font-weight="600" fill="#9AA7AF">%d</text>'
                   % (x + 3, top - 22, MONO, year))

    # --- edges, behind the bars ------------------------------------------
    span_px = {}
    for row in rows:
        x1 = start_px(row["start"])
        span_px[row["name"]] = (x1, max(start_px(row["end"]), x1 + 4))

    # One edge per source-and-target pair, at the last such merge. The raw
    # list has many repeats of the same branch merged into the same place;
    # drawing them all produced a picket fence over the bars it annotates.
    latest = {}
    for event in events:
        latest[(event["source"], event["target"])] = event

    drawn = 0
    for event in latest.values():
        src = at.get(event["source"])
        when = day(event["date"])
        if not src or not when:
            continue
        x = min(max(start_px(when), X0), start_px(today))

        # The named target only serves if its bar is actually there at that
        # moment; a branch merged into another branch that had already ended
        # would otherwise get a line to nowhere. Fall back to the trunk of
        # the day, which by construction is.
        target = event["target"] if event["target"] in span_px else None
        if target:
            low, high = span_px[target]
            if not low - 2 <= x <= high + 2:
                target = None
        if not target:
            target = ("dev-next" if when >= spans["dev-next"][0] else "dev")
        dst = at.get(target)
        if not dst or target == event["source"]:
            continue

        colour = EDGE[src[1]["state"]]
        out.append('<path d="M %.1f %.1f L %.1f %.1f" stroke="%s" '
                   'stroke-width="0.7" fill="none" opacity="0.32"/>'
                   % (x, src[0], x, dst[0], colour))
        out.append('<circle cx="%.1f" cy="%.1f" r="1.9" fill="%s" '
                   'opacity="0.6"/>' % (x, dst[0], colour))
        drawn += 1

    # Branches the trunk contains that no merge commit names -- fast-forwarded,
    # squashed or rebased. They arrived, and the date is known, but nothing
    # records the join. Dashed, because the line is inferred rather than read.
    for row in rows:
        entry = info.get(row["name"], {})
        if (row["trunk"] or entry.get("state") != "merged"
                    or entry.get("merge_named") or row["name"] in outgoing):
            continue
        when = day(entry.get("merged_on", ""))
        if not when:
            continue
        x = min(max(start_px(when), X0), start_px(today))
        dst = at.get("dev-next" if when >= spans["dev-next"][0] else "dev")
        if not dst:
            continue
        out.append('<path d="M %.1f %.1f L %.1f %.1f" stroke="%s" '
                   'stroke-width="0.7" stroke-dasharray="2 2" fill="none" '
                   'opacity="0.42"/>'
                   % (x, at[row["name"]][0], x, dst[0], EDGE[row["state"]]))
        drawn += 1

    # --- bars -------------------------------------------------------------
    for row in rows:
        y = lane_y(row)
        x1, x2 = start_px(row["start"]), start_px(row["end"])
        w = max(x2 - x1, 4)
        state = row["state"]
        h = BAR_H + (2 if row["trunk"] else 0)
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%d" rx="%.1f" '
                   'fill="%s" stroke="%s" stroke-width="1"/>'
                   % (x1, y - h / 2, w, h, h / 2, FILL[state], EDGE[state]))
        label = facts.SHORT.get(row["name"], row["name"])
        size = 8.4 if row["trunk"] else 7.2
        out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="%.1f" '
                   'font-weight="%s" fill="%s">%s</text>'
                   % (x1 + w + LABEL_PAD, y + size / 2 - 0.6, MONO, size,
                      "600" if row["trunk"] else "500", TEXT[state],
                      esc(label)))

    out.append(panel(width, len(rows), drawn))

    script = "docs/dev/make_history_diagram.py"
    stamp, commit = facts.provenance(script)
    out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7" '
               'fill="#9AA7AF" text-anchor="end">%s</text>'
               % (width - 24, height - 20, MONO,
                  esc("generated %s by %s" % (stamp, script))))
    out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7" '
               'fill="#9AA7AF" text-anchor="end">%s</text>'
               % (width - 24, height - 10, MONO,
                  esc("script at commit %s" % commit)))
    out.append("</svg>")
    return "\n".join(out)


def panel(width, n_rows, n_merges):
    r"""Returns the worked example in the top-right corner.

    A caption describing a row in words is harder to follow than a row with
    the parts named, which is why ``branches.svg`` has one of these too.

    The vertical positions are laid out from a single list of bands rather
    than by subtracting from the box height, which is how an earlier version
    came to draw its last note on top of its colour key.
    """
    w, h = 322, 104
    x, y = width - w - 24, 26
    out = ['<text x="%.1f" y="%.1f" font-size="9" font-weight="600" '
           'fill="#7A8994">HOW TO READ A ROW</text>' % (x, y - 6),
           '<rect x="%.1f" y="%.1f" width="%d" height="%d" rx="4" '
           'fill="#F8FAFB" stroke="#B7C4CC" stroke-width="1.1" '
           'stroke-dasharray="4 3"/>' % (x, y, w, h)]

    # The worked row.
    bx, by = x + 46, y + 24
    out.append('<path d="M %.1f %.1f L %.1f %.1f" stroke="#AEBAC2" '
               'stroke-width="0.9" opacity="0.7"/>' % (bx, by - 14, bx, by))
    out.append('<rect x="%.1f" y="%.1f" width="66" height="9" rx="4.5" '
               'fill="#E1F1EA" stroke="#1D7A57" stroke-width="1"/>'
               % (bx, by - 4.5))
    out.append('<text x="%.1f" y="%.1f" font-family="%s" font-size="7.2" '
               'font-weight="500" fill="#1D7A57">branch-name</text>'
               % (bx + 73, by + 2.4, MONO))
    out.append('<path d="M %.1f %.1f L %.1f %.1f" stroke="#1D7A57" '
               'stroke-width="0.9" opacity="0.6"/>'
               % (bx + 66, by, bx + 66, by + 13))
    out.append('<circle cx="%.1f" cy="%.1f" r="1.9" fill="#1D7A57" '
               'opacity="0.8"/>' % (bx + 66, by + 13))
    for label, dx, dy, anchor in (("forked", -4, -16, "end"),
                                  ("life", 33, -8, "middle"),
                                  ("merged in", 72, 17, "start")):
        out.append('<text x="%.1f" y="%.1f" font-size="7.4" fill="#8B99A3" '
                   'text-anchor="%s">%s</text>'
                   % (bx + dx, by + dy, anchor, label))

    for i, line in enumerate((
            'dashed: in the trunk, but no merge commit names it',
            'amber with no line out: forked and never came back')):
        out.append('<text x="%.1f" y="%.1f" font-size="7.4" fill="#8B99A3">%s'
                   '</text>' % (x + 10, y + 52 + i * 10, line))

    # Two rows of swatches: five of them will not fit across this box, and
    # shortening the labels to make them fit loses the distinction the colours
    # exist to draw.
    rows = (("merged", "in dev-next"), ("unmerged", "still out, undecided"),
            ("retired", "decided against")), \
           (("absorbed", "content taken, patch not merged"),
            ("trunk", "the trunk"), ("gone", "merged, then deleted")),
    for r, group in enumerate(rows):
        lx = x + 10
        for state, text in group:
            out.append('<rect x="%.1f" y="%.1f" width="8" height="8" rx="4" '
                       'fill="%s" stroke="%s"/>'
                       % (lx, y + 74 + r * 13, FILL[state], EDGE[state]))
            out.append('<text x="%.1f" y="%.1f" font-size="7.4" fill="#7A8994">'
                       '%s</text>' % (lx + 11, y + 81 + r * 13, text))
            lx += 22 + len(text) * 4.05
    return "\n".join(out)


if __name__ == "__main__":
    target = facts.ROOT / "resources" / "dev" / "dev-next" / "history.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(facts.collect(include_historical=True)),
                      encoding="utf-8")
    print("wrote %s" % target, file=sys.stderr)
