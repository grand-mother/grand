#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Draws where the dev-next recovery has got to, phase by phase.

    python docs/dev/make_recovery_diagram.py

Writes ``resources/dev/dev-next/recovery.svg`` and a copy at
``docs/source/recovery.svg`` -- the plan that embeds it is read both on GitHub
and through Sphinx, which resolve the path differently.

**This used to show the merge queue as well.** It no longer does:
``branches.svg`` and ``BRANCHES.md`` show every branch and its status, in more
detail and read from git rather than from a list somebody maintained. Two
pictures of the same thing is how the merge queue here came to show a decision
as blocked three days after it was settled. This one now answers a question
the other two cannot: *how far along is the repair*.

Everything except the blocked decisions is **parsed out of the plan** --
the phase headings and their ``- [x]`` / ``- [ ]`` checkboxes. Ticking a box
in ``RECOVERY_PLAN.md`` and regenerating is the whole update procedure, and
the diagram cannot claim a phase is finished while the plan says otherwise.
"""

import os
import re

#: The plan, relative to the repository root.
PLAN = os.path.join('resources', 'dev', 'dev-next', 'RECOVERY_PLAN.md')

#: Phases whose remaining work is waiting on a decision rather than on effort.
#: Progress alone cannot tell these apart -- an untouched phase and one that
#: nobody may touch look identical in the checkboxes -- so this is the one
#: judgement in the picture.
BLOCKED_PHASES = {'5'}

#: What those decisions are. Hand-maintained: they are prose, and there is
#: nowhere to read them from. Keep them in step with the plan's *Blocked on a
#: decision* section -- the NUTRIG entry outlived its answer here by three days.
NOTES = [
    'Scope: where reconstruction lives, and whether GRANDlib splits — needs the collaboration',
    'Docker: publish an image on ROOT 6.36, or state that Docker is unsupported — needs the collaboration',
    'Reprocessing: the noise fix raises every simulated voltage by √2 — needs the collaboration',
]

FILL = {'done': '#E1F1EA', 'doing': '#E2F0F0', 'blocked': '#F6EDDA', 'todo': '#EDF1F3'}
EDGE = {'done': '#1D7A57', 'doing': '#0E6E70', 'blocked': '#8A6210', 'todo': '#BCC7CE'}
TEXT = {'done': '#1D7A57', 'doing': '#0E6E70', 'blocked': '#8A6210', 'todo': '#7A8994'}
MARK = {'done': '✓', 'doing': '◐', 'blocked': '✗', 'todo': '○'}

W = 1080
X0, X1 = 90, 990
BOX_W, BOX_H = 204, 58
COLS = 4
COL_GAP = (X1 - X0 - BOX_W) / (COLS - 1)
ROW_TOP, ROW_GAP = 108, 74

#: Names too long for a box at the title size. Display only.
SHORT_NAMES = {
    'delineate input, processing, output': 'delineate I/O',
    'tests before features': 'tests first',
    'governance and weight': 'governance',
}

# Approximate advance width as a fraction of font size, per family. The small
# lines are set in the sans face, which is appreciably narrower than the mono
# one; using the mono figure for both truncated them a third of a line early.
MONO_ADV, SANS_ADV = 0.60, 0.50


def fit(text, width, size, adv=MONO_ADV):
    r"""Returns `text` shortened at a word boundary to fit `width` at `size`.

    Breaking mid-word and appending an ellipsis is worse than dropping the
    word: "Announce the freeze dat…" reads as a typo, not as an abbreviation.
    """
    budget = int(width / (size * adv))
    if len(text) <= budget:
        return text
    cut = text[:budget - 1]
    if ' ' in cut[budget // 2:]:
        cut = cut[:cut.rindex(' ')]
    return cut.rstrip(' ,;:—-') + '…'


def esc(s):
    r"""Returns `s` with XML metacharacters escaped."""
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def plain(s):
    r"""Strips the markdown a plan item carries, leaving readable prose.

    Parameters
    ----------
    s : str
        One checklist item, possibly with backticks, bold, strikethrough or
        links in it.

    Returns
    -------
    str
        The same text with the markup removed and whitespace collapsed.
    """
    s = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', s)
    s = s.replace('**', '').replace('~~', '').replace('`', '')
    s = re.sub(r'\*([^*]+)\*', r'\1', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip(' .—-')


def phases(root):
    r"""Reads the phases and their progress out of the recovery plan.

    Parameters
    ----------
    root : str
        Repository root.

    Returns
    -------
    list of dict
        One entry per phase, in plan order, with ``number``, ``name``,
        ``done``, ``total``, ``state`` and ``next`` (the first open item, or
        an empty string).

    Raises
    ------
    SystemExit
        If the plan cannot be found or has no phase headings.  A diagram drawn
        from an empty parse would be a picture of nothing, drawn confidently.
    """
    path = os.path.join(root, PLAN)
    if not os.path.exists(path):
        raise SystemExit('cannot find the plan at %s' % path)
    with open(path, encoding='utf-8') as handle:
        text = handle.read()

    out = []
    for block in re.split(r'^### Phase ', text, flags=re.M)[1:]:
        heading = block.split('\n', 1)[0]
        body = re.split(r'\n#{2,3} ', block)[0]
        number, _, name = heading.partition('—')
        items = re.findall(r'^- \[([x ])\] (.*(?:\n      .*)*)', body, re.M)
        done = sum(1 for mark, _ in items if mark == 'x')
        total = len(items)
        nxt = next((plain(body_) for mark, body_ in items if mark == ' '), '')
        number = number.strip()
        if total and done == total:
            state = 'done'
        elif number in BLOCKED_PHASES:
            state = 'blocked'
        elif done:
            state = 'doing'
        else:
            state = 'todo'
        out.append(dict(number=number, name=plain(name), done=done,
                        total=total, state=state, next=nxt))
    if not out:
        raise SystemExit('no phase headings found in %s' % path)
    return out


def phase_box(x, y, phase):
    r"""Returns the SVG for one phase: title, progress bar, count and next step.

    Parameters
    ----------
    x, y : float
        Top-left corner.
    phase : dict
        One entry from :func:`phases`.

    Returns
    -------
    str
        The SVG for the box.
    """
    state = phase['state']
    inner = BOX_W - 18
    out = ['<rect x="%.1f" y="%.1f" width="%d" height="%d" rx="4" fill="%s" '
           'stroke="%s" stroke-width="1.2"/>'
           % (x, y, BOX_W, BOX_H, FILL[state], EDGE[state])]

    name = SHORT_NAMES.get(phase['name'], phase['name'])
    title = '%s %s %s' % (MARK[state], phase['number'], name)
    out.append('<text x="%.1f" y="%.1f" font-family="IBM Plex Mono, monospace" '
               'font-size="10" font-weight="600" fill="%s">%s</text>'
               % (x + 9, y + 17, TEXT[state], esc(fit(title, inner, 10))))

    frac = phase['done'] / phase['total'] if phase['total'] else 0.0
    out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="5" rx="2.5" '
               'fill="#FFFFFF" stroke="%s" stroke-width="0.7"/>'
               % (x + 9, y + 25, inner, EDGE[state]))
    if frac:
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="5" rx="2.5" '
                   'fill="%s"/>' % (x + 9, y + 25, inner * frac, EDGE[state]))

    count = ('all %d done' % phase['total'] if frac == 1
             else '%d of %d done' % (phase['done'], phase['total']))
    out.append('<text x="%.1f" y="%.1f" font-size="8.5" font-weight="600" '
               'fill="%s">%s</text>' % (x + 9, y + 43, TEXT[state], count))
    if phase['next']:
        out.append('<text x="%.1f" y="%.1f" font-size="8.5" fill="#7A8994">'
                   'next: %s</text>'
                   % (x + 9, y + 53,
                      esc(fit(phase['next'], inner - 26, 8.5, SANS_ADV))))
    return '\n'.join(out)


def main():
    r"""Writes the diagram to both of its homes."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    data = phases(root)

    rows = (len(data) + COLS - 1) // COLS
    notes_top = ROW_TOP + rows * ROW_GAP + 22
    height = int(notes_top + 26 + len(NOTES) * 17 + 22)

    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'width="100%%" font-family="IBM Plex Sans, sans-serif">' % (W, height)]
    # An explicit light surface: the labels are dark by design, and this is
    # embedded in pages that follow the reader's theme.
    s.append('<rect width="%d" height="%d" rx="6" fill="#FBFCFC" '
             'stroke="#D8DEE3"/>' % (W, height))

    total_done = sum(p['done'] for p in data)
    total_all = sum(p['total'] for p in data)
    s.append('<text x="%d" y="34" font-size="15" font-weight="600" '
             'fill="#14202A">dev-next recovery, by phase</text>' % X0)
    s.append('<text x="%d" y="52" font-size="10" fill="#5A6A73">%d of %d items '
             'done across %d phases. Read from the plan\'s own checkboxes; '
             'every branch and its status is in branches.svg.</text>'
             % (X0, total_done, total_all, len(data)))

    lx = X0
    for state, label in [('done', 'done'), ('doing', 'in progress'),
                         ('blocked', 'blocked on a decision'),
                         ('todo', 'not started')]:
        s.append('<rect x="%d" y="66" width="9" height="9" rx="2" fill="%s" '
                 'stroke="%s"/>' % (lx, FILL[state], EDGE[state]))
        s.append('<text x="%d" y="74.5" font-size="10" fill="#4C5C69">%s</text>'
                 % (lx + 14, label))
        lx += 26 + len(label) * 5.6

    for i, phase in enumerate(data):
        x = X0 + (i % COLS) * COL_GAP
        y = ROW_TOP + (i // COLS) * ROW_GAP
        s.append(phase_box(x, y, phase))

    s.append('<text x="%d" y="%d" font-size="10.5" font-weight="600" '
             'fill="#8A6210" letter-spacing="0.08em">BLOCKED ON A DECISION'
             '</text>' % (X0, notes_top))
    for i, note in enumerate(NOTES):
        s.append('<text x="%d" y="%d" font-size="10.5" fill="#4C5C69">• %s'
                 '</text>' % (X0, notes_top + 19 + i * 17, esc(note)))

    s.append('</svg>')
    svg = '\n'.join(s) + '\n'
    for out in (os.path.join(root, 'resources', 'dev', 'dev-next',
                             'recovery.svg'),
                os.path.join(root, 'docs', 'source', 'recovery.svg')):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as handle:
            handle.write(svg)
        print('wrote %s' % out)


if __name__ == '__main__':
    main()
