#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Draws the dev-next recovery diagram.

The state lives in `STATE` below and the geometry is computed, so updating
the picture means editing one list rather than moving shapes.  Regenerate
after any merge::

    python docs/dev/make_recovery_diagram.py

Writes ``docs/source/_static/recovery.svg``.  Colours follow the recovery
plan: green done, amber blocked or in progress, grey not started.
"""

import os

# (branch, one-line description, state) -- state in {'done','blocked','todo'}
QUEUE = [
    ('dev_fix_root_warnings_lwp',      'ROOT 6.38 warnings',      'done'),
    ('dev_nutrig_fields',              'NUTRIG fields in TADC',   'done'),
    ('dev_reprocessing',               'Snakemake pipeline',      'done'),
    ('dev_Event_write',                'tshower writing',         'done'),
    ('..._lwp_new_fields',             'name clash: NUTRIG',      'blocked'),
    ('..._aoi_levels_lwp',             'levels, +40% speed',      'blocked'),
    ('dev_snonis',                     'noise √2 fix',            'blocked'),
    ('dev_database',                   'datamanager conflict',    'blocked'),
]

# (label, state) for the infrastructure track
WORK = [
    ('conda env',      'done'),
    ('setup.sh',       'done'),
    ('pyproject',      'done'),
    ('Sphinx tree',    'done'),
    ('schema test',    'done'),
    ('noise test',     'done'),
    ('CI restore',     'todo'),
    ('interface',      'todo'),
]

FILL = {'done': '#E1F1EA', 'blocked': '#F6EDDA', 'todo': '#EDF1F3'}
EDGE = {'done': '#1D7A57', 'blocked': '#8A6210', 'todo': '#BCC7CE'}
TEXT = {'done': '#1D7A57', 'blocked': '#8A6210', 'todo': '#7A8994'}
MARK = {'done': '✓', 'blocked': '✗', 'todo': '○'}

W, H = 1080, 620
SPINE_Y = 392
X0, X1 = 90, 990
BOX_W, BOX_H = 204, 40
COLS = 4
COL_GAP = (X1 - X0 - BOX_W) / (COLS - 1)

# Approximate advance width of IBM Plex Mono, as a fraction of font size.
# Used to keep labels inside their boxes: the diagram is regenerated as the
# state changes, so a label that fits today must not overflow tomorrow.
MONO_ADV = 0.60


def fit(text, width, size):
    r"""Returns `text` shortened with an ellipsis to fit `width` at `size`."""
    budget = int(width / (size * MONO_ADV))
    return text if len(text) <= budget else text[:max(1, budget - 1)] + '\u2026'


def esc(s):
    r"""Returns `s` with XML metacharacters escaped."""
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def box(x, y, w, h, state, lines, small=False):
    r"""Returns the SVG for one labelled box."""
    out = ['<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="4" '
           'fill="%s" stroke="%s" stroke-width="1.2"/>'
           % (x, y, w, h, FILL[state], EDGE[state])]
    fs = 9.5 if small else 10.5
    ty = y + (h - len(lines) * (fs + 2.5)) / 2 + fs
    for i, line in enumerate(lines):
        line = fit(line, w - 12, fs if i == 0 else fs - 0.8)
        weight = '600' if i == 0 else '400'
        fill = TEXT[state] if i == 0 else '#4C5C69'
        out.append('<text x="%.1f" y="%.1f" font-family="IBM Plex Mono, '
                   'monospace" font-size="%.1f" font-weight="%s" fill="%s" '
                   'text-anchor="middle">%s</text>'
                   % (x + w / 2, ty + i * (fs + 2.5), fs if i == 0 else fs - 0.8,
                      weight, fill, esc(line)))
    return '\n'.join(out)


def main():
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'width="100%%" font-family="IBM Plex Sans, sans-serif">' % (W, H)]
    # An explicit light surface.  The labels are dark by design, and the SVG
    # is embedded both in the Sphinx docs (light) and in the recovery plan
    # (which follows the reader's theme); without a background of its own the
    # text would disappear against a dark page.
    s.append('<rect width="%d" height="%d" rx="6" fill="#FBFCFC" '
             'stroke="#D8DEE3"/>' % (W, H))

    # --- title and legend -------------------------------------------------
    s.append('<text x="%d" y="34" font-size="15" font-weight="600" '
             'fill="#14202A">dev-next recovery</text>' % X0)
    lx = X0
    for state, label in [('done', 'done'), ('blocked', 'blocked'),
                         ('todo', 'not started')]:
        s.append('<rect x="%d" y="46" width="9" height="9" rx="2" fill="%s" '
                 'stroke="%s"/>' % (lx, FILL[state], EDGE[state]))
        s.append('<text x="%d" y="54.5" font-size="10" fill="#4C5C69">%s</text>'
                 % (lx + 14, label))
        lx += 26 + len(label) * 5.6

    # --- infrastructure track (above the spine) ---------------------------
    s.append('<text x="%d" y="92" font-size="10.5" font-weight="600" '
             'fill="#0E6E70" letter-spacing="0.08em">INFRASTRUCTURE</text>' % X0)
    iw = (X1 - X0) / len(WORK)
    for i, (label, state) in enumerate(WORK):
        x = X0 + i * iw
        s.append(box(x, 104, iw - 10, 30, state, ['%s %s' % (MARK[state], label)],
                     small=True))

    # --- the merge queue, in two rows: merged above, blocked below -------
    s.append('<text x="%d" y="176" font-size="10.5" font-weight="600" '
             'fill="#0E6E70" letter-spacing="0.08em">MERGE QUEUE</text>' % X0)

    merged = [q for q in QUEUE if q[2] == 'done']
    stuck = [q for q in QUEUE if q[2] != 'done']
    rows = [(merged, 190, 'merged'), (stuck, 268, 'blocked')]

    joins = []
    for items, top, _ in rows:
        for i, (name, desc, state) in enumerate(items):
            cx = X0 + i * COL_GAP + BOX_W / 2
            s.append(box(cx - BOX_W / 2, top, BOX_W, BOX_H, state, [name, desc]))
            joins.append((cx, top + BOX_H, state))

    # Connectors run from each box down to its node on the spine.  Merged
    # boxes meet the spine directly below themselves; blocked ones meet it
    # further right, past the point work has reached.
    # Spine nodes are spaced evenly across the whole spine, independent of
    # where the boxes sit: the merged boxes already span the full width, so
    # dropping each node directly below its box would leave no spine for the
    # blocked ones.  Order along the spine is merge order.
    n_done = len(merged)
    total = len(joins)
    node_step = (X1 - X0) / total
    node_x = [X0 + (i + 0.5) * node_step for i in range(total)]
    done_x = node_x[n_done - 1] + node_step / 2 if n_done else X0

    for k, (cx, y_from, state) in enumerate(joins):
        nx = node_x[k]
        colour = EDGE[state]
        dash = '' if state == 'done' else ' stroke-dasharray="3 3" opacity="0.8"'
        elbow = SPINE_Y - (30 if state != 'done' else 46)
        s.append('<path d="M %.1f %.1f L %.1f %.1f L %.1f %.1f L %.1f %.1f" '
                 'stroke="%s" stroke-width="1.3" fill="none"%s/>'
                 % (cx, y_from, cx, elbow, nx, elbow, nx, SPINE_Y - 7,
                    colour, dash))
        s.append('<circle cx="%.1f" cy="%d" r="4.5" fill="%s" stroke="%s" '
                 'stroke-width="1.4"/>' % (nx, SPINE_Y, FILL[state], EDGE[state]))
        s.append('<text x="%.1f" y="%d" font-size="8" font-weight="700" '
                 'fill="%s" text-anchor="middle">%s</text>'
                 % (nx, SPINE_Y + 2.6, TEXT[state], MARK[state]))

    # --- the spine --------------------------------------------------------
    s.append('<line x1="%d" y1="%d" x2="%.1f" y2="%d" stroke="#1D7A57" '
             'stroke-width="3"/>' % (X0 - 40, SPINE_Y, done_x, SPINE_Y))
    s.append('<line x1="%.1f" y1="%d" x2="%d" y2="%d" stroke="#BCC7CE" '
             'stroke-width="3" stroke-dasharray="5 4"/>'
             % (done_x, SPINE_Y, X1 + 30, SPINE_Y))
    s.append('<circle cx="%d" cy="%d" r="6" fill="#0E6E70"/>' % (X0 - 40, SPINE_Y))
    s.append('<text x="%d" y="%d" font-size="9.5" font-family="IBM Plex Mono, '
             'monospace" fill="#4C5C69" text-anchor="middle">dev@1ca1847</text>'
             % (X0 - 40, SPINE_Y + 22))
    s.append('<path d="M %d %d l -10 -5.5 l 0 11 z" fill="#7A8994"/>'
             % (X1 + 40, SPINE_Y))
    s.append('<text x="%d" y="%d" font-size="11" font-weight="600" '
             'fill="#4C5C69" text-anchor="middle">main</text>'
             % (X1 + 8, SPINE_Y - 14))

    # --- phase strip (below) ---------------------------------------------
    s.append('<text x="%d" y="470" font-size="10.5" font-weight="600" '
             'fill="#0E6E70" letter-spacing="0.08em">PHASES</text>' % X0)
    phases = [('0 branch', 'done'), ('1 env', 'done'), ('2 CI', 'todo'),
              ('3 tests', 'blocked'), ('4 merge', 'blocked'),
              ('5 decide', 'todo'), ('6 interface', 'todo'), ('7 docs', 'blocked'),
              ('8 govern', 'todo'), ('9 promote', 'todo'), ('10 cleanup', 'todo')]
    pw = (X1 - X0) / len(phases)
    for i, (label, state) in enumerate(phases):
        x = X0 + i * pw
        s.append('<rect x="%.1f" y="482" width="%.1f" height="24" rx="3" '
                 'fill="%s" stroke="%s"/>' % (x, pw - 5, FILL[state], EDGE[state]))
        s.append('<text x="%.1f" y="498" font-size="9" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % (x + (pw - 5) / 2, TEXT[state], esc(label)))

    # --- the two decisions blocking the queue -----------------------------
    s.append('<text x="%d" y="538" font-size="10.5" font-weight="600" '
             'fill="#8A6210" letter-spacing="0.08em">BLOCKED ON A DECISION</text>'
             % X0)
    notes = ['NUTRIG field names: nutrig_rhox/rhoy or correlation_x/y — needs lwpiotr',
             'Galactic noise: dev_snonis √2 fix, or the refact_galaxy rewrite — needs the collaboration']
    for i, note in enumerate(notes):
        s.append('<text x="%d" y="%d" font-size="10.5" fill="#4C5C69">• %s</text>'
                 % (X0, 558 + i * 17, esc(note)))

    s.append('</svg>')
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'source', '_static', 'recovery.svg')
    with open(out, 'w') as handle:
        handle.write('\n'.join(s) + '\n')
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
