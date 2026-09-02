#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Draws the dependencies between the grand subpackages.

    python docs/dev/make_modules_diagram.py -> docs/source/_static/modules.svg

The edges are **measured**, not asserted: ``python docs/dev/make_modules_diagram.py
--measure`` re-derives them by walking the import statements of every module
under ``grand/``, and the counts below come from that.  Run it again after any
change that moves code between subpackages.

What the figure is for is the cycle.  ``geo`` -> ``dataio`` -> ``basis`` ->
``geo`` is a genuine import cycle at module level, and it is not visible from
any one file.
"""

import ast
import collections
import os
import pathlib
import sys

W, H = 940, 620
INK, SOFT, FAINT = '#14202A', '#4C5C69', '#7A8994'
RULE = '#D8DEE3'
CYCLE = '#B4472A'
FILL = {'geo': '#E2F0F0', 'dataio': '#F6EDDA', 'basis': '#E1F1EA',
        'sim': '#E1F1EA', 'aoi': '#EFE9F5', 'recon': '#F2F3F5'}
EDGE = {'geo': '#0E6E70', 'dataio': '#8A6210', 'basis': '#1D7A57',
        'sim': '#1D7A57', 'aoi': '#6B4E9B', 'recon': '#9AA5AE'}

SUB = ('aoi', 'basis', 'dataio', 'geo', 'recon', 'sim')

#: Laid out so that the three edges of the cycle form a visible triangle.
POS = {'geo': (90, 300), 'dataio': (390, 300), 'basis': (240, 130),
       'sim': (640, 130), 'aoi': (640, 300), 'recon': (90, 420)}
BW, BH = 170, 76

#: Measured with --measure; see the module docstring.
EDGES = [('geo', 'dataio', 1, True), ('dataio', 'basis', 1, True),
         ('basis', 'geo', 1, True), ('sim', 'geo', 3, False),
         ('sim', 'basis', 2, False), ('sim', 'dataio', 1, False),
         ('aoi', 'dataio', 3, False), ('aoi', 'geo', 1, False),
         ('basis', 'sim', 1, False)]

NOTES = {
    'geo': 'coordinates, topography,\ngeomagnet — frames and Earth',
    'dataio': 'the ROOT schema:\nTRun, TEfield, TVoltage',
    'basis': 'traces, signals,\nDU networks',
    'sim': 'antenna, RF chain,\nnoise, ADC',
    'aoi': 'events and antennas\nas objects',
    'recon': 'placeholder:\nno algorithm',
}


def measure():
    r"""Prints the subpackage import graph derived from the source."""
    edges = collections.Counter()

    def pkg(mod):
        if not mod or not mod.startswith('grand'):
            return None
        parts = mod.split('.')
        return parts[1] if len(parts) > 1 and parts[1] in SUB else None

    for path in sorted(pathlib.Path('grand').rglob('*.py')):
        if '__pycache__' in str(path):
            continue
        parts = path.parts
        src = parts[1] if len(parts) > 2 and parts[1] in SUB else 'grand'
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                dst = pkg(name)
                if dst and dst != src and src in SUB:
                    edges[(src, dst)] += 1
    for (a, b), n in sorted(edges.items(), key=lambda kv: -kv[1]):
        print('  %-8s -> %-8s %d' % (a, b, n))


def esc(text):
    r"""Returns `text` with the XML metacharacters escaped."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def label(x, y, text, colour, size=10, anchor='start', weight='400', mono=False):
    r"""Returns one text element."""
    font = ' font-family="IBM Plex Mono, monospace"' if mono else ''
    return ('<text x="%.1f" y="%.1f" font-size="%s" fill="%s" text-anchor="%s" '
            'font-weight="%s"%s>%s</text>'
            % (x, y, size, colour, anchor, weight, font, esc(text)))


def edge_points(a, b):
    r"""Returns the two box-edge points of the segment joining boxes `a` and `b`."""
    (ax, ay), (bx, by) = POS[a], POS[b]
    ax, ay = ax + BW / 2, ay + BH / 2
    bx, by = bx + BW / 2, by + BH / 2
    dx, dy = bx - ax, by - ay
    # Clip each end to its own box, so an arrow never covers a label.
    def clip(cx, cy, dx, dy):
        if dx == 0:
            return cx, cy + (BH / 2) * (1 if dy > 0 else -1)
        t = min(abs((BW / 2) / dx), abs((BH / 2) / dy) if dy else 1e9)
        return cx + dx * t, cy + dy * t
    p1 = clip(ax, ay, dx, dy)
    p2 = clip(bx, by, -dx, -dy)
    return p1, p2


def main():
    r"""Writes the diagram to ``docs/source/_static/modules.svg``."""
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'width="%d" height="%d" font-family="Inter, Helvetica, Arial, '
         'sans-serif">' % (W, H, W, H)]
    s.append('<defs>')
    for name, colour in (('a', FAINT), ('c', CYCLE)):
        s.append('<marker id="%s" viewBox="0 0 10 10" refX="9" refY="5" '
                 'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                 '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>' % (name, colour))
    s.append('</defs>')
    s.append('<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, H))

    s.append(label(28, 34, 'Subpackage dependencies', INK, 15, weight='600'))
    s.append(label(28, 54, 'Measured from the import statements. An arrow means '
                   '"imports from"; the number is how many statements.', SOFT, 11))

    # Edges first, so the boxes sit on top of them.
    for a, b, n, in_cycle in EDGES:
        (x1, y1), (x2, y2) = edge_points(a, b)
        colour = CYCLE if in_cycle else FAINT
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                 'stroke-width="%s" marker-end="url(#%s)"%s/>'
                 % (x1, y1, x2, y2, colour, 2.0 if in_cycle else 1.4,
                    'c' if in_cycle else 'a',
                    ' stroke-dasharray="5 4"' if (a, b) == ('basis', 'sim') else ''))
        s.append(label((x1 + x2) / 2, (y1 + y2) / 2 - 5, str(n), colour, 9.5,
                       anchor='middle', weight='600'))

    for name, (x, y) in POS.items():
        s.append('<rect x="%d" y="%d" width="%d" height="%d" rx="5" fill="%s" '
                 'stroke="%s" stroke-width="1.4"/>'
                 % (x, y, BW, BH, FILL[name], EDGE[name]))
        s.append(label(x + BW / 2, y + 24, 'grand.' + name, EDGE[name], 12.5,
                       anchor='middle', weight='600', mono=True))
        for i, line in enumerate(NOTES[name].split('\n')):
            s.append(label(x + BW / 2, y + 42 + i * 13, line, SOFT, 8.5,
                           anchor='middle'))

    # --- legend and the point of the figure ---------------------------------
    s.append('<line x1="28" y1="530" x2="912" y2="530" stroke="%s"/>' % RULE)
    s.append('<line x1="28" y1="552" x2="60" y2="552" stroke="%s" '
             'stroke-width="2" marker-end="url(#c)"/>' % CYCLE)
    s.append(label(70, 556, 'part of a cycle:', CYCLE, 10.5, weight='600'))
    s.append(label(168, 556, 'geo → dataio → basis → geo, all at module level. '
                   'Nothing breaks today, but import order decides whether it '
                   'stays that way.', INK, 10.5))
    s.append('<line x1="28" y1="576" x2="60" y2="576" stroke="%s" '
             'stroke-width="1.4" stroke-dasharray="5 4" '
             'marker-end="url(#a)"/>' % FAINT)
    s.append(label(70, 580, 'deferred:', SOFT, 10.5, weight='600'))
    s.append(label(132, 580, 'basis/pipeline.py imports grand.sim inside a '
                   'function, which is what keeps that second cycle from '
                   'existing at import time.', SOFT, 10.5))
    s.append(label(28, 604, 'grand.recon has no incoming or outgoing edges: '
                   'nothing imports it and it imports nothing.', FAINT, 10))

    s.append('</svg>')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'source', '_static', 'modules.svg')
    with open(out, 'w') as handle:
        handle.write('\n'.join(s) + '\n')
    print('wrote %s' % out)


if __name__ == '__main__':
    if '--measure' in sys.argv:
        measure()
    else:
        main()
