#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Draws the data model for the documentation.

    python docs/dev/make_datamodel_diagram.py -> docs/source/_static/datamodel.svg

Two things the prose has trouble carrying.  First, that run-level and
event-level trees are joined by different keys, so a mismatch is silent rather
than an error.  Second, that DataDirectory groups by tree type and analysis
level and *not* by run, which is the trap notebook 02 works through.
"""

import os

W, H = 1000, 560
INK, SOFT, FAINT = '#14202A', '#4C5C69', '#7A8994'
RULE = '#D8DEE3'
RUN_FILL, RUN_EDGE = '#F6EDDA', '#8A6210'
EVT_FILL, EVT_EDGE = '#E1F1EA', '#1D7A57'
SIM_FILL, SIM_EDGE = '#E2F0F0', '#0E6E70'
WARN_FILL, WARN_EDGE = '#FBEDE9', '#B4472A'


def esc(text):
    r"""Returns `text` with the XML metacharacters escaped."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def box(x, y, w, h, fill, edge, title, lines):
    r"""Returns one labelled box with a title and body lines."""
    out = ['<rect x="%d" y="%d" width="%d" height="%d" rx="5" fill="%s" '
           'stroke="%s" stroke-width="1.4"/>' % (x, y, w, h, fill, edge)]
    out.append('<text x="%d" y="%d" font-family="IBM Plex Mono, monospace" '
               'font-size="12.5" font-weight="600" fill="%s" '
               'text-anchor="middle">%s</text>'
               % (x + w // 2, y + 21, edge, esc(title)))
    for i, line in enumerate(lines):
        out.append('<text x="%d" y="%d" font-size="9.5" fill="%s" '
                   'text-anchor="middle">%s</text>'
                   % (x + w // 2, y + 38 + i * 12.5, SOFT, esc(line)))
    return '\n'.join(out)


def arrow(x1, y1, x2, y2, label='', colour=None, dashed=False, dy=-7):
    r"""Returns an arrow, optionally labelled and dashed."""
    colour = colour or FAINT
    d = ' stroke-dasharray="5 4"' if dashed else ''
    parts = ['<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.6"%s marker-end="url(#a)"/>'
             % (x1, y1, x2, y2, colour, d)]
    if label:
        parts.append('<text x="%.0f" y="%.0f" font-size="9.5" fill="%s" '
                     'text-anchor="middle">%s</text>'
                     % ((x1 + x2) / 2, (y1 + y2) / 2 + dy, colour, esc(label)))
    return '\n'.join(parts)


def main():
    r"""Writes the diagram to ``docs/source/_static/datamodel.svg``."""
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'width="%d" height="%d" font-family="Inter, Helvetica, Arial, '
         'sans-serif">' % (W, H, W, H)]
    s.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker></defs>' % FAINT)
    s.append('<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, H))

    s.append('<text x="30" y="34" font-size="15" font-weight="600" fill="%s">'
             'The data model</text>' % INK)
    s.append('<text x="30" y="54" font-size="11" fill="%s">'
             'Run-level and event-level trees are joined by different keys. A '
             'mismatch is silent, not an error.</text>' % SOFT)

    # Run level.
    s.append('<text x="30" y="92" font-size="10.5" font-weight="600" '
             'fill="%s">RUN LEVEL — one entry per run</text>' % RUN_EDGE)
    s.append(box(30, 104, 210, 74, RUN_FILL, RUN_EDGE, 'TRun',
                 ['du_id, du_xyz, t_bin_size', 'origin_geoid', 'software_version']))
    s.append(box(262, 104, 210, 74, RUN_FILL, RUN_EDGE, 'TRunEfieldSim',
                 ['simulation settings', 'for the run']))

    # Event level.
    s.append('<text x="30" y="228" font-size="10.5" font-weight="600" '
             'fill="%s">EVENT LEVEL — one entry per (run, event)</text>' % EVT_EDGE)
    s.append(box(30, 240, 210, 82, EVT_FILL, EVT_EDGE, 'TEfield',
                 ['trace (du, 3, n) uV/m', 'du_id, du_seconds', 'du_nanoseconds']))
    s.append(box(262, 240, 210, 82, EVT_FILL, EVT_EDGE, 'TVoltage',
                 ['trace (du, 3, n) uV', 'after antenna + chain']))
    s.append(box(494, 240, 210, 82, EVT_FILL, EVT_EDGE, 'TADC',
                 ['trace, ADC counts', 'what a unit records']))
    s.append(box(726, 240, 210, 82, SIM_FILL, SIM_EDGE, 'TShower',
                 ['zenith, azimuth', 'energy_primary', 'xmax_pos_shc']))

    # Keys.
    # The label goes beside the line, not on top of it.
    for x in (135, 367):
        s.append(arrow(x, 178, x, 238))
        s.append('<text x="%d" y="214" font-size="9.5" fill="%s">run_number'
                 '</text>' % (x + 8, RUN_EDGE))

    s.append('<line x1="240" y1="281" x2="262" y2="281" stroke="%s" '
             'stroke-width="1.6" marker-end="url(#a)"/>' % EVT_EDGE)
    s.append('<line x1="472" y1="281" x2="494" y2="281" stroke="%s" '
             'stroke-width="1.6" marker-end="url(#a)"/>' % EVT_EDGE)
    s.append('<line x1="704" y1="281" x2="726" y2="281" stroke="%s" '
             'stroke-width="1.6" stroke-dasharray="5 4" '
             'marker-end="url(#a)" marker-start="url(#a)"/>' % SIM_EDGE)
    s.append('<text x="715" y="336" font-size="9.5" fill="%s" '
             'text-anchor="middle">(run_number, event_number)</text>' % SIM_EDGE)

    # The traps.
    s.append('<line x1="30" y1="368" x2="970" y2="368" stroke="%s"/>' % RULE)
    s.append('<text x="30" y="392" font-size="12" font-weight="600" fill="%s">'
             'Three things that are not written down anywhere else</text>' % WARN_EDGE)

    s.append(box(30, 406, 296, 108, WARN_FILL, WARN_EDGE, 'grouping',
                 ['DataDirectory groups by tree type', 'and analysis level — not by run.',
                  'Two runs in a directory do not', 'give two handles.']))
    s.append(box(352, 406, 296, 108, WARN_FILL, WARN_EDGE, 'levels',
                 ['The _L0_ / _L1_ in a file name must',
                  'match the tree’s analysis_level.', 'Both levels come back; a bare',
                  'attribute follows the highest.']))
    s.append(box(674, 406, 296, 108, WARN_FILL, WARN_EDGE, 'uniqueness',
                 ['Writing a second entry with the same', '(run_number, event_number)',
                  'raises NotUniqueEvent. Each run of a', 'simulation needs its own file.']))

    s.append('<text x="30" y="540" font-size="10.5" fill="%s">'
             'Worked through in notebook 02. The fields themselves are '
             'descriptor-driven: see grand/dataio/descriptors.py.</text>' % FAINT)

    s.append('</svg>')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'source', '_static', 'datamodel.svg')
    with open(out, 'w') as handle:
        handle.write('\n'.join(s) + '\n')
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
