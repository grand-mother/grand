#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Draws the simulation pipeline for the documentation.

    python docs/dev/make_pipeline_diagram.py  ->  docs/source/_static/pipeline.svg

The point of the figure is the two boundaries, not the arrow.  GRANDlib does
not simulate the shower and does not reconstruct the event; what it owns is the
instrument response in between, and the schema on both sides.  Everything that
is somebody else's code is drawn outside the shaded band.
"""

import os

W, H = 1000, 520
INK, SOFT, FAINT = '#14202A', '#4C5C69', '#7A8994'
RULE = '#D8DEE3'
OUT_FILL, OUT_EDGE = '#F2F3F5', '#9AA5AE'      # outside GRANDlib
IN_FILL, IN_EDGE = '#E1F1EA', '#1D7A57'        # the instrument response
IO_FILL, IO_EDGE = '#F6EDDA', '#8A6210'        # files and trees
BAND = '#F4FAF7'


def esc(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def box(x, y, w, h, fill, edge, title, lines, mono=True):
    r"""Returns one labelled box."""
    font = 'IBM Plex Mono, monospace' if mono else 'inherit'
    out = ['<rect x="%d" y="%d" width="%d" height="%d" rx="5" fill="%s" '
           'stroke="%s" stroke-width="1.4"/>' % (x, y, w, h, fill, edge)]
    out.append('<text x="%d" y="%d" font-family="%s" font-size="12.5" '
               'font-weight="600" fill="%s" text-anchor="middle">%s</text>'
               % (x + w // 2, y + 22, font, edge, esc(title)))
    for i, line in enumerate(lines):
        out.append('<text x="%d" y="%d" font-size="10" fill="%s" '
                   'text-anchor="middle">%s</text>'
                   % (x + w // 2, y + 40 + i * 13, SOFT, esc(line)))
    return '\n'.join(out)


def arrow(x1, y1, x2, y2, label='', dashed=False):
    d = ' stroke-dasharray="5 4"' if dashed else ''
    parts = ['<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.7"%s marker-end="url(#a)"/>'
             % (x1, y1, x2, y2, FAINT, d)]
    if label:
        parts.append('<text x="%.0f" y="%.0f" font-size="9.5" fill="%s" '
                     'text-anchor="middle">%s</text>'
                     % ((x1 + x2) / 2, (y1 + y2) / 2 - 7, FAINT, esc(label)))
    return '\n'.join(parts)


def main():
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'width="%d" height="%d" font-family="Inter, Helvetica, Arial, '
         'sans-serif">' % (W, H, W, H)]
    s.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker></defs>' % FAINT)
    s.append('<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, H))

    s.append('<text x="30" y="34" font-size="15" font-weight="600" fill="%s">'
             'The simulation chain</text>' % INK)
    s.append('<text x="30" y="54" font-size="11" fill="%s">'
             'What GRANDlib owns is the shaded band: the instrument response, '
             'and the schema on both sides of it.</text>' % SOFT)

    # The band marking what GRANDlib is responsible for.
    s.append('<rect x="24" y="86" width="952" height="250" rx="8" fill="%s" '
             'stroke="%s" stroke-width="1" stroke-dasharray="6 4"/>'
             % (BAND, IN_EDGE))
    s.append('<text x="38" y="106" font-size="10.5" font-weight="600" '
             'fill="%s">GRANDlib</text>' % IN_EDGE)

    # Upstream, outside.
    s.append(box(24, 372, 200, 78, OUT_FILL, OUT_EDGE, 'ZHAireS / CoREAS',
                 ['air shower and its', 'radio emission', 'not GRANDlib']))
    s.append(box(258, 372, 170, 78, OUT_FILL, OUT_EDGE, 'sim2root',
                 ['converts to the', 'GRAND schema', 'not linted or tested']))

    # The four stages.
    y = 150
    s.append(box(44, y, 190, 96, IN_FILL, IN_EDGE, 'antenna',
                 ['effective length', 'V_oc = l . E', 'notebook 03']))
    s.append(box(276, y, 190, 96, IN_FILL, IN_EDGE, 'RF chain',
                 ['6 stages, ABCD', 'V_out / V_oc', 'notebook 04']))
    s.append(box(508, y, 190, 96, IN_FILL, IN_EDGE, 'Galactic noise',
                 ['LFMap tables', 'added, not convolved', 'notebook 05']))
    s.append(box(740, y, 190, 96, IN_FILL, IN_EDGE, 'ADC',
                 ['14 bit, 1.8 V', 'LSB ~ 110 uV', 'notebook 06']))

    for x1, x2 in ((234, 276), (466, 508), (698, 740)):
        s.append(arrow(x1, y + 48, x2, y + 48))

    # The trees at each boundary.
    s.append(box(44, 268, 190, 52, IO_FILL, IO_EDGE, 'TEfield',
                 ['traces in uV/m']))
    s.append(box(740, 268, 190, 52, IO_FILL, IO_EDGE, 'TVoltage / TADC',
                 ['traces in uV / counts']))
    s.append(box(392, 268, 190, 52, IO_FILL, IO_EDGE, 'TRun + TShower',
                 ['geometry and metadata']))

    s.append(arrow(139, 268, 139, 246))
    s.append(arrow(487, 268, 487, 246, dashed=True))
    s.append(arrow(835, 246, 835, 268))

    # Downstream, outside.
    s.append(box(776, 372, 200, 78, OUT_FILL, OUT_EDGE, 'reconstruction',
                 ['direction, energy,', 'Xmax', 'not GRANDlib']))

    # ZHAireS -> sim2root -> TEfield, and TVoltage -> reconstruction.  Every
    # arrow stops on a box edge; one that ends inside a box lands on its title.
    s.append(arrow(224, 411, 256, 411))
    s.append(arrow(343, 370, 343, 324))
    s.append('<text x="353" y="350" font-size="9.5" fill="%s">writes</text>' % FAINT)
    s.append(arrow(835, 324, 835, 370))
    s.append('<text x="845" y="350" font-size="9.5" fill="%s">reads</text>' % FAINT)

    s.append('<line x1="24" y1="474" x2="976" y2="474" stroke="%s"/>' % RULE)
    s.append('<text x="24" y="496" font-size="10.5" fill="%s">'
             'Everything runs one detection unit at a time and one event at a '
             'time; the chain has no notion of an array trigger.</text>' % FAINT)
    s.append('<text x="24" y="512" font-size="10.5" fill="%s">'
             'Absolute noise level carries the open sqrt(2) question and depends '
             'on du_type; vga_gain is currently ignored.</text>' % FAINT)

    s.append('</svg>')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'source', '_static', 'pipeline.svg')
    with open(out, 'w') as handle:
        handle.write('\n'.join(s) + '\n')
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
