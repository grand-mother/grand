#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Draws why an antenna arm is not a Cartesian field component.

    python docs/dev/make_antenna_diagram.py -> docs/source/_static/antenna_arms.svg

The most costly misreading of a GRAND trace is that ``trace[:, 2]`` holds
:math:`E_z`.  It does not: the response is the projection of the field onto the
effective length in the spherical basis of the *arrival direction*, so which
arm sees a given field depends on where the shower came from.  Numbers on the
right are measured at 100 MHz and zenith 85 degrees, the geometry of notebook
06, where the input field components are 1.0 : 0.6 : 0.2 and the outputs come
out closer to 600 : 400 : 1.
"""

import math
import os

W, H = 1000, 480
INK, SOFT, FAINT = '#14202A', '#4C5C69', '#7A8994'
RULE = '#D8DEE3'
ARM = '#1D7A57'
FIELD = '#B4472A'
BASIS = '#0E6E70'
WARN_FILL, WARN_EDGE = '#FBEDE9', '#B4472A'


def esc(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def line(x1, y1, x2, y2, colour, width=1.8, dashed=False, head=True):
    d = ' stroke-dasharray="5 4"' if dashed else ''
    m = ' marker-end="url(#%s)"' % ('h' + colour.lstrip('#')) if head else ''
    return ('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
            'stroke-width="%s"%s%s/>' % (x1, y1, x2, y2, colour, width, d, m))


def label(x, y, text, colour, size=10.5, anchor='start', weight='400', mono=False):
    font = ' font-family="IBM Plex Mono, monospace"' if mono else ''
    return ('<text x="%.1f" y="%.1f" font-size="%s" fill="%s" '
            'text-anchor="%s" font-weight="%s"%s>%s</text>'
            % (x, y, size, colour, anchor, weight, font, esc(text)))


def main():
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'width="%d" height="%d" font-family="Inter, Helvetica, Arial, '
         'sans-serif">' % (W, H, W, H)]
    defs = ['<defs>']
    for colour in (ARM, FIELD, BASIS, FAINT):
        defs.append('<marker id="h%s" viewBox="0 0 10 10" refX="9" refY="5" '
                    'markerWidth="6" markerHeight="6" '
                    'orient="auto-start-reverse">'
                    '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>'
                    % (colour.lstrip('#'), colour))
    defs.append('</defs>')
    s.append('\n'.join(defs))
    s.append('<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, H))

    s.append(label(30, 34, 'An antenna arm is not a field component', INK, 15, weight='600'))
    s.append(label(30, 54,
                   'V_oc is the projection of E onto the effective length, in the '
                   'spherical basis of the arrival direction.', SOFT, 11))

    # --- left panel: the antenna and the arriving wave ----------------------
    ox, oy = 250, 300           # origin of the antenna
    s.append('<line x1="60" y1="%d" x2="440" y2="%d" stroke="%s" '
             'stroke-width="1.2"/>' % (oy, oy, RULE))
    s.append(label(66, oy + 16, 'ground', FAINT, 9.5))

    # The three arms.
    s.append(line(ox, oy, ox, oy - 120, ARM))
    s.append(label(ox + 8, oy - 122, 'Z arm', ARM, 10.5, weight='600'))
    s.append(line(ox, oy, ox + 105, oy - 42, ARM))
    s.append(label(ox + 110, oy - 44, 'EW arm', ARM, 10.5, weight='600'))
    s.append(line(ox, oy, ox - 105, oy - 42, ARM))
    s.append(label(ox - 110, oy - 44, 'SN arm', ARM, 10.5, anchor='end', weight='600'))

    # The incoming wave at 85 degrees zenith: nearly horizontal.
    zen = math.radians(85.0)
    length = 175
    dx, dy = length * math.sin(zen), length * math.cos(zen)
    s.append(line(ox - dx, oy - dy - 60, ox - 14, oy - 66, FAINT, 2.0))
    s.append(label(ox - dx + 4, oy - dy - 70, 'incoming wave, zenith 85°', FAINT, 10))

    # The local spherical basis at that direction: theta nearly vertical,
    # phi horizontal and out of the page (drawn as a short stub).
    s.append(line(ox - 40, oy - 96, ox - 40 + 46 * math.cos(zen),
                  oy - 96 + 46 * math.sin(zen), BASIS, 1.8, dashed=True))
    s.append(label(ox + 14, oy - 78, 'θ̂', BASIS, 12, weight='600'))
    s.append(line(ox - 40, oy - 96, ox - 76, oy - 118, BASIS, 1.8, dashed=True))
    s.append(label(ox - 92, oy - 120, 'φ̂', BASIS, 12, weight='600'))

    # The field, in the theta-phi plane.
    s.append(line(ox - 40, oy - 96, ox - 6, oy - 148, FIELD, 2.2))
    s.append(label(ox - 2, oy - 150, 'E', FIELD, 12, weight='600'))

    s.append(label(250, 352, 'the geometry', SOFT, 11, anchor='middle', weight='600'))
    s.append(label(250, 370,
                   'θ̂ and φ̂ are set by where the shower came from,', FAINT, 9.5,
                   anchor='middle'))
    s.append(label(250, 384, 'not by how the antenna is built.', FAINT, 9.5,
                   anchor='middle'))

    # --- right panel: the numbers -------------------------------------------
    s.append('<line x1="500" y1="86" x2="500" y2="392" stroke="%s"/>' % RULE)

    s.append(label(540, 108, 'Effective length at 100 MHz, zenith 85°', INK, 12,
                   weight='600'))
    rows = [('arm', '|ℓ_θ| [m]', '|ℓ_φ| [m]'),
            ('sn', '0.060', '0.796'),
            ('ew', '0.086', '0.799'),
            ('z', '0.603', '0.011')]
    for i, (a, b, c) in enumerate(rows):
        y = 134 + i * 20
        weight = '600' if i == 0 else '400'
        colour = SOFT if i == 0 else INK
        s.append(label(540, y, a, colour, 10.5, weight=weight, mono=True))
        s.append(label(680, y, b, colour, 10.5, anchor='end', weight=weight, mono=True))
        s.append(label(790, y, c, colour, 10.5, anchor='end', weight=weight, mono=True))
    s.append('<line x1="540" y1="140" x2="790" y2="140" stroke="%s"/>' % RULE)

    s.append(label(540, 236,
                   'The Z arm is the most sensitive of the three here —', SOFT, 10.5))
    s.append(label(540, 252,
                   'and it is the one that sees almost nothing.', SOFT, 10.5))

    s.append('<rect x="540" y="272" width="420" height="86" rx="5" fill="%s" '
             'stroke="%s" stroke-width="1.4"/>' % (WARN_FILL, WARN_EDGE))
    s.append(label(556, 294, 'In notebook 06', WARN_EDGE, 11, weight='600'))
    s.append(label(556, 314, 'input field components   1.0 : 0.6 : 0.2', INK, 10.5,
                   mono=True))
    s.append(label(556, 332, 'output arm amplitudes    600 : 400 : 1', INK, 10.5,
                   mono=True))
    s.append(label(556, 350, 'The ratio is set by the direction, not the field.',
                   WARN_EDGE, 9.5))

    s.append('<line x1="30" y1="418" x2="970" y2="418" stroke="%s"/>' % RULE)
    s.append(label(30, 440,
                   'Consequence: an analysis that treats the three channels as a '
                   'Cartesian decomposition of E is wrong, and wrong in a', FAINT, 10.5))
    s.append(label(30, 456,
                   'direction-dependent way that does not look like a bug. Rotate '
                   'through the effective length instead.', FAINT, 10.5))

    s.append('</svg>')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'source', '_static', 'antenna_arms.svg')
    with open(out, 'w') as handle:
        handle.write('\n'.join(s) + '\n')
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
