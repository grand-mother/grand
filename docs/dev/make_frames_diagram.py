#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Draws the coordinate-frame relationships for the documentation.

python docs/dev/make_frames_diagram.py   ->  docs/source/_static/frames.svg
"""

import os

W, H = 940, 470
INK, SOFT, FAINT = '#14202A', '#4C5C69', '#7A8994'
ACC, RULE = '#0E6E70', '#D8DEE3'
BOX = {'geo': '#E2F0F0', 'ecef': '#F6EDDA', 'local': '#E1F1EA'}
EDGE = {'geo': '#0E6E70', 'ecef': '#8A6210', 'local': '#1D7A57'}


def box(x, y, w, h, kind, title, lines):
    r"""Returns one labelled box with a title and body lines."""
    out = ['<rect x="%d" y="%d" width="%d" height="%d" rx="5" fill="%s" '
           'stroke="%s" stroke-width="1.4"/>' % (x, y, w, h, BOX[kind], EDGE[kind])]
    out.append('<text x="%d" y="%d" font-family="IBM Plex Mono, monospace" '
               'font-size="13" font-weight="600" fill="%s" text-anchor="middle">'
               '%s</text>' % (x + w // 2, y + 24, EDGE[kind], title))
    for i, line in enumerate(lines):
        out.append('<text x="%d" y="%d" font-size="10.5" fill="%s" '
                   'text-anchor="middle">%s</text>'
                   % (x + w // 2, y + 44 + i * 14, SOFT, line))
    return '\n'.join(out)


def arrow(x1, y1, x2, y2, label, dashed=False, above=True):
    r"""Returns a double-headed arrow with a label."""
    d = ' stroke-dasharray="4 3"' if dashed else ''
    mid = ((x1 + x2) / 2, (y1 + y2) / 2)
    return ('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
            'stroke-width="1.6"%s marker-end="url(#a)" marker-start="url(#a)"/>'
            '<text x="%.0f" y="%.0f" font-size="10" fill="%s" '
            'text-anchor="middle">%s</text>'
            % (x1, y1, x2, y2, FAINT, d, mid[0], mid[1] + (-8 if above else 16),
               FAINT, label))


def main():
    r"""Writes the diagram to ``docs/source/_static/frames.svg``."""
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'width="100%%" font-family="IBM Plex Sans, sans-serif">' % (W, H),
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
         '<path d="M 0 0 L 10 5 L 0 10 z" fill="%s"/></marker></defs>' % FAINT,
         '<rect width="%d" height="%d" rx="6" fill="#FBFCFC" stroke="%s"/>'
         % (W, H, RULE)]

    s.append('<text x="40" y="38" font-size="15" font-weight="600" fill="%s">'
             'Coordinate frames, and what converts to what</text>' % INK)
    s.append('<text x="40" y="58" font-size="11" fill="%s">Every conversion '
             'passes through ECEF. There is no direct path between a local '
             'frame and geodetic.</text>' % SOFT)

    s.append(box(40, 90, 210, 96, 'geo', 'Geodetic',
                 ['latitude, longitude, height', 'degrees and metres',
                  'height needs a Reference']))
    s.append(box(365, 90, 210, 96, 'ecef', 'ECEF',
                 ['Earth-centred, Earth-fixed', 'metres from the geocentre',
                  'the common pivot']))
    s.append(box(690, 60, 210, 78, 'local', 'LTP',
                 ['local tangent plane', 'x East, y North, z Up']))
    s.append(box(690, 158, 210, 78, 'local', 'GRANDCS',
                 ['the array frame', 'x North, y West, z Up']))

    s.append(arrow(250, 138, 365, 138, 'ECEF(g) / Geodetic(e)'))
    s.append(arrow(575, 120, 690, 99, 'rotation'))
    s.append(arrow(575, 156, 690, 197, 'rotation', above=False))

    s.append('<text x="40" y="238" font-size="12" font-weight="600" '
             'fill="%s">The trap</text>' % '#9C3830')
    s.append('<rect x="40" y="250" width="860" height="86" rx="5" '
             'fill="#F7E6E4" stroke="#9C3830" stroke-width="1.2"/>')
    for i, line in enumerate([
            'GRANDCS and LTP take the same three numbers and mean different '
            'things by them: GRANDCS x runs North, ENU x runs East.',
            'At Dunhuang, GRANDCS(x=1000, y=0, z=0) and LTP(x=1000, y=0, z=0) '
            'name points 1410 m apart — with no error raised,',
            'no warning, and both results perfectly valid. Name the frame in '
            'the variable, not just in the constructor.']):
        s.append('<text x="60" y="%d" font-size="11" fill="%s">%s</text>'
                 % (274 + i * 18, '#4C5C69', line))

    s.append('<text x="40" y="374" font-size="12" font-weight="600" '
             'fill="%s">Heights</text>' % ACC)
    s.append('<line x1="40" y1="404" x2="440" y2="404" stroke="%s" '
             'stroke-width="2"/>' % EDGE['geo'])
    s.append('<text x="450" y="408" font-size="10.5" fill="%s">'
             'WGS-84 ellipsoid — Reference.ELLIPSOID</text>' % SOFT)
    s.append('<path d="M 40 428 q 100 -12 200 0 q 100 12 200 0" fill="none" '
             'stroke="%s" stroke-width="2" stroke-dasharray="5 3"/>' % EDGE['ecef'])
    s.append('<text x="450" y="432" font-size="10.5" fill="%s">'
             'geoid, mean sea level — Reference.GEOID</text>' % SOFT)
    s.append('<text x="40" y="454" font-size="10.5" fill="%s">'
             'They differ by up to ~100 m worldwide, and by −7.75 m at '
             'Dunhuang. A height without a reference is not a height.</text>' % FAINT)

    s.append('</svg>')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'source', '_static', 'frames.svg')
    open(out, 'w').write('\n'.join(s) + '\n')
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
