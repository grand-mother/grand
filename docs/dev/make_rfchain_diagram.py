#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Draws the RF-chain cascade for the documentation.

    python docs/dev/make_rfchain_diagram.py -> docs/source/_static/rfchain.svg

Three things the prose has trouble carrying at once: which measured file each
stage reads, that the matrices are multiplied in an order that is not the
physical one, and that the stage named for the variable-gain amplifier does not
read a VGA file.
"""

import os

W, H = 1130, 560
INK, SOFT, FAINT = '#14202A', '#4C5C69', '#7A8994'
RULE = '#D8DEE3'
STAGE_FILL, STAGE_EDGE = '#E1F1EA', '#1D7A57'
TERM_FILL, TERM_EDGE = '#F6EDDA', '#8A6210'
WARN_FILL, WARN_EDGE = '#FBEDE9', '#B4472A'

#: (attribute, class, file it reads, order in the matrix product).
STAGES = [
    ('matcnet', 'MatchingNetwork',   'MatchingNetworkX.s2p',    2),
    ('lna',     'LowNoiseAmplifier', 'LNA-X.s2p',               3),
    ('balun1',  'BalunAfterLNA',     'balun_in_nut.s2p',        1),
    ('cable',   'Cable',             'cable+Connector.s2p',     4),
    ('vgaf',    'VGAFilter',         'feb+amfitler+biast.s2p',  5),
    ('balun2',  'BalunBeforeADC',    'balun_before_ad.s2p',     6),
]


def esc(text):
    r"""Returns `text` with the XML metacharacters escaped."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def label(x, y, text, colour, size=10, anchor='start', weight='400', mono=False):
    r"""Returns one text element."""
    font = ' font-family="IBM Plex Mono, monospace"' if mono else ''
    return ('<text x="%.1f" y="%.1f" font-size="%s" fill="%s" text-anchor="%s" '
            'font-weight="%s"%s>%s</text>'
            % (x, y, size, colour, anchor, weight, font, esc(text)))


def main():
    r"""Writes the diagram to ``docs/source/_static/rfchain.svg``."""
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'width="%d" height="%d" font-family="Inter, Helvetica, Arial, '
         'sans-serif">' % (W, H, W, H)]
    s.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker></defs>' % FAINT)
    s.append('<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, H))

    s.append(label(28, 34, 'The RF chain', INK, 15, weight='600'))
    s.append(label(28, 54,
                   'Six two-port stages between the antenna terminals and the '
                   'ADC, terminated in Z_ant at the input and Z_load at the '
                   'output.', SOFT, 11))

    # --- the cascade, left to right in signal-flow order --------------------
    y, bw, bh, gap = 108, 140, 92, 12
    x0 = 46

    s.append('<rect x="%d" y="%d" width="70" height="%d" rx="5" fill="%s" '
             'stroke="%s" stroke-width="1.4"/>' % (x0 - 40, y + 16, 60, TERM_FILL, TERM_EDGE))
    s.append(label(x0 - 10, y + 44, 'Z_ant', TERM_EDGE, 11, anchor='middle',
                   weight='600', mono=True))
    s.append(label(x0 - 10, y + 60, 'antenna', SOFT, 9, anchor='middle'))

    for i, (attr, cls, path, order) in enumerate(STAGES):
        x = x0 + 52 + i * (bw + gap)
        warn = attr == 'vgaf'
        fill, edge = (WARN_FILL, WARN_EDGE) if warn else (STAGE_FILL, STAGE_EDGE)
        s.append('<rect x="%d" y="%d" width="%d" height="%d" rx="5" fill="%s" '
                 'stroke="%s" stroke-width="1.4"/>' % (x, y, bw, bh, fill, edge))
        s.append(label(x + bw / 2, y + 21, attr, edge, 12.5, anchor='middle',
                       weight='600', mono=True))
        s.append(label(x + bw / 2, y + 37, cls, SOFT, 8.5, anchor='middle'))
        # The file, wrapped by hand: these names are long and must stay legible.
        head, _, tail = path.partition('.')
        s.append(label(x + bw / 2, y + 56, head, INK if not warn else WARN_EDGE,
                       8, anchor='middle', mono=True))
        s.append(label(x + bw / 2, y + 67, '.' + tail, INK if not warn else WARN_EDGE,
                       8, anchor='middle', mono=True))
        # The position of this stage in the matrix product.
        s.append('<circle cx="%d" cy="%d" r="9" fill="%s"/>' % (x + bw - 16, y + 80, edge))
        s.append(label(x + bw - 16, y + 83.5, str(order), '#FFFFFF', 9.5,
                       anchor='middle', weight='600'))
        if i:
            s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
                     'stroke-width="1.6" marker-end="url(#a)"/>'
                     % (x - gap, y + bh / 2, x - 2, y + bh / 2, FAINT))

    xe = x0 + 52 + len(STAGES) * (bw + gap)
    s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.6" marker-end="url(#a)"/>'
             % (xe - gap, y + bh / 2, xe + 10, y + bh / 2, FAINT))
    s.append('<rect x="%d" y="%d" width="60" height="%d" rx="5" fill="%s" '
             'stroke="%s" stroke-width="1.4"/>' % (xe + 14, y + 16, 60, TERM_FILL, TERM_EDGE))
    s.append(label(xe + 44, y + 44, 'Z_load', TERM_EDGE, 11, anchor='middle',
                   weight='600', mono=True))
    s.append(label(xe + 44, y + 60, 'ADC', SOFT, 9, anchor='middle'))

    s.append(label(x0 + 52, y + bh + 26,
                   'Numbered circles give the order of the matrix product, '
                   'which is not the signal-flow order drawn above:', SOFT, 10.5))
    s.append(label(x0 + 52, y + bh + 44,
                   'balun1 . matcnet . lna . cable . vgaf . balun2', INK, 11.5,
                   weight='600', mono=True))
    s.append(label(x0 + 52, y + bh + 62,
                   'The first factor is the class named BalunAfterLNA, applied '
                   'before the LNA. Either the name or the order is misleading.',
                   FAINT, 10))

    # --- S to ABCD ----------------------------------------------------------
    s.append('<line x1="28" y1="330" x2="1102" y2="330" stroke="%s"/>' % RULE)
    s.append(label(28, 356, 'Why ABCD and not S', INK, 12.5, weight='600'))
    s.append(label(28, 378,
                   'S-parameters are what a network analyser measures, but they '
                   'do not cascade by multiplication. Each stage is converted', SOFT, 10.5))
    s.append(label(28, 394,
                   'with s2abcd() and the ABCD matrices are multiplied by '
                   'matmul(). Z_ant enters at the input and Z_load at the '
                   'output, which is why', SOFT, 10.5))
    s.append(label(28, 410,
                   'a stage’s isolated |S21| does not add up to the total.',
                   SOFT, 10.5))
    s.append(label(28, 436, 'H(nu) = V_out / V_oc', INK, 12, weight='600', mono=True))
    s.append(label(190, 436, '- one complex number per frequency and arm; '
                   'peak |H| is about 95 on the horizontal arms, 39 dB.',
                   SOFT, 10.5))

    # --- the warning --------------------------------------------------------
    s.append('<rect x="28" y="460" width="1074" height="76" rx="5" fill="%s" '
             'stroke="%s" stroke-width="1.4"/>' % (WARN_FILL, WARN_EDGE))
    s.append(label(44, 482, 'vgaf does not read a VGA file', WARN_EDGE, 11.5,
                   weight='600'))
    s.append(label(44, 501,
                   'feb+amfitler+biast.s2p is a front-end board with an AM '
                   'filter and a bias tee. The three per-gain tables '
                   '(filter+vga0db, vga5db, vga20db)', INK, 10))
    s.append(label(44, 517,
                   'ship in data/detector/RFchain_v2/ and are opened by nothing, '
                   'so vga_gain selects nothing. See the known issues page.',
                   INK, 10))

    s.append('</svg>')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'source', '_static', 'rfchain.svg')
    with open(out, 'w') as handle:
        handle.write('\n'.join(s) + '\n')
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
