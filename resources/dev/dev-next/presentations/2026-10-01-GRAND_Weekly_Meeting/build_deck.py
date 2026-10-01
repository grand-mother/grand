#!/usr/bin/env python
"""Builds the GNN Board talk as an .odp. The machinery; the content is deck.py.

Same machinery as the GRANDlib weekly deck of 3 September 2026, with the
layout conventions of its hand-corrected copy,

    ../2026.09.03 GRAND Weekly Meeting/
        2026-09-03-Bustamante-GRAND_Weekly_Meeting (Copy).odp

which is also the carrier: its styles.xml, master page, background bitmap and
UCPH/Villum logos are kept verbatim and only the slides are replaced. Set
DECK_TEMPLATE to use another carrier.

What the hand-corrected copy changed, and this script now does itself:

  * Titles are real title placeholders (presentation:class="title"), not loose
    text frames. They show in the outline and navigator, and the master's
    title style centres them vertically in the 1.76 cm slot at the top.
  * The slide number sits 0.15 cm lower in its blue box: the copy's number
    frame carries the padding of the Default_3_1 graphic style.
  * The box itself moved 6 um up and left, to (26.593, 14.840) cm.
  * Page layout AL1T1 rather than AL2T1.
  * Numbered items hang: the continuation lines align with the text, not with
    the number. The copy faked it with a tab and six spaces; here it is a real
    hanging indent (pass a (label, text) tuple as the paragraph).
  * Two-line section dividers set the small function words smaller, 40 pt
    against 60 pt ("What {is} needed / {from} you"). Braces mark them.
  * The credit on a full-figure slide is right-aligned.

    python deck.py

Writes 2026-09-28-Bustamante-GNN_Board.odp beside this script.
"""

from __future__ import annotations

import html
import os
import math
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
TEMPLATE = Path(os.environ.get("DECK_TEMPLATE") or (
    Path.home() / "Documents/Expos/2026" / "2026.09.03 GRAND Weekly Meeting"
    / "2026-09-03-Bustamante-GRAND_Weekly_Meeting (Copy).odp"))
OUT = HERE / "2026-09-28-Bustamante-GNN_Board.odp"
TITLE = "GRAND: application to join the GNN"

FIGS = HERE / "figures"

BUILD = HERE / ".build"

# Slide geometry, in cm, from the template's page layout.
W, H = 27.991, 15.748
MARGIN = 1.27
BODY_W = W - 2 * MARGIN

# The template's palette.
RED = "#ed1c24"          # title slide, strongest emphasis
DARKRED = "#bb0000"      # emphasis in body text
BLUE = "#0262ab"         # structural lead-ins, references
DEEPBLUE = "#0066b3"     # section-divider fill, page-number box
GREY = "#666666"         # authors, affiliation, venue
GREEN = "#158466"        # side annotations
MAGENTA = "#bf0041"      # the second annotation colour
BLACK = "#000000"
WHITE = "#ffffff"
HIGHLIGHT = "#fff200"    # ??placeholders?? only -- none should survive to the talk

PT = 0.03528             # 1 pt in cm

# Impress applies a proportional line height to the face's own line (ascent +
# descent), not to the point size. For Palatino that line is 1.151 em: 132% of
# 18 pt lays out at 0.965 cm, which is what the hand-corrected copy records.
# Every height estimate multiplies by this, so frames placed below a measured
# block land where the block actually ends.
LINE = 1.151

# ---------------------------------------------------------------- style pool

_styles: list[str] = []
_seen: dict[tuple, str] = {}


def _text_style(size, color, bold=False, italic=False, font=None, position=None,
                background=None, underline=False):
    """Returns the name of a text style, creating it on first use."""
    key = ("T", size, color, bold, italic, font, position, background, underline)
    if key in _seen:
        return _seen[key]
    name = f"xT{len(_seen)}"
    _seen[key] = name
    # Bold is a separate family, not a weight: only Regular, Italic and Bold
    # Italic of "Palatino Linotype" are installed, so fo:font-weight="bold"
    # silently lands on Bold Italic. "Palatino-Bold" is the upright bold, and is
    # what the template itself uses (the "MB" in its citation lines).
    if font is None:
        font = "Palatino-Bold" if bold else "Palatino Linotype"
    # Times New Roman is asked for bold by weight; Palatino-Bold already is bold.
    weight = ' fo:font-weight="bold"' if (bold and font != "Palatino-Bold") else ""
    style = ' fo:font-style="italic"' if italic else ""
    pos = f' style:text-position="{position}"' if position else ""
    bg = f' fo:background-color="{background}"' if background else ""
    if underline:
        bg += (' style:text-underline-style="solid" '
               'style:text-underline-width="auto" '
               'style:text-underline-color="font-color"')
    _styles.append(
        f'<style:style style:name="{name}" style:family="text">'
        f'<style:text-properties fo:color="{color}" '
        f'style:font-name="{font}" fo:font-size="{size}pt" '
        f'style:font-size-asian="{size}pt" style:font-size-complex="{size}pt"'
        f"{weight}{style}{pos}{bg}/></style:style>"
    )
    return name


# Palatino-Bold resolves to a 400-glyph face that silently drops most symbols —
# ² × ° – α τ → ≈ all render as blanks, while − ± ν ≳ survive. Rather than track
# which, any non-ASCII character in a bold run is set in Times New Roman Bold,
# which was checked to carry the whole set. Regular weight is unaffected: full
# Palatino Linotype has every glyph used here.
SYMBOL_BOLD_FONT = "Times New Roman"


def _bold_safe(ch):
    """Whether Palatino-Bold draws `ch`: ASCII, and the accented Latin letters.

    Accented letters stay in the bold face -- sending the ç of Nançay to Times
    New Roman changes the line height of the whole heading it sits in.
    """
    return ch.isascii() or (ch.isalpha() and ord(ch) < 0x180)


def _bold_spans(text, size, color, italic=False, background=None):
    """Splits a bold run so symbols fall to a face that can actually draw them."""
    out, buf, buf_ascii = [], "", True
    for ch in text:
        is_ascii = _bold_safe(ch)
        if buf and is_ascii != buf_ascii:
            out.append((buf, _text_style(
                size, color, bold=True, italic=italic,
                font=None if buf_ascii else SYMBOL_BOLD_FONT,
                background=background)))
            buf = ""
        buf, buf_ascii = buf + ch, is_ascii
    if buf:
        out.append((buf, _text_style(
            size, color, bold=True, italic=italic,
            font=None if buf_ascii else SYMBOL_BOLD_FONT,
            background=background)))
    return out


def _para_style(align="left", line_height=118, space_after=0.0, hang=0.0):
    key = ("P", align, line_height, space_after, hang)
    if key in _seen:
        return _seen[key]
    name = f"xP{len(_seen)}"
    _seen[key] = name
    # A hanging indent: the first line starts at the frame edge, the rest at
    # `hang`, and the label is separated from the text by a tab that lands on
    # the indent (Impress sets that implicit stop when the indent is negative).
    indent = (f'fo:margin-left="{hang:.3f}cm" fo:text-indent="-{hang:.3f}cm" '
              if hang else "")
    tabs = (f'><style:tab-stops><style:tab-stop style:position="0cm"/>'
            f'</style:tab-stops></style:paragraph-properties>' if hang else "/>")
    _styles.append(
        f'<style:style style:name="{name}" style:family="paragraph">'
        f'<style:paragraph-properties fo:line-height="{line_height}%" '
        f'fo:margin-bottom="{space_after:.3f}cm" {indent}'
        f'fo:text-align="{align}" style:writing-mode="lr-tb"{tabs}</style:style>'
    )
    return name


def _frame_style(valign="top", padding=None, area="justify"):
    # "justify" makes the text area as wide as the frame. With "left", Impress
    # shrinks the area to the text and anchors it left, so a centred or
    # right-aligned paragraph silently lands flush left. Only the slide
    # number keeps "left": that is where the hand-corrected copy's digit sits.
    key = ("gr", valign, padding, area)
    if key in _seen:
        return _seen[key]
    name = f"xgr{len(_seen)}"
    _seen[key] = name
    t, b, l, r = padding or (0, 0, 0, 0)
    _styles.append(
        f'<style:style style:name="{name}" style:family="graphic" '
        f'style:parent-style-name="standard"><style:graphic-properties '
        f'draw:stroke="none" draw:fill="none" draw:textarea-horizontal-align="{area}" '
        f'draw:textarea-vertical-align="{valign}" draw:auto-grow-height="true" '
        f'draw:auto-grow-width="false" fo:padding-top="{t}cm" '
        f'fo:padding-bottom="{b}cm" fo:padding-left="{l}cm" '
        f'fo:padding-right="{r}cm" loext:decorative="false"/></style:style>'
    )
    return name


def _shape_style(fill, stroke="none", stroke_color=WHITE, stroke_w=0.05,
                 dash=False):
    key = ("sh", fill, stroke, stroke_color, stroke_w, dash)
    if key in _seen:
        return _seen[key]
    name = f"xsh{len(_seen)}"
    _seen[key] = name
    if stroke == "none":
        strokexml = 'draw:stroke="none"'
    else:
        kind = "dash" if dash else "solid"
        strokexml = (f'draw:stroke="{kind}" svg:stroke-width="{stroke_w}cm" '
                     f'svg:stroke-color="{stroke_color}"'
                     + (' draw:stroke-dash="Dashed_20__28_var_29__20_4"'
                        if dash else ""))
    fillxml = ('draw:fill="none"' if fill == "none" else
               f'draw:fill="solid" draw:fill-color="{fill}"')
    _styles.append(
        f'<style:style style:name="{name}" style:family="graphic" '
        f'style:parent-style-name="standard"><style:graphic-properties '
        f'{strokexml} {fillxml} '
        f'draw:textarea-vertical-align="middle" '
        f'loext:decorative="false"/></style:style>'
    )
    return name


def _img_style():
    key = ("img",)
    if key in _seen:
        return _seen[key]
    name = f"xim{len(_seen)}"
    _seen[key] = name
    _styles.append(
        f'<style:style style:name="{name}" style:family="graphic" '
        f'style:parent-style-name="standard"><style:graphic-properties '
        f'draw:stroke="none" draw:fill="none" '
        f'loext:decorative="false"/></style:style>'
    )
    return name


def _fixed_styles():
    """Page and placeholder styles, owned here rather than borrowed by name.

    The GRANDlib builder borrowed dp1/dp3/dp4/pr1 from its carrier's
    content.xml. The hand-corrected copy has no dp4 (Impress renumbered its
    notes style to dp2), so these are declared outright, with the copy's own
    properties, and the build no longer depends on what the carrier calls them.
    """
    return (
        '<style:style style:name="xdpTitle" style:family="drawing-page">'
        '<style:drawing-page-properties presentation:background-visible="true" '
        'presentation:background-objects-visible="true" draw:fill="bitmap" '
        'draw:fill-color="#f2edd8" draw:fill-image-name="background" '
        'draw:opacity="12%" style:repeat="stretch" '
        'presentation:display-footer="false" presentation:display-page-number="false" '
        'presentation:display-date-time="false"/></style:style>'
        '<style:style style:name="xdpBody" style:family="drawing-page">'
        '<style:drawing-page-properties presentation:background-visible="true" '
        'presentation:background-objects-visible="true" '
        'presentation:display-footer="false" presentation:display-page-number="false" '
        'presentation:display-date-time="false"/></style:style>'
        '<style:style style:name="xdpNotes" style:family="drawing-page">'
        '<style:drawing-page-properties presentation:display-header="true" '
        'presentation:display-footer="true" presentation:display-page-number="false" '
        'presentation:display-date-time="true"/></style:style>'
        '<style:style style:name="xprTitle" style:family="presentation" '
        'style:parent-style-name="Default-title"><style:graphic-properties '
        'fo:min-height="1.76cm" loext:decorative="false"/>'
        '<style:paragraph-properties style:writing-mode="lr-tb"/></style:style>'
        '<style:style style:name="xprNotes" style:family="presentation" '
        'style:parent-style-name="Default-notes"><style:graphic-properties '
        'draw:fill-color="#ffffff" draw:auto-grow-height="true" '
        'fo:min-height="12.572cm" loext:decorative="false"/>'
        '<style:paragraph-properties style:writing-mode="lr-tb"/></style:style>'
    )


# ---------------------------------------------------------------- rich text

# Doubled markers are bold, single markers are the same colour at regular
# weight -- the same relationship *italic* already had to **bold**.  The
# doubled forms must come first in the alternation or they never match.
#
# Added for this deck, all outside the colour scheme:
#   ??x??   a placeholder: bold on a yellow highlight, to be replaced before
#           the talk. `python deck.py` lists every one that is left.
#   {{x}}   superscript, for exponents: 10{{19}} eV, cm{{−2}}
#   __x__   subscript: ν__τ__
#   ++x++   underlined, in the running colour: the presenter's name
#
# A tilde directly before a digit is a literal "about": ~75%, ~0.1°.
MARKUP = re.compile(
    r"\?\?(.+?)\?\?|\{\{(.+?)\}\}|__(.+?)__|\+\+(.+?)\+\+"
    r"|\*\*(.+?)\*\*|!!(.+?)!!|~~(.+?)~~|\^\^(.+?)\^\^|%%(.+?)%%|@@(.+?)@@"
    r"|~(?!\d)(.+?)~|\^(.+?)\^|\$(.+?)\$|\*(.+?)\*")

PLACEHOLDERS: list[tuple[int, str]] = []


def _runs(text, size, color):
    """Splits a string on the inline markup into styled spans.

    Bold:     **bold**  !!dark red!!  ~~blue~~  ^^deep blue^^  %%green%%
    Regular:  @@grey@@  ~blue~  ^deep blue^  $dark red$  *italic*
    Other:    ??placeholder??  {{superscript}}  __subscript__  ++underline++
    """
    out, pos = [], 0
    for m in MARKUP.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], _text_style(size, color)))
        (hole, sup, sub, under, emph, red, blue, deep, green, grey,
         blue_p, deep_p, red_p, ital) = m.groups()
        if hole is not None:
            PLACEHOLDERS.append((len(_pages) + 1, hole))
            out.extend(_bold_spans(hole, size, BLACK, background=HIGHLIGHT))
        elif sup is not None:
            out.append((sup, _text_style(size, color, position="super 58%")))
        elif sub is not None:
            out.append((sub, _text_style(size, color, position="sub 58%")))
        elif under is not None:
            out.append((under, _text_style(size, color, underline=True)))
        elif emph is not None:
            out.extend(_bold_spans(emph, size, color))
        elif red is not None:
            out.extend(_bold_spans(red, size, DARKRED))
        elif blue is not None:
            out.extend(_bold_spans(blue, size, BLUE))
        elif deep is not None:
            out.extend(_bold_spans(deep, size, DEEPBLUE))
        elif green is not None:
            out.extend(_bold_spans(green, size, GREEN))
        elif grey is not None:
            out.append((grey, _text_style(size, GREY)))
        elif blue_p is not None:
            out.append((blue_p, _text_style(size, BLUE)))
        elif deep_p is not None:
            out.append((deep_p, _text_style(size, DEEPBLUE)))
        elif red_p is not None:
            out.append((red_p, _text_style(size, DARKRED)))
        elif ital is not None:
            out.append((ital, _text_style(size, color, italic=True)))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], _text_style(size, color)))
    return out


def _plain(text):
    return MARKUP.sub(lambda m: next(g for g in m.groups() if g is not None), text)


_MEASURE_PX = 200
_font_cache: dict[str, object] = {}

# The face the widths are measured with. Palatino Linotype first; TeX Gyre
# Pagella is metric-compatible to well under a percent, for machines without it.
_MEASURE_FONTS = [
    Path.home() / ".local/share/fonts/pala.ttf",
    Path.home() / ".fonts/pala.ttf",
    Path("/usr/share/fonts/truetype/msttcorefonts/pala.ttf"),
    Path.home() / ".fonts/PalLin-R.otf",
    Path("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-regular.otf"),
    Path("/usr/share/fonts/opentype/tex-gyre/texgyrepagella-regular.otf"),
]


def _measure(text, size):
    """Width of a string in cm, measured with the actual face, not guessed."""
    from PIL import ImageFont
    path = next((str(p) for p in _MEASURE_FONTS if p.exists()), None)
    if path is None:
        return len(text) * 0.47 * size * PT     # last resort: an average glyph
    f = _font_cache.get(path)
    if f is None:
        f = _font_cache[path] = ImageFont.truetype(path, _MEASURE_PX)
    return f.getlength(text) / _MEASURE_PX * size * PT


def _lines(text, size, width_cm):
    """How many lines a paragraph wraps to, by measuring rather than counting."""
    if not text.strip():
        return 1
    words, n, cur = _plain(text).split(), 1, ""
    for wd in words:
        trial = f"{cur} {wd}" if cur else wd
        if cur and _measure(trial, size) > width_cm:
            n, cur = n + 1, wd
        else:
            cur = trial
    return n


def _spans(text, size, color):
    # Curl the apostrophe before escaping: html.escape would otherwise turn
    # it into &#x27; and the replacement would find nothing to curl.
    return "".join(
        f'<text:span text:style-name="{st}">'
        f"{html.escape(t.replace(chr(39), chr(8217)))}</text:span>"
        for t, st in _runs(text, size, color)
    )


# ---------------------------------------------------------------- elements

def textbox(x, y, w, paragraphs, size=20, color=BLACK, align="left",
            line_height=118, valign="top", h=None, space_after=0.0, hang=0.9,
            padding=None, angle=None, area="justify"):
    """One text frame. `paragraphs` is a list of strings; "" is a blank line.

    A paragraph given as a (label, text) tuple hangs: the label sits at the
    frame edge and every line of the text starts `hang` cm in.

    `angle` (degrees, anticlockwise) turns the frame about its top-left
    corner, which stays at (x, y), as Impress writes a rotated frame.
    """
    ps = _para_style(align, line_height, space_after)
    body, total = [], 0
    for p in paragraphs:
        if isinstance(p, tuple):
            label, txt = p
            hs = _para_style(align, line_height, space_after, hang)
            body.append(f'<text:p text:style-name="{hs}">'
                        f'{_spans(label, size, color)}<text:tab/>'
                        f'{_spans(txt, size, color)}</text:p>')
            total += _lines(txt, size, w - hang)
            continue
        total += _lines(p, size, w)
        if not p:
            body.append(f'<text:p text:style-name="{ps}"/>')
            continue
        body.append(f'<text:p text:style-name="{ps}">'
                    f'{_spans(p, size, color)}</text:p>')
    height = h if h is not None else max(
        0.6, total * size * PT * LINE * line_height / 100
        + space_after * len(paragraphs) + 0.2)
    if angle:
        place = (f'draw:transform="rotate ({math.radians(angle):.6f}) '
                 f'translate ({x:.3f}cm {y:.3f}cm)"')
    else:
        place = f'svg:x="{x:.3f}cm" svg:y="{y:.3f}cm"'
    return (
        f'<draw:frame draw:style-name="{_frame_style(valign, padding, area)}" '
        f'draw:layer="layout" '
        f'svg:width="{w:.3f}cm" svg:height="{height:.3f}cm" '
        f'{place}><draw:text-box>'
        f'{"".join(body)}</draw:text-box></draw:frame>'
    ), height


def picture(href, x, y, w, h, xml_id=None):
    """An image frame. `xml_id` names it for an on-click animation (slide())."""
    mime = "image/png" if href.lower().endswith(".png") else "image/jpeg"
    ident = f'xml:id="{xml_id}" draw:id="{xml_id}" ' if xml_id else ""
    return (
        f'<draw:frame draw:style-name="{_img_style()}" {ident}draw:layer="layout" '
        f'svg:width="{w:.3f}cm" svg:height="{h:.3f}cm" '
        f'svg:x="{x:.3f}cm" svg:y="{y:.3f}cm">'
        f'<draw:image xlink:href="Pictures/{href}" xlink:type="simple" '
        f'xlink:show="embed" xlink:actuate="onLoad" draw:mime-type="{mime}">'
        f"<text:p/></draw:image></draw:frame>"
    )


ROUNDRECT_GEOM = (
    '<draw:enhanced-geometry svg:viewBox="0 0 21600 21600" '
    'draw:path-stretchpoint-x="10800" draw:path-stretchpoint-y="10800" '
    'draw:text-areas="?f3 ?f4 ?f5 ?f6" draw:type="round-rectangle" '
    'draw:modifiers="3600" draw:enhanced-path="M ?f7 0 X 0 ?f8 L 0 ?f9 Y ?f7 21600 '
    'L ?f10 21600 X 21600 ?f9 L 21600 ?f8 Y ?f10 0 Z N">'
    '<draw:equation draw:name="f0" draw:formula="45"/>'
    '<draw:equation draw:name="f1" draw:formula="$0 *sin(?f0 *(pi/180))"/>'
    '<draw:equation draw:name="f2" draw:formula="?f1 *3163/7636"/>'
    '<draw:equation draw:name="f3" draw:formula="left+?f2 "/>'
    '<draw:equation draw:name="f4" draw:formula="top+?f2 "/>'
    '<draw:equation draw:name="f5" draw:formula="right-?f2 "/>'
    '<draw:equation draw:name="f6" draw:formula="bottom-?f2 "/>'
    '<draw:equation draw:name="f7" draw:formula="left+$0 "/>'
    '<draw:equation draw:name="f8" draw:formula="top+$0 "/>'
    '<draw:equation draw:name="f9" draw:formula="bottom-$0 "/>'
    '<draw:equation draw:name="f10" draw:formula="right-$0 "/>'
    '<draw:handle draw:handle-position="$0 top" draw:handle-switched="true" '
    'draw:handle-range-x-minimum="0" draw:handle-range-x-maximum="10800"/>'
    "</draw:enhanced-geometry>"
)


def roundrect(x, y, w, h, fill, stroke="none", stroke_color=WHITE, stroke_w=0.05,
              dash=False):
    return (
        f'<draw:custom-shape draw:style-name='
        f'"{_shape_style(fill, stroke, stroke_color, stroke_w, dash)}" '
        f'draw:layer="layout" '
        f'svg:width="{w:.3f}cm" svg:height="{h:.3f}cm" '
        f'svg:x="{x:.3f}cm" svg:y="{y:.3f}cm"><text:p/>{ROUNDRECT_GEOM}'
        f"</draw:custom-shape>"
    )


def _line_style(color, width, arrow):
    key = ("ln", color, width, arrow)
    if key in _seen:
        return _seen[key]
    name = f"xln{len(_seen)}"
    _seen[key] = name
    head = (' draw:marker-end="Arrow" draw:marker-end-width="0.40cm"'
            if arrow else "")
    _styles.append(
        f'<style:style style:name="{name}" style:family="graphic" '
        f'style:parent-style-name="standard"><style:graphic-properties '
        f'draw:stroke="solid" svg:stroke-width="{width}cm" '
        f'svg:stroke-color="{color}"{head} draw:fill="none" '
        f'loext:decorative="false"/></style:style>'
    )
    return name


def line(x1, y1, x2, y2, color=BLACK, width=0.053, arrow=True, xml_id=None):
    """A straight line, with an arrowhead at (x2, y2) unless `arrow` is off."""
    ident = f'xml:id="{xml_id}" draw:id="{xml_id}" ' if xml_id else ""
    return (
        f'<draw:line draw:style-name="{_line_style(color, width, arrow)}" '
        f'{ident}draw:layer="layout" svg:x1="{x1:.3f}cm" svg:y1="{y1:.3f}cm" '
        f'svg:x2="{x2:.3f}cm" svg:y2="{y2:.3f}cm"><text:p/></draw:line>'
    )


def appear_on_click(ids):
    """Impress timing: the named shapes appear together on the next click."""
    if not ids:
        return ""
    sets = []
    for i, ident in enumerate(ids):
        kind = "on-click" if i == 0 else "with-previous"
        sets.append(
            f'<anim:par smil:begin="0s" smil:fill="hold" '
            f'presentation:node-type="{kind}" presentation:preset-class="entrance" '
            f'presentation:preset-id="ooo-entrance-appear">'
            f'<anim:set smil:begin="0s" smil:dur="0.001s" smil:fill="hold" '
            f'smil:targetElement="{ident}" smil:attributeName="visibility" '
            f'smil:to="visible"/></anim:par>')
    return (
        '<anim:par presentation:node-type="timing-root">'
        '<anim:seq presentation:node-type="main-sequence">'
        '<anim:par smil:begin="next"><anim:par smil:begin="0s">'
        + "".join(sets) +
        "</anim:par></anim:par></anim:seq></anim:par>"
    )


def rule(x, y, w, thickness=0.025, color="#999999"):
    """A hairline, drawn as a thin filled rectangle."""
    return (
        f'<draw:rect draw:style-name="{_shape_style(color)}" draw:layer="layout" '
        f'svg:width="{w:.3f}cm" svg:height="{thickness:.3f}cm" '
        f'svg:x="{x:.3f}cm" svg:y="{y:.3f}cm"><text:p/></draw:rect>'
    )


def placeholder_box(x, y, w, h, lines, size=14):
    """A dashed frame standing in for a plot that has to be pasted in."""
    frame = roundrect(x, y, w, h, "#f4f4f4", stroke="dash",
                      stroke_color="#999999", stroke_w=0.03, dash=True)
    t, th = textbox(x + 0.5, y, w - 1.0, lines, size=size, color=GREY,
                    align="center", valign="middle", h=h)
    return frame + t


def table(x, y, col_x, rows, size=16, header=True, row_h=0.78, rules=True,
          aligns=None, colors=None, width=None, band=None, band_color="#eef3f8",
          pad=0.42):
    """A grid of positioned text frames — Impress tables are not worth the pain.

    `col_x` gives each column's left edge relative to `x`; `rows[0]` is the header
    when `header` is true. `band` shades the given row indices (after the header).
    `row_h` may be a list, one height per row, or "auto": each row then gets
    the height of its longest cell, measured, plus `pad`.
    """
    parts, cur = [], y
    aligns = aligns or ["left"] * len(col_x)
    total_w = width if width is not None else (col_x[-1] + 5.0)

    def cell_w(c):
        return ((col_x[c + 1] - col_x[c] - 0.25) if c + 1 < len(col_x)
                else (total_w - col_x[c]))

    if row_h == "auto":
        heights = [max(_lines(cell or "", size, cell_w(c))
                       for c, cell in enumerate(row)) * size * PT * LINE * 1.18
                   + pad
                   for row in rows]
    else:
        heights = row_h if isinstance(row_h, list) else [row_h] * len(rows)
    for i, row in enumerate(rows):
        rh = heights[i]
        is_head = header and i == 0
        if band and (i - (1 if header else 0)) in band and not is_head:
            parts.append(rule(x - 0.2, cur - 0.12, total_w + 0.4, rh,
                              band_color))
        for c, cell in enumerate(row):
            if cell is None:
                continue
            col = colors[c] if colors else BLACK
            if is_head:
                col = BLUE
            w = cell_w(c)
            t, _ = textbox(x + col_x[c], cur, w, [cell], size=size, color=col,
                           align=aligns[c], h=rh)
            parts.append(t)
        cur += rh
        if is_head and rules:
            parts.append(rule(x, cur - 0.12, total_w, 0.03, "#7f7f7f"))
            cur += 0.14
    if rules:
        parts.append(rule(x, cur - 0.06, total_w, 0.025, "#bbbbbb"))
    return parts, cur


# The title sits in a fixed slot at the very top of the page, full bleed,
# rather than being inset with the body.  Every content slide uses the same
# rectangle, so the titles do not shift as their length changes.  It is a real
# title placeholder, so the master's title style centres it in the slot.
LEAD_X, LEAD_Y, LEAD_W, LEAD_H = 0.598, 0.007, 26.824, 1.760
LEAD_SIZE = 28

#: Where the body starts on a slide with a lead.
BODY_TOP = 2.599

#: The lowest a body frame may reach without meeting the slide number.
BODY_BOTTOM = 14.70


def lead_frame(lead, color=BLUE, size=LEAD_SIZE):
    ps = _para_style("left", 118, 0.0)
    lines = lead if isinstance(lead, list) else [lead]
    paras = "".join(f'<text:p text:style-name="{ps}">{_spans(t, size, color)}'
                    f"</text:p>" for t in lines)
    return (
        f'<draw:frame presentation:style-name="xprTitle" draw:layer="layout" '
        f'svg:width="{LEAD_W:.3f}cm" svg:height="{LEAD_H:.3f}cm" '
        f'svg:x="{LEAD_X:.3f}cm" svg:y="{LEAD_Y:.3f}cm" presentation:class="title" '
        f'presentation:user-transformed="true"><draw:text-box>{paras}'
        f"</draw:text-box></draw:frame>"
    )


# The slide number, where the hand-corrected copy put it. The padding is the
# Default_3_1 graphic style's, which the copy's number frame inherits; it is
# what drops the digit into the middle of the box.
PAGENUM_BOX = (26.593, 14.840, 1.610, 1.336)
PAGENUM_TEXT = (26.867, 14.978, 1.073, 1.186)
PAGENUM_PAD = (0.152, 0.254, 0.152, 0.152)


def pagenum(n):
    """The template's blue rounded box in the bottom-right corner."""
    bx, by, bw, bh = PAGENUM_BOX
    tx, ty, tw, th = PAGENUM_TEXT
    box = roundrect(bx, by, bw, bh, DEEPBLUE)
    txt, _ = textbox(tx, ty, tw, [str(n)], size=14, color=WHITE,
                     align="center", h=th, line_height=100,
                     padding=PAGENUM_PAD, area="left")
    return box + txt


def reference(text, width=24.8):
    """Small blue source line, bottom left — the template's citation style."""
    frame, _ = textbox(MARGIN, 14.86, width, [text], size=11, color=BLUE)
    return frame


def attribution(text):
    frame, _ = textbox(MARGIN, 14.86, 22.0, [text], size=10, color=GREY)
    return frame


# ---------------------------------------------------------------- pages

_pages: list[str] = []
_labels: dict[str, int] = {}
_labels_prev: dict[str, int] = {}


def label(name):
    """Names the slide about to be made, so another slide can cite its number."""
    _labels[name] = len(_pages) + 1


def ref(name):
    """The number of a labelled slide. Right on the second of the two passes."""
    return str(_labels_prev.get(name, "?"))


def _page(content, style="xdpBody", notes="", anims=None):
    n = len(_pages) + 1
    content += appear_on_click(anims)
    notes_xml = (
        f'<presentation:notes draw:style-name="xdpNotes">'
        f'<draw:frame presentation:style-name="xprNotes" draw:layer="layout" '
        f'svg:width="17.271cm" svg:height="12.572cm" svg:x="2.159cm" svg:y="13.271cm" '
        f'presentation:class="notes"><draw:text-box>'
        + "".join(
            f"<text:p>{html.escape(line)}</text:p>" for line in notes.split("\n")
        )
        + "</draw:text-box></draw:frame></presentation:notes>"
    )
    _pages.append(
        f'<draw:page draw:name="page{n}" draw:style-name="{style}" '
        f'draw:master-page-name="Default" '
        f'presentation:presentation-page-layout-name="AL1T1">'
        f"{content}{notes_xml}</draw:page>"
    )


def title_slide(title_lines, authors, affiliation, venue_lines, logos, notes=""):
    parts = []
    t, _ = textbox(1.176, 1.223, 26.5, title_lines, size=45, color=RED,
                   line_height=115)
    parts.append(t)
    a, _ = textbox(1.176, 7.323, 23.86, [authors], size=28, color=GREY)
    parts.append(a)
    af, _ = textbox(1.176, 8.75, 23.86, [affiliation], size=23, color=GREY)
    parts.append(af)
    v, _ = textbox(1.176, 13.023, 23.86, venue_lines, size=20, color=GREY,
                   line_height=115)
    parts.append(v)
    for href, x, y, w, h in logos:
        parts.append(picture(href, x, y, w, h))
    _page("".join(parts), style="xdpTitle", notes=notes)


SMALL = re.compile(r"\{(.+?)\}")


def section(lines, notes=""):
    """The template's triple rounded rectangle, bleeding off the left edge.

    Words in braces are set at two thirds of the size, as the hand-corrected
    copy does with the small function words of a two-line divider.
    """
    parts = [
        roundrect(-3.139, 1.978, 25.653, 11.684, DEEPBLUE),
        roundrect(-2.739, 2.341, 24.891, 10.921, WHITE),
        roundrect(-2.54, 2.54, 24.384, 10.414, DEEPBLUE),
    ]
    n = len(lines)
    size = 60 if n <= 2 else 51
    small = round(size * 2 / 3)
    # Placed by measurement rather than by centring the block: the two-line
    # divider sits 0.8 cm above the one-line one, which is not what centring
    # a 60 pt block would give.
    y = 6.853 - n * 0.800
    ps = _para_style("left", 120, 0.0)
    body = []
    for ln in lines:
        spans, pos = [], 0
        for m in SMALL.finditer(ln):
            if m.start() > pos:
                spans.append((ln[pos:m.start()], _text_style(size, WHITE)))
            spans.append((m.group(1), _text_style(small, WHITE)))
            pos = m.end()
        if pos < len(ln):
            spans.append((ln[pos:], _text_style(size, WHITE)))
        body.append(f'<text:p text:style-name="{ps}">' + "".join(
            f'<text:span text:style-name="{st}">'
            f'{html.escape(t.replace(chr(39), chr(8217)))}</text:span>'
            for t, st in spans) + "</text:p>")
    h = n * size * PT * LINE * 1.2 + 0.2
    parts.append(
        f'<draw:frame draw:style-name="{_frame_style("top")}" draw:layer="layout" '
        f'svg:width="19.458cm" svg:height="{h:.3f}cm" svg:x="1.270cm" '
        f'svg:y="{y:.3f}cm"><draw:text-box>{"".join(body)}</draw:text-box>'
        f"</draw:frame>")
    _page("".join(parts), notes=notes)


def slide(lead=None, body=None, lead_color=BLUE, lead_size=LEAD_SIZE, body_size=20,
          image=None, image_path=None, image_w=None, ref=None, attrib=None,
          extras=None, notes="", body_y=None, gap=0.55, body_space=0.34,
          bottom=15.05, line_height=132, backdrop=None, anims=None):
    """A standard content slide: title, body paragraphs, optional figure.

    Pass `image_path` rather than a pre-fitted `image` to have the figure sized
    into whatever vertical space the text actually left — the text estimator is
    approximate, and a figure that runs off the bottom is the failure it causes.
    """
    parts, y = list(backdrop or []), 0.85
    if lead:
        parts.append(lead_frame(lead, lead_color, lead_size))
        y = BODY_TOP
    if body_y is not None:
        y = body_y
    if body:
        t, h = textbox(MARGIN, y, BODY_W, body, size=body_size, color=BLACK,
                       line_height=line_height, space_after=body_space)
        parts.append(t)
        y += h + gap
    if image_path is not None:
        avail = max(2.0, (bottom - 0.25 if (ref or attrib) else bottom) - y)
        href, iw, ih = fit(image_path, image_w or 22.0, avail)
        parts.append(picture(href, (W - iw) / 2, y, iw, ih))
    elif image:
        href, iw, ih = image
        parts.append(picture(href, (W - iw) / 2, y, iw, ih))
    if extras:
        parts.extend(extras)
    if ref:
        parts.append(reference(ref))
    if attrib:
        parts.append(attribution(attrib))
    parts.append(pagenum(len(_pages) + 1))
    _page("".join(parts), notes=notes, anims=anims)


def full_figure(href, iw, ih, lead=None, ref=None, attrib=None, extras=None,
                notes="", top=0.55, x=None):
    """A slide the figure owns, with at most one line of text above it."""
    parts, y = [], top
    if lead:
        parts.append(lead_frame(lead))
        y = BODY_TOP
    parts.append(picture(href, (W - iw) / 2 if x is None else x, y, iw, ih))
    if extras:
        parts.extend(extras)
    if ref:
        parts.append(reference(ref))
    if attrib:
        parts.append(attribution(attrib))
    parts.append(pagenum(len(_pages) + 1))
    _page("".join(parts), notes=notes)


def credit(text, x=16.656, y=0.462, w=11.078):
    """A figure credit in the top right, right-aligned, as the copy has it."""
    frame, _ = textbox(x, y, w, [text], size=10, color=GREY, align="right")
    return frame


# ---------------------------------------------------------------- pictures

_pics: dict[str, str] = {}


def add_picture(path: Path, max_px=2400) -> tuple[str, float, float]:
    """Copies an image into the package, downscaling it, and returns its aspect."""
    path = Path(path)
    key = str(path)
    im = Image.open(path)
    if key not in _pics:
        jpeg = path.suffix.lower() in (".jpg", ".jpeg")
        ext = "jpg" if jpeg else "png"
        name = f"gnn_{len(_pics):02d}_{re.sub(r'[^A-Za-z0-9]+', '_', path.stem)}.{ext}"
        if max(im.size) > max_px:
            scale = max_px / max(im.size)
            im2 = im.resize((round(im.width * scale), round(im.height * scale)),
                            Image.LANCZOS)
        else:
            im2 = im
        if jpeg:
            im2.convert("RGB").save(BUILD / "Pictures" / name, "JPEG", quality=90)
        else:
            if im2.mode not in ("RGB", "RGBA"):
                im2 = im2.convert("RGB")
            im2.save(BUILD / "Pictures" / name, "PNG", optimize=True)
        _pics[key] = name
    return _pics[key], im.width, im.height


def fit(path, max_w, max_h, max_px=2400):
    """Places an image at the largest size fitting a box, preserving aspect."""
    name, pw, ph = add_picture(path, max_px)
    scale = min(max_w / pw, max_h / ph)
    return name, pw * scale, ph * scale


# ---------------------------------------------------------------- assembly

def unpack_template():
    if not TEMPLATE.exists():
        raise SystemExit(f"carrier not found: {TEMPLATE}\n"
                         "set DECK_TEMPLATE to the hand-corrected weekly deck")
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(BUILD)


def write_content():
    src = (BUILD / "content.xml").read_text(encoding="utf8")
    head, rest = src.split("<office:automatic-styles>", 1)
    auto, tail = rest.split("</office:automatic-styles>", 1)
    # A carrier built by this script carries its own x-named styles (xT24, ...);
    # left in, they come first and win over this deck's styles of the same name.
    auto = re.sub(r'<style:style style:name="x[^"]*".*?</style:style>', "", auto,
                  flags=re.S)
    body_open = tail.index("<office:presentation>") + len("<office:presentation>")

    # Keep the presentation settings element, drop every page, close the document.
    after = tail[body_open:]
    m = re.search(r"<presentation:settings[^>]*/>", after)
    settings = (m.group(0) if m else "") + (
        "</office:presentation></office:body></office:document-content>"
    )

    # Every face a style names must be declared, or Impress quietly uses its
    # default. The carrier declares these; this only guards another carrier.
    for face in ("Palatino Linotype", "Palatino-Bold", SYMBOL_BOLD_FONT):
        if f'style:name="{face}"' not in head:
            head = head.replace(
                "</office:font-face-decls>",
                f'<style:font-face style:name="{face}" svg:font-family='
                f'"&apos;{face}&apos;" style:font-family-generic="roman"/>'
                "</office:font-face-decls>")

    out = (
        head
        + "<office:automatic-styles>"
        + auto
        + _fixed_styles()
        + "".join(_styles)
        + "</office:automatic-styles>"
        + tail[:body_open]
        + "".join(_pages)
        + settings
    )
    (BUILD / "content.xml").write_text(out, encoding="utf8")


def prune():
    """Drops what the carrier brought and this deck does not use.

    The carrier is a finished deck, so it arrives with its own figures and its
    thumbnail. Anything under Pictures/ that neither content.xml nor
    styles.xml names is removed, with its manifest entry.
    """
    used = (BUILD / "content.xml").read_text(encoding="utf8") + \
        (BUILD / "styles.xml").read_text(encoding="utf8")
    man = BUILD / "META-INF/manifest.xml"
    x = man.read_text(encoding="utf8")
    for f in sorted((BUILD / "Pictures").iterdir()):
        if f"Pictures/{f.name}" not in used:
            f.unlink()
            x = re.sub(rf'\s*<manifest:file-entry manifest:full-path="Pictures/'
                       rf'{re.escape(f.name)}"[^>]*/>', "", x)
    thumb = BUILD / "Thumbnails"
    if thumb.exists():
        shutil.rmtree(thumb)
        x = re.sub(r'\s*<manifest:file-entry manifest:full-path="Thumbnails/'
                   r'[^"]*"[^>]*/>', "", x)
    man.write_text(x, encoding="utf8")

    meta = BUILD / "meta.xml"
    if meta.exists():
        mx = meta.read_text(encoding="utf8")
        mx = re.sub(r"<dc:title>.*?</dc:title>", "", mx, flags=re.S)
        mx = mx.replace("<office:meta>",
                        f"<office:meta><dc:title>{html.escape(TITLE)}</dc:title>",
                        1)
        meta.write_text(mx, encoding="utf8")


def write_manifest():
    p = BUILD / "META-INF/manifest.xml"
    x = p.read_text(encoding="utf8")
    entries = "".join(
        f' <manifest:file-entry manifest:full-path="Pictures/{n}" '
        f'manifest:media-type="image/{"jpeg" if n.endswith(".jpg") else "png"}"/>\n'
        for n in _pics.values()
    )
    x = x.replace("</manifest:manifest>", entries + "</manifest:manifest>")
    p.write_text(x, encoding="utf8")


def zip_up():
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/vnd.oasis.opendocument.presentation",
                   compress_type=zipfile.ZIP_STORED)
        for root, _, files in os.walk(BUILD):
            for f in files:
                fp = Path(root) / f
                rel = fp.relative_to(BUILD).as_posix()
                if rel == "mimetype":
                    continue
                z.write(fp, rel)


def _reset():
    _styles.clear()
    _seen.clear()
    _pages.clear()
    _pics.clear()
    PLACEHOLDERS.clear()
    _labels.clear()


def build(deck):
    # Two passes: the first only learns which slide each label landed on, so
    # that a slide can cite a later one by number.
    for _ in range(2):
        _reset()
        unpack_template()
        (BUILD / "Pictures").mkdir(exist_ok=True)
        deck()
        _labels_prev.clear()
        _labels_prev.update(_labels)
    write_content()
    write_manifest()
    prune()
    zip_up()
    shutil.rmtree(BUILD)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, {len(_pages)} slides)")
    if PLACEHOLDERS:
        print(f"\n{len(PLACEHOLDERS)} placeholders still to fill:")
        for n, t in PLACEHOLDERS:
            print(f"  slide {n:2d}: {t}")
