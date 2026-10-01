#!/usr/bin/env python
"""The schematic figures for the GNN Board talk.

Every figure is drawn at the size it occupies on its slide, in centimetres,
so 13 pt here is 13 pt on the projector and nothing is rescaled. The sizes are
the FIG_* constants below; deck.py places each PNG at exactly that size.

    python make_figures.py            # all figures
    python make_figures.py roadmap    # one figure, by name

Palette and face are the deck's own (build_deck.py), so the figures read as
part of the slides rather than as pasted-in plots.

Dates drawn dashed are placeholders to confirm; they are listed at the top of
each function so they can be edited without reading the drawing code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Ellipse)

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

# The deck's palette (build_deck.py).
BLUE = "#0262ab"
DEEPBLUE = "#0066b3"
DARKRED = "#bb0000"
GREEN = "#158466"
GREY = "#666666"
MAGENTA = "#bf0041"
BLACK = "#000000"

# Light fills that sit under those colours.
BLUE_BG = "#e8f0f8"
GREEN_BG = "#e3f2ec"
GREY_BG = "#f0f0f0"
RED_BG = "#f8e1e1"
RULE = "#999999"

DPI = 300

# Every size in pt below is multiplied by this. The figures are read on a
# projector or a shared screen; 1.12 brings the smallest labels to ~12 pt.
FONT_SCALE = 1.12

# Placement sizes on the slide, in cm (width, height).
FIG_DETECTION = (25.4, 8.0)
FIG_LADDER = (25.4, 6.0)
FIG_KM3 = (12.6, 10.6)
FIG_TIMELINE = (25.4, 5.1)
FIG_ROADMAP = (25.4, 9.95)
FIG_DU_CHAIN = (25.4, 6.4)
FIG_PIPELINE = (25.4, 7.2)
FIG_GANTT = (25.4, 7.9)
FIG_STAIRCASE = (11.6, 10.6)
FIG_HERON = (12.0, 10.6)
FIG_SITES = (15.0, 9.4)
FIG_GP300 = (16.2, 11.0)


# ---------------------------------------------------------------- faces

def _face(families, weight=400, style="normal"):
    """The first installed face from `families`, matched by weight and style.

    Palatino Linotype is installed here as Regular, Italic and Bold Italic
    only; the upright bold is a separate family, "Palatino-Bold". So bold is
    looked up by family first and never by asking Linotype for weight 700,
    which would land on Bold Italic.
    """
    for fam in families:
        for f in fm.fontManager.ttflist:
            if f.name == fam and f.style == style and abs(f.weight - weight) < 150:
                return fm.FontProperties(fname=f.fname)
    return fm.FontProperties(family="serif", weight=weight, style=style)


REG = _face(["Palatino Linotype", "TeX Gyre Pagella", "P052", "URW Palladio L",
             "Palatino", "DejaVu Serif"])
BOLD = _face(["Palatino-Bold", "TeX Gyre Pagella", "P052", "URW Palladio L",
              "DejaVu Serif"], weight=700)
ITAL = _face(["Palatino Linotype", "TeX Gyre Pagella", "P052", "DejaVu Serif"],
             style="italic")

plt.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": REG.get_name(),
    "mathtext.it": f"{ITAL.get_name()}:italic",
    "svg.fonttype": "none",
})


# ---------------------------------------------------------------- canvas

def canvas(size):
    """A figure whose data coordinates are centimetres, origin bottom left."""
    w, h = size
    fig = plt.figure(figsize=(w / 2.54, h / 2.54), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(HERE)}")


def text(ax, x, y, s, size=13, color=BLACK, ha="left", va="center", bold=False,
         italic=False, **kw):
    fp = BOLD if bold else (ITAL if italic else REG)
    return ax.text(x, y, s, fontproperties=fp, fontsize=size * FONT_SCALE,
                   color=color,
                   ha=ha, va=va, **kw)


def box(ax, x, y, w, h, title=None, sub=None, sub2=None, fc=BLUE_BG,
        ec=DEEPBLUE, tc=None, sc=GREY, lw=0.9, ls="-", r=0.18, tsize=13,
        ssize=11.5):
    """A rounded box with a bold title and up to two lines of subtitle."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    lines = [s for s in (title, sub, sub2) if s]
    if not lines:
        return
    cx = x + w / 2
    gap = 0.50 * FONT_SCALE
    top = y + h / 2 + gap * (len(lines) - 1) / 2
    for i, s in enumerate(lines):
        is_title = i == 0 and title
        text(ax, cx, top - i * gap, s, size=tsize if is_title else ssize,
             color=(tc or ec) if is_title else sc, ha="center", bold=is_title,
             zorder=3)


def arrow(ax, p, q, color=GREY, lw=1.1, ls="-", head=9, zorder=1):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
        ls=ls, shrinkA=0, shrinkB=0, zorder=zorder))


def line(ax, xs, ys, color=GREY, lw=1.0, ls="-", zorder=1):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=zorder,
            solid_capstyle="butt")


# ---------------------------------------------------------------- figures

def detection():
    """Slide 'The Earth is the target': an Earth-skimming ντ, side view."""
    fig, ax = canvas(FIG_DETECTION)
    ground = 1.55

    # rock below the ground line, and the mountain the tau is born in
    ax.add_patch(Rectangle((0, 0), 25.4, ground, fc="#ecebe6", ec="none"))
    line(ax, [0, 25.4], [ground, ground], color="#8a8780", lw=1.0)
    ax.add_patch(Polygon([(0.9, ground), (4.3, 6.2), (7.7, ground)],
                         closed=True, fc="#d9d6cc", ec="#8a8780", lw=1.0))

    # the neutrino, the tau, the decay
    y_nu0, y_mid, y_dec = 0.45, 2.6, 3.35
    line(ax, [0.0, 4.2], [y_nu0, y_mid], color=GREEN, lw=1.8, ls=(0, (5, 3)))
    line(ax, [4.2, 7.9], [y_mid, y_dec], color=DEEPBLUE, lw=2.2)
    ax.add_patch(Ellipse((7.95, y_dec), 0.26, 0.26, fc=DARKRED, ec="none",
                         zorder=4))

    # the shower: a thin cone, nearly horizontal
    x_end, y_hi, y_lo = 19.6, 4.85, 3.05
    ax.add_patch(Polygon([(8.05, y_dec), (x_end, y_hi), (x_end, y_lo)],
                         closed=True, fc=RED_BG, ec=DARKRED, lw=0.7, zorder=2))

    # antennas, and the radio reaching them
    xs = [12.6, 14.6, 16.6, 18.6, 20.6, 22.6]
    for x in xs:
        line(ax, [x, x], [ground, ground + 0.95], color="#555555", lw=1.4)
        line(ax, [x - 0.32, x + 0.32], [ground + 0.95, ground + 0.95],
             color="#555555", lw=1.4)
    for x in xs[:4]:
        y0 = y_dec + (y_lo - y_dec) * (x - 8.05) / (x_end - 8.05)
        line(ax, [x, x], [y0 - 0.12, ground + 1.1], color=DARKRED, lw=1.1,
             ls=(0, (1.2, 1.8)))

    # an inclined cosmic ray from the other side
    arrow(ax, (24.2, 6.35), (21.5, 3.95), color=GREY, lw=1.6, head=12)

    # labels
    text(ax, 0.4, 7.35, "Earth-skimming ντ converts to τ inside the mountain",
         size=13, color=GREEN)
    text(ax, 6.0, 4.45, "The τ exits and decays", size=13, color=DEEPBLUE)
    text(ax, 14.2, 5.35, "Near-horizontal air shower", size=13, color=DARKRED,
         ha="center")
    text(ax, 25.0, 7.55, "Inclined cosmic ray:", size=13, color=GREY,
         ha="right")
    text(ax, 25.0, 6.95, "Calibration beam and background", size=13,
         color=GREY, ha="right")
    text(ax, 17.6, 0.8,
         "Autonomous antennas · 50–200 MHz · ~1 km apart · solar-powered",
         size=13, color="#444444", ha="center")
    save(fig, "detection_principle")


def ladder():
    """Slide 'GRAND targets the window above 100 PeV': where each technique is strongest.

    Schematic ranges of best sensitivity, not exposure curves.
    """
    fig, ax = canvas(FIG_LADDER)
    x0, x1 = 8.2, 24.4            # the log-energy axis, 1 TeV to 10 EeV
    lo, hi = 3.0, 10.0            # log10(E / GeV)

    def X(logE):
        return x0 + (logE - lo) / (hi - lo) * (x1 - x0)

    rows = [
        ("Optical Cherenkov", "IceCube · KM3NeT · Baikal-GVD · P-ONE",
         [(3.0, 7.0, 1.0), (7.0, 8.5, 0.35)], DEEPBLUE, BLUE_BG),
        ("In-ice radio", "RNO-G · IceCube-Gen2 Radio",
         [(7.0, 10.0, 1.0)], DEEPBLUE, BLUE_BG),
        ("In-air radio", "GRAND · HERON",
         [(7.7, 8.0, 0.35), (8.0, 10.0, 1.0)], GREEN, GREEN_BG),
        ("UHECR array", "Pierre Auger",
         [(8.0, 10.0, 1.0)], GREY, GREY_BG),
    ]
    top = 5.35
    for i, (name, who, segs, col, bg) in enumerate(rows):
        yc = top - i * 1.15
        text(ax, 0.2, yc, name, size=13, bold=True,
             color=GREEN if col == GREEN else BLACK)
        for a, b, alpha in segs:
            ax.add_patch(FancyBboxPatch(
                (X(a), yc - 0.22), X(b) - X(a), 0.44,
                boxstyle="round,pad=0,rounding_size=0.08",
                fc=col, ec="none", alpha=alpha, zorder=2))
        # the names sit to the right of short bars and inside long ones
        seg_a = min(s[0] for s in segs)
        if seg_a < 5:
            text(ax, X(seg_a) + 0.25, yc + 0.02, who, size=11.5, color="white",
                 zorder=3)
        else:
            text(ax, X(seg_a) - 0.25, yc + 0.02, who, size=11.5, color=GREY,
                 ha="right")
    # axis
    ya = 0.95
    line(ax, [x0, x1], [ya, ya], color=RULE, lw=0.8)
    for lg, lab in [(3, "1 TeV"), (6, "1 PeV"), (8, "100 PeV"), (9, "1 EeV"),
                    (10, "10 EeV")]:
        line(ax, [X(lg), X(lg)], [ya, ya - 0.15], color=RULE, lw=0.8)
        text(ax, X(lg), ya - 0.5, lab, size=11.5, color=GREY, ha="center")
    ax.axvline(X(8), ymin=ya / 6.0 + 0.02, ymax=0.97, color=GREEN, lw=0.8,
               ls=(0, (3, 3)), zorder=0)
    text(ax, 0.2, ya - 0.5, "Schematic: range of best sensitivity",
         size=11, color=GREY, italic=True)
    save(fig, "energy_ladder")


def km3():
    """Slide 'Why now': KM3-230213A against the IceCube and Auger limits.

    SCHEMATIC. Shapes and relative placement follow the KM3NeT paper's
    comparison figure only qualitatively; no numbers are read off it, and the
    y axis carries none.
    """
    import numpy as np
    fig, ax = canvas(FIG_KM3)
    ox, oy, w, h = 1.0, 2.25, 10.6, 7.3

    def X(lg):      # log10(E/GeV) from 7 to 10
        return ox + (lg - 7.0) / 3.0 * w

    def Y(v):       # arbitrary units, 0 to 1
        return oy + v * h

    line(ax, [ox, ox + w], [oy, oy], color=RULE, lw=0.8)
    line(ax, [ox, ox], [oy, oy + h], color=RULE, lw=0.8)
    for lg, lab in [(7, "10 PeV"), (8, "100 PeV"), (9, "1 EeV"),
                    (10, "10 EeV")]:
        line(ax, [X(lg)] * 2, [oy, oy - 0.15], color=RULE, lw=0.8)
        text(ax, X(lg), oy - 0.5, lab, size=11.5, color=GREY,
             ha="right" if lg == 10 else "center")
    text(ax, ox + w / 2, oy - 1.1, "Neutrino energy", size=12, color=GREY,
         ha="center")
    text(ax, ox - 0.4, oy + h / 2, "E²Φ per flavour", size=12, color=GREY,
         ha="center", rotation=90)

    lg = np.linspace(7.0, 10.0, 200)
    ice = 0.36 + 0.30 * ((lg - 8.6) / 1.6) ** 2
    aug = 0.28 + 0.33 * ((lg - 9.1) / 1.2) ** 2
    m = lg > 7.9
    ax.plot([X(v) for v in lg], [Y(v) for v in ice], color=DEEPBLUE, lw=1.8)
    ax.plot([X(v) for v in lg[m]], [Y(v) for v in aug[m]], color=GREY,
            lw=1.8, ls=(0, (6, 3)))

    # the event: a flux estimate above both limits
    ex, ey = 8.34, 0.78
    ax.add_patch(Rectangle((X(7.95), Y(ey) - 0.05), X(8.85) - X(7.95), 0.10,
                           fc=DARKRED, ec="none", zorder=4))
    line(ax, [X(ex)] * 2, [Y(0.64), Y(0.93)], color=DARKRED, lw=1.6, zorder=4)
    ax.add_patch(Ellipse((X(ex), Y(ey)), 0.3, 0.3, fc=DARKRED, ec="none",
                         zorder=5))

    tx = X(8.98)
    text(ax, tx, Y(0.92), "2.5–3σ tension", size=12.5, color=DARKRED)
    text(ax, tx, Y(0.85), "If diffuse;", size=12.5, color=DARKRED)
    text(ax, tx, Y(0.76), "Eases if transient", size=12.5, color=GREY)
    # legend
    ly = 0.4
    line(ax, [1.0, 1.8], [ly, ly], color=DARKRED, lw=3)
    text(ax, 2.0, ly, "KM3-230213A", size=11.5)
    line(ax, [5.0, 5.8], [ly, ly], color=DEEPBLUE, lw=1.8)
    text(ax, 6.0, ly, "IceCube", size=11.5)
    line(ax, [8.1, 8.9], [ly, ly], color=GREY, lw=1.8, ls=(0, (6, 3)))
    text(ax, 9.1, ly, "Auger", size=11.5)
    text(ax, ox + w, oy + h + 0.3, "Schematic", size=11, color=GREY,
         italic=True, ha="right")
    save(fig, "km3_tension")



def timeline():
    """Slide 'Fifteen years of autonomous radio detection'. Evenly spaced."""
    fig, ax = canvas(FIG_TIMELINE)
    events = [
        ("2009", "TREND:", "Radio self-trigger"),
        ("2011", "TREND50:", "50 antennas, Tianshan"),
        ("2018", "White paper:", "Science and design"),
        ("2021", "Carbon footprint", "Study published"),
        ("2023", "Three prototypes", "Deployed"),
        ("2024–25", "First CR candidates;", "Auger coincidence"),
        ("2026", "New leadership;", "GP300 growing"),
        ("2027", "First CR spectrum", "In progress"),
    ]
    yl = 2.55
    xa, xb = 2.65, 22.75
    n = len(events)
    line(ax, [0.4, xb - (xb - xa) / (n - 1) / 2], [yl, yl], color=RULE, lw=1.0)
    line(ax, [xb - (xb - xa) / (n - 1) / 2, 25.0], [yl, yl], color=RULE, lw=1.0,
         ls=(0, (3, 3)))
    n = len(events)
    for i, (yr, a, b) in enumerate(events):
        x = xa + i * (xb - xa) / (n - 1)
        up = i % 2 == 0
        future = b == "In progress"
        ax.add_patch(Ellipse((x, yl), 0.30, 0.30, fc="white" if future else GREEN,
                             ec=GREEN, lw=1.4, zorder=3))
        if up:
            line(ax, [x, x], [yl + 0.2, yl + 0.55], color=RULE, lw=0.6)
            text(ax, x, 4.65, yr, size=13, bold=True, color=GREEN, ha="center")
            text(ax, x, 4.05, a, size=11.5, ha="center")
            text(ax, x, 3.5, b, size=11.5, ha="center", color=GREY)
        else:
            line(ax, [x, x], [yl - 0.2, yl - 0.55], color=RULE, lw=0.6)
            text(ax, x, 1.55, yr, size=13, bold=True, color=GREEN, ha="center")
            text(ax, x, 0.95, a, size=11.5, ha="center")
            text(ax, x, 0.4, b, size=11.5, ha="center", color=GREY)
    save(fig, "heritage_timeline")


def roadmap():
    """Slide 'Each stage retires one risk': swimlanes 2023 to 2035.

    Dashed bars are dates to confirm. Edit them here.
    """
    GP65 = (2024.9, 2027.6)              # its data make the ICRC 2027 spectrum
    GP300_FULL = (2027.6, 2029.5)        # to confirm
    HERON = (2026.5, 2032.5)             # to confirm: ERC Synergy, six years
    G10K_START = 2030.0
    FULL = (2033.0, 2034.9)              # to confirm

    fig, ax = canvas(FIG_ROADMAP)
    xl, xr = 3.95, 24.6
    y0, y1 = 2023.0, 2035.0

    def X(t):
        return xl + (t - y0) / (y1 - y0) * (xr - xl)

    lanes = ["Milestones", "Prototypes", "GRANDProto300", "HERON", "GRAND10k",
             "Full GRAND"]
    ytop, dy, bh = 9.1, 1.12, 0.6

    def Yl(i):
        return ytop - i * dy

    for i, name in enumerate(lanes):
        text(ax, 0.2, Yl(i), name, size=13, bold=True,
             color=BLACK if i else DARKRED)
        if i:
            line(ax, [xl, xr], [Yl(i) - dy / 2] * 2, color="#dddddd", lw=0.6)

    def bar(i, a, b, label, col, bg, dashed=False, arrow_end=False, size=11.5):
        ax.add_patch(FancyBboxPatch(
            (X(a), Yl(i) - bh / 2), X(b) - X(a), bh,
            boxstyle="round,pad=0,rounding_size=0.1", fc=bg, ec=col, lw=1.0,
            ls=(0, (4, 2.5)) if dashed else "-", zorder=2))
        text(ax, (X(a) + X(b)) / 2, Yl(i), label, size=size, color=col,
             ha="center", zorder=3)
        if arrow_end:
            arrow(ax, (X(b), Yl(i)), (X(b) + 0.45, Yl(i)), color=col, lw=1.2)

    bar(1, 2023.1, 2027.6, "GP300 · GRAND@Auger · Nançay", GREEN, GREEN_BG)
    bar(2, *GP65, "GP65: 65 units", GREEN, GREEN_BG, size=10.5)
    bar(2, *GP300_FULL, "~300 units", GREEN, GREEN_BG, dashed=True, size=10.5)
    bar(3, *HERON, "HERON · ERC Synergy, six years", DEEPBLUE, BLUE_BG,
        dashed=True)
    bar(4, G10K_START, 2034.3, "GRAND10k · North + South", GREEN, GREEN_BG,
        arrow_end=True)
    bar(5, *FULL, "Multiple arrays", GREY, GREY_BG, dashed=True)

    for t, lab in [(2024.95, "First CRs"), (2027.55, "CR spectrum"),
                   (2030.0, "10k starts")]:
        x = X(t)
        ax.add_patch(Polygon([(x, Yl(0) + 0.28), (x + 0.28, Yl(0)),
                              (x, Yl(0) - 0.28), (x - 0.28, Yl(0))],
                             closed=True, fc=RED_BG, ec=DARKRED, lw=1.0))
        text(ax, x + 0.42, Yl(0), lab, size=11.5, color=DARKRED)
    ax.add_patch(Rectangle((xr - 3.1, Yl(0) - 0.14), 0.5, 0.28, fc="white",
                           ec=GREY, lw=0.8, ls=(0, (3, 2))))
    text(ax, xr - 2.45, Yl(0), "Date to confirm", size=10.5, color=GREY)

    ya = Yl(5) - dy / 2 - 0.05
    line(ax, [xl, xr], [ya, ya], color=RULE, lw=0.8)
    for yr in range(2023, 2036, 2):
        line(ax, [X(yr)] * 2, [ya, ya - 0.12], color=RULE, lw=0.8)
        text(ax, X(yr), ya - 0.42, str(yr), size=11, color=GREY, ha="center")

    # the risk retired at each stage
    stages = [("Prototypes", "Autonomous detection"),
              ("GRANDProto300", "Efficiency, reconstruction"),
              ("HERON", "Neutrino sensitivity"),
              ("GRAND10k", "Discovery, final design")]
    gap = 0.35
    bw = (xr - xl - 3 * gap) / 4
    text(ax, 0.2, 0.95, "Risk retired", size=13, bold=True)
    for k, (t, s) in enumerate(stages):
        box(ax, xl + k * (bw + gap), 0.3, bw, 1.3, t, s, fc="white", ec=RULE,
            tc=BLACK, tsize=12, ssize=10)
    save(fig, "roadmap")


def du_chain():
    """Slide 'Three prototypes, one detector design': the signal chain."""
    fig, ax = canvas(FIG_DU_CHAIN)
    # containers
    ax.add_patch(FancyBboxPatch((0.1, 2.75), 16.35, 3.5,
                                boxstyle="round,pad=0,rounding_size=0.25",
                                fc="none", ec=RULE, lw=0.8, ls=(0, (4, 3))))
    text(ax, 0.45, 5.85, "Detection unit, in the field", size=11.5,
         color=GREY)
    ax.add_patch(FancyBboxPatch((16.95, 0.1), 8.35, 6.15,
                                boxstyle="round,pad=0,rounding_size=0.25",
                                fc="none", ec=RULE, lw=0.8, ls=(0, (4, 3))))
    text(ax, 17.3, 5.85, "Array level", size=11.5, color=GREY)

    bw, bh, y = 4.95, 2.2, 3.15
    box(ax, 0.45, y, bw, bh, "Horizon antenna", "NS, EW, vertical;",
        "LNAs at 3.5 m", fc=GREEN_BG, ec=GREEN)
    box(ax, 5.95, y, bw, bh, "Front-end board", "50–200 MHz,",
        "14-bit, 500 MS/s", fc=GREEN_BG, ec=GREEN)
    box(ax, 11.45, y, bw, bh, "SoC trigger", "FPGA first level,",
        "GPS timestamp", fc=GREEN_BG, ec=GREEN)
    arrow(ax, (0.45 + bw, y + bh / 2), (5.95, y + bh / 2))
    arrow(ax, (5.95 + bw, y + bh / 2), (11.45, y + bh / 2))

    box(ax, 5.95, 0.35, bw, 1.55, "Solar 180 W", "~15 W per unit", fc="white",
        ec=RULE, tc=BLACK)
    arrow(ax, (5.95 + bw / 2, 1.9), (5.95 + bw / 2, y))

    box(ax, 17.35, y, 7.55, bh, "Wi-Fi to central DAQ", "Second level: 3–5 units",
        "in coincidence; 2 µs traces", fc=BLUE_BG, ec=DEEPBLUE)
    box(ax, 17.35, 0.45, 7.55, 1.85, "CC-IN2P3", "Database for all members",
        fc=BLUE_BG, ec=DEEPBLUE)
    arrow(ax, (11.45 + bw, y + bh / 2), (17.35, y + bh / 2))
    arrow(ax, (17.35 + 7.55 / 2, y), (17.35 + 7.55 / 2, 2.3))
    save(fig, "du_signal_chain")


def pipeline():
    """Slide 'Every piece a neutrino search needs': trigger to spectrum."""
    fig, ax = canvas(FIG_PIPELINE)
    bw, bh, gap = 5.55, 1.75, 0.9
    xs = [0.2 + k * (bw + gap) for k in range(4)]
    y_top, y_bot = 5.25, 2.75
    top = [("Unit trigger", "First level, on FPGA"),
           ("Array trigger", "3–5 units coincide"),
           ("Quality cuts", "Noise and RFI"),
           ("CR candidates", "Templates, ML")]
    bot = [("Energy spectrum", "ICRC 2027"),
           ("Exposure", "GP300 simulation"),
           ("Reconstruction", "LDF · ADF · GNN · SBI"),
           ("Electric field", "Antenna deconvolution")]
    for x, (t, s) in zip(xs, top):
        box(ax, x, y_top, bw, bh, t, s, fc=GREEN_BG, ec=GREEN)
    for x, (t, s) in zip(xs, bot):
        red = t == "Energy spectrum"
        box(ax, x, y_bot, bw, bh, t, s, fc=RED_BG if red else BLUE_BG,
            ec=DARKRED if red else DEEPBLUE)
    for k in range(3):
        arrow(ax, (xs[k] + bw, y_top + bh / 2), (xs[k + 1], y_top + bh / 2))
        arrow(ax, (xs[k + 1], y_bot + bh / 2), (xs[k] + bw, y_bot + bh / 2))
    arrow(ax, (xs[3] + bw / 2, y_top), (xs[3] + bw / 2, y_bot + bh))

    vw = bw + 1.9
    vx = xs[2] + bw / 2 - vw / 2
    box(ax, vx, 0.2, vw, 1.7, "GRAND@Auger × Auger",
        "Efficiency, purity, energy scale", fc="white", ec=GREY, tc=BLACK)
    arrow(ax, (xs[2] + bw / 2, 1.9), (xs[2] + bw / 2, y_bot))
    save(fig, "cr_pipeline")


def gantt():
    """Slide 'First cosmic-ray energy spectrum at ICRC 2027'.

    Months counted from October 2026 = 0. Only the December analysis meeting
    and ICRC itself are fixed; everything with fixed=False is a placeholder.
    """
    ROWS = [
        # label, start, end (None = milestone), critical, fixed
        ("Per-antenna calibration", 0.0, 3.6, True, False),
        ("Firmware freeze", 2.2, None, True, False),
        ("GRANDlib revamp", 0.0, 4.5, False, False),
        ("Data taking, GP65", 0.0, 6.6, True, False),
        ("Analysis meeting", 2.5, None, False, True),
        ("Analysis freeze and review", 7.6, None, True, False),
        ("ICRC 2027", 9.4, None, True, True),
    ]
    fig, ax = canvas(FIG_GANTT)
    xl, xr = 8.3, 25.1
    months = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul"]

    def X(m):
        return xl + m / len(months) * (xr - xl)

    yh = 6.7
    dy = 0.74
    ybot = yh - 0.95 - (len(ROWS) - 1) * dy - 0.4
    for k, mo in enumerate(months):
        text(ax, X(k + 0.5), yh, mo, size=11.5, color=GREY, ha="center")
    for k in range(len(months) + 1):
        line(ax, [X(k)] * 2, [ybot, yh - 0.35], color="#e6e6e6", lw=0.6,
             zorder=0)
    text(ax, X(1.5), yh + 0.55, "2026", size=11, color=GREY, ha="center")
    text(ax, X(6.5), yh + 0.55, "2027", size=11, color=GREY, ha="center")
    line(ax, [X(3)] * 2, [yh + 0.3, yh + 0.8], color=RULE, lw=0.6)

    for i, (lab, a, b, crit, fixed) in enumerate(ROWS):
        y = yh - 0.95 - i * dy
        col = DARKRED if crit else GREY
        bg = RED_BG if crit else GREY_BG
        ls = "-" if fixed else (0, (4, 2.5))
        text(ax, 0.2, y, lab, size=12.5)
        if b is None:
            x = X(a)
            ax.add_patch(Polygon([(x, y + 0.26), (x + 0.26, y), (x, y - 0.26),
                                  (x - 0.26, y)], closed=True, fc=bg, ec=col,
                                 lw=1.1, ls=ls, zorder=3))
        else:
            ax.add_patch(FancyBboxPatch(
                (X(a), y - 0.2), X(b) - X(a), 0.4,
                boxstyle="round,pad=0,rounding_size=0.08", fc=bg, ec=col,
                lw=1.0, ls=ls, zorder=2))
    # legend
    ly = 0.3
    for x, fc, ec, ls, lab in [(0.2, RED_BG, DARKRED, "-", "Critical path"),
                               (3.4, GREY_BG, GREY, "-", "Supporting"),
                               (6.3, "white", GREY, (0, (3, 2)),
                                "Date to confirm")]:
        ax.add_patch(Rectangle((x, ly - 0.14), 0.45, 0.28, fc=fc, ec=ec,
                               lw=1.0, ls=ls))
        text(ax, x + 0.6, ly, lab, size=11, color=GREY)
    save(fig, "icrc2027_plan")


def staircase():
    """Slide 'Path to UHE neutrino sensitivity'. Qualitative: no numbers on y."""
    fig, ax = canvas(FIG_STAIRCASE)
    ox, oy, w, h = 0.75, 1.0, 10.7, 9.1
    line(ax, [ox, ox], [oy, oy + h], color=RULE, lw=0.8)
    arrow(ax, (ox, oy), (ox + w, oy), color=RULE, lw=0.8, head=8)
    text(ax, ox + w, oy - 0.45, "Time", size=11.5, color=GREY, ha="right")
    text(ax, ox - 0.35, oy + h / 2, "UHE ν flux sensitivity, lower is better",
         size=11.5, color=GREY, ha="center", rotation=90)

    steps = [
        ("Limits today", "IceCube, Auger", 8.4, GREY),
        ("HERON", "~10×, for transients", 6.5, DEEPBLUE),
        ("GRAND10k", "From 2030", 4.6, GREEN),
        ("Full GRAND", "Multiple arrays", 2.7, GREEN),
    ]
    # treads get wider down the stair, so the longest name fits the last one
    widths = [2.3, 2.45, 2.55, 3.4]
    edges = [ox + sum(widths[:k]) for k in range(len(widths) + 1)]
    for k, (t, s, y, col) in enumerate(steps):
        xa, xb = edges[k], edges[k + 1]
        line(ax, [xa, xb], [y, y], color=col, lw=2.4)
        if k:
            line(ax, [xa, xa], [steps[k - 1][2], y], color=col, lw=2.4)
        # labels above each tread, left-aligned: nothing rises above a tread
        # but its own labels, so they never meet the risers
        text(ax, xa + 0.2, y + 1.0, t, size=12.5, bold=True, color=col)
        text(ax, xa + 0.2, y + 0.42, s, size=11, color=GREY)
    save(fig, "nu_staircase")


def heron():
    """Slide 'HERON': two techniques merge, and feed the GRAND10k design."""
    fig, ax = canvas(FIG_HERON)
    box(ax, 0.1, 8.1, 5.7, 2.3, "GRAND", "Autonomous antennas",
        "Precise reconstruction", fc=GREEN_BG, ec=GREEN)
    box(ax, 6.2, 8.1, 5.7, 2.3, "BEACON", "Phased arrays",
        "Low threshold", fc=GREY_BG, ec=GREY)
    box(ax, 0.6, 3.6, 10.8, 3.0, "HERON", "24 phased stations × 24 antennas",
        "+ 360 standalone antennas", fc=BLUE_BG, ec=DEEPBLUE, tsize=14,
        ssize=12)
    arrow(ax, (2.95, 8.1), (4.3, 6.6), color=GREY, lw=1.2)
    arrow(ax, (9.05, 8.1), (7.7, 6.6), color=GREY, lw=1.2)
    box(ax, 2.6, 0.2, 6.8, 2.0, "GRAND10k", "Hybrid design option",
        fc=GREEN_BG, ec=GREEN)
    arrow(ax, (6.0, 3.6), (6.0, 2.2), color=GREY, lw=1.2)
    text(ax, 6.25, 2.9, "Design input", size=10.5, color=GREY)
    save(fig, "heron_relation")


def sites():
    """Slide 'Sites': where GRAND is, and where GRAND10k most likely goes."""
    data = json.loads((OUT / "data" / "world_110m.json").read_text())
    fig, ax = canvas(FIG_SITES)
    lon0, lon1, lat0, lat1 = -92.0, 150.0, -60.0, 62.0
    mx, my, mw, mh = 0.1, 0.1, 14.8, 9.2

    def P(lon, lat):
        return (mx + (lon - lon0) / (lon1 - lon0) * mw,
                my + (lat - lat0) / (lat1 - lat0) * mh)

    ax.add_patch(Rectangle((mx, my), mw, mh, fc="#f5f8fb", ec="none"))
    for c in data["countries"]:
        polys = (c["coordinates"] if c["type"] == "MultiPolygon"
                 else [c["coordinates"]])
        hot = c["name"] in ("China", "Argentina")
        for poly in polys:
            ring = [P(lo, la) for lo, la in poly[0]]
            ax.add_patch(Polygon(ring, closed=True,
                                 fc="#cfe8db" if hot else "#e4e2dc",
                                 ec="white", lw=0.3))
    ax.set_xlim(0, FIG_SITES[0])
    ax.set_ylim(0, FIG_SITES[1])
    ax.add_patch(Rectangle((mx, my), mw, mh, fc="none", ec=RULE, lw=0.6))

    marks = [
        # name, lon, lat, colour, label dx, dy (cm), ha
        ("GRANDProto300", 93.94, 40.99, GREEN, 0.0, 0.55, "center"),
        ("HERON", -68.5, -31.0, DEEPBLUE, 0.4, 0.2, "left"),
        ("GRAND@Auger", -69.3, -35.2, GREEN, 0.4, -0.3, "left"),
        ("GRAND@Nançay", 2.2, 47.4, GREY, 0.0, 0.5, "center"),
    ]
    for name, lo, la, col, dx, dy, ha in marks:
        x, y = P(lo, la)
        ax.add_patch(Ellipse((x, y), 0.3, 0.3, fc=col, ec="white", lw=0.8,
                             zorder=4))
        text(ax, x + dx, y + dy, name, size=11.5, color=col, ha=ha,
             bold=True, zorder=5)
    x, y = P(104, 31)
    text(ax, x, y, "GRAND10k North,", size=11, color=GREEN, ha="center",
         italic=True)
    text(ax, x, y - 0.45, "most likely", size=11, color=GREEN, ha="center",
         italic=True)
    x, y = P(-69.3, -35.2)
    text(ax, x + 0.4, y - 0.8, "GRAND10k South,", size=11, color=GREEN,
         italic=True)
    text(ax, x + 0.4, y - 1.25, "most likely", size=11, color=GREEN,
         italic=True)
    save(fig, "sites_map")


def gp300_growth():
    """Slide 'GRANDProto300 grows in stages': the planned layout, and what is up.

    Both layouts are digitised from the papers by prepare_src_figures.py:
    the planned 299 units from arXiv:2507.06629 (Fig. 1) and the deployed
    GP300_s13 / GP300_s65 units from arXiv:2509.21306 (Fig. 2). They share
    one frame: every deployed GP65 unit sits within 50 m of a planned site.
    """
    import numpy as np
    planned = np.array(json.loads((OUT / "data" / "gp300_planned_layout.json")
                                  .read_text()))
    dep = json.loads((OUT / "data" / "gp65_gp13_layout.json").read_text())
    gp65 = np.array(dep["gp65"]) / 1000
    gp13 = np.array(dep["gp13"]) / 1000

    d = np.sqrt(((planned[:, None] - planned[None]) ** 2).sum(2))
    np.fill_diagonal(d, 99)
    infill = d.min(1) < 0.7

    w, h = FIG_GP300
    fig = plt.figure(figsize=(w / 2.54, h / 2.54), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    x0, x1, y0, y1 = -8.6, 13.8, -8.7, 9.0
    # keep 1 km = 1 km: fit the box to the figure's aspect
    sx = (x1 - x0) / w
    sy = (y1 - y0) / h
    s = max(sx, sy)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.set_xlim(cx - s * w / 2, cx + s * w / 2)
    ax.set_ylim(cy - s * h / 2, cy + s * h / 2)
    ax.axis("off")

    ms = 5.0
    ax.plot(planned[~infill, 0], planned[~infill, 1], "o", ms=ms, mfc="white",
            mec="#9a9a9a", mew=0.9, zorder=2)
    ax.plot(planned[infill, 0], planned[infill, 1], "o", ms=ms * 0.8,
            mfc="white", mec=GREEN, mew=0.9, zorder=2)
    ax.plot(gp65[:, 0], gp65[:, 1], "o", ms=ms * 0.85, mfc=GREEN, mec=GREEN,
            zorder=3)
    ax.plot(gp13[:, 0], gp13[:, 1], "x", ms=ms * 0.75, color=DARKRED, mew=1.1,
            zorder=3)

    def label(x, y, s, **kw):
        ax.text(x, y, s, fontproperties=kw.pop("fp", REG),
                fontsize=kw.pop("size", 12) * FONT_SCALE, **kw)

    # legend under the array, two columns
    for k, (mk, mfc, mec, txt) in enumerate([
            ("o", GREEN, GREEN, "GP65, operating"),
            ("o", "white", GREEN, "Planned in-fill, 577 m"),
            ("x", DARKRED, DARKRED, "GP13 (2023–25), now in GP65"),
            ("o", "white", "#9a9a9a", "Planned sparse array, 1 km")]):
        lx = -8.2 + (k // 2) * 11.4
        ly = -6.9 - (k % 2) * 1.05
        ax.plot([lx], [ly], mk, ms=ms, mfc=mfc, mec=mec, color=mec, mew=1.0)
        label(lx + 0.45, ly, txt, va="center", color=BLACK, size=11.5)

    # scale bar, top left, where the array leaves room
    bx, by = -8.2, 7.2
    ax.plot([bx, bx + 2], [by, by], color=BLACK, lw=1.4)
    for xx in (bx, bx + 2):
        ax.plot([xx, xx], [by - 0.12, by + 0.12], color=BLACK, lw=1.4)
    label(bx + 1, by + 0.3, "2 km", ha="center", va="bottom", size=11)
    fig.savefig(OUT / "gp300_growth.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print("  figures/gp300_growth.png")


FIG_REACH = (25.4, 8.7)


def reach():
    """Slide 'GRAND opens the window above 100 PeV': GNN members, and GRAND.

    Left, the energies where each detector is most sensitive (schematic, as
    the old ladder); right, when it runs. Dates: ANTARES 2008-2022; KM3NeT
    detection units from 2015-16 (ARCA; ORCA 2017); IceCube complete in 2011;
    Baikal-GVD from 2016; RNO-G deployment from 2021, 12 of 35 stations
    working in 2026 (GNN Monthly, Aug. 2026); P-ONE-1 planned for 2026
    (pacific-neutrino.org).
    HERON and GRAND10k dates are the roadmap's, dashed: to confirm.
    """
    NOW = 2026.75
    rows = [
        # name, [(lgE0, lgE1, alpha)], colour, bg, [(t0, t1, kind)], note
        ("ANTARES", [(1.0, 6.0, 1.0)], GREY, GREY_BG,
         [(2008, 2022, "run")], ("Ended 2022", "left")),
        ("IceCube", [(1.0, 7.0, 1.0), (7.0, 10.0, 0.35)], DEEPBLUE, BLUE_BG,
         [(2011, NOW, "run"), (NOW, 2040, "plan")], None),
        ("KM3NeT", [(0.6, 2.0, 1.0), (3.0, 7.0, 1.0)], DEEPBLUE, BLUE_BG,
         [(2016, NOW, "build"), (NOW, 2040, "plan")], ("ORCA · ARCA", "left")),
        ("Baikal-GVD", [(3.0, 7.0, 1.0)], DEEPBLUE, BLUE_BG,
         [(2016, NOW, "build"), (NOW, 2040, "plan")], None),
        ("P-ONE", [(3.0, 7.0, 1.0)], DEEPBLUE, BLUE_BG,
         [(2026, 2040, "plan")], ("First line 2026", "left")),
        ("RNO-G", [(7.0, 10.0, 1.0)], DEEPBLUE, BLUE_BG,
         [(2021, NOW, "build"), (NOW, 2040, "plan")],
         ("12 of 35 stations", "left")),
        ("HERON", [(7.7, 9.7, 1.0)], GREEN, GREEN_BG,
         [(2026.5, 2032.5, "plan")], ("ERC Synergy", "left")),
        ("GRAND", [(7.7, 8.0, 0.35), (8.0, 11.0, 1.0)], GREEN, GREEN_BG,
         [(2023, NOW, "build"), (2030, 2040, "plan")],
         ("Prototypes, cosmic rays", "left")),
    ]
    fig, ax = canvas(FIG_REACH)
    ex0, ex1, lo, hi = 3.6, 13.2, 0.0, 11.0          # energy panel

    def XE(lg):
        return ex0 + (lg - lo) / (hi - lo) * (ex1 - ex0)

    tx0, tx1, t0, t1 = 14.8, 24.8, 2005.0, 2040.0     # time panel

    def XT(t):
        return tx0 + (t - t0) / (t1 - t0) * (tx1 - tx0)

    top, dy, bh = 7.55, 0.8, 0.42
    text(ax, ex0, 8.4, "Where it is most sensitive", size=12, color=GREY)
    text(ax, tx0, 8.4, "When it runs", size=12, color=GREY)
    for i, (name, segs, col, bg, spans, note) in enumerate(rows):
        yc = top - i * dy
        text(ax, 0.2, yc, name, size=13, bold=True,
             color=GREEN if col == GREEN else BLACK)
        for a, b, alpha in segs:
            ax.add_patch(FancyBboxPatch(
                (XE(a), yc - bh / 2), XE(b) - XE(a), bh,
                boxstyle="round,pad=0,rounding_size=0.08", fc=col, ec="none",
                alpha=alpha, zorder=2))
        for a, b, kind in spans:
            if kind == "run":
                fc, ec, ls, al = col, col, "-", 1.0
            elif kind == "build":
                fc, ec, ls, al = col, col, "-", 0.5
            else:
                fc, ec, ls, al = bg, col, (0, (3, 2)), 1.0
            ax.add_patch(FancyBboxPatch(
                (XT(a), yc - bh / 2), XT(b) - XT(a), bh,
                boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec,
                lw=0.9, ls=ls, alpha=al, zorder=2))
        if note:
            s_, side = note
            if side == "right":
                text(ax, XT(spans[-1][1]) + 0.2, yc, s_, size=10.5, color=GREY)
            else:
                text(ax, XT(spans[0][0]) - 0.2, yc, s_, size=10.5, color=GREY,
                     ha="right")
    yg = top - 7 * dy
    text(ax, (XT(2030) + XT(2040)) / 2, yg, "GRAND10k", size=10.5,
         color=GREEN, ha="center", zorder=3)
    # KM3-230213A on KM3NeT's energy row
    yk = top - 2 * dy
    xk = XE(8.34)
    ax.plot([xk], [yk], marker="*", ms=10, color=DARKRED, zorder=4)
    text(ax, xk + 0.25, yk + 0.02, "KM3-230213A", size=10.5, color=DARKRED)

    ya = top - 7 * dy - 0.62
    for a0, a1 in [(ex0, ex1), (tx0, tx1)]:
        line(ax, [a0, a1], [ya, ya], color=RULE, lw=0.8)
    for lg, lab in [(0, "1 GeV"), (3, "1 TeV"), (6, "1 PeV"), (9, "1 EeV"),
                    (11, "100 EeV")]:
        line(ax, [XE(lg)] * 2, [ya, ya - 0.13], color=RULE, lw=0.8)
        text(ax, XE(lg), ya - 0.45, lab, size=10.5, color=GREY, ha="center")
    ax.plot([XE(8)] * 2, [ya, top + 0.35], color=GREEN, lw=0.8,
            ls=(0, (3, 3)), zorder=0)
    text(ax, XE(8), top + 0.52, "100 PeV", size=10.5, color=GREEN,
         ha="center")
    for yr in range(2005, 2041, 5):
        line(ax, [XT(yr)] * 2, [ya, ya - 0.13], color=RULE, lw=0.8)
        text(ax, XT(yr), ya - 0.45, str(yr), size=10.5, color=GREY,
             ha="center")
    ax.plot([XT(NOW)] * 2, [ya, top + 0.35], color=DARKRED, lw=0.8,
            zorder=0)
    text(ax, XT(NOW), top + 0.52, "Now", size=10.5, color=DARKRED,
         ha="center")
    text(ax, 0.2, 0.22, "Schematic. Solid: running · pale: under construction "
         "· dashed: planned", size=10.5, color=GREY, italic=True)
    save(fig, "gnn_reach")


FIG_GEN2 = (25.4, 8.6)

# Expected neutrino events in ten years, for the benchmark fluxes the two
# forecasts share. GRAND10k: Y. Li, MSc thesis (2026), Table 1 (Earth-skimming
# nu_tau, 1e8-1e11 GeV shower energy, 86-93 deg zenith). IceCube-Gen2 radio:
# Valera, Bustamante & Glaser, PRD 107 (2023) 043019, Table I (all-sky, all
# flavours, 1e7-1e10 GeV). Decisive discovery (<B> > 100) in years: Gen2 from
# the same table (* = atmospheric-muon background only); GRAND10k as the
# thesis groups it (Sec. 5.1.1: models 3 and 5 within ~1 yr; 1, 4, 6 within
# ten; 2 not within ten).
GEN2_ROWS = [
    # label, GRAND10k, Gen2 radio, GRAND10k T_disc, Gen2 T_disc
    ("Fang et al., newborn pulsars", 17.03, 125.38, "~1", "1.9"),
    ("Rodrigues et al., all AGN (source)", 15.10, 107.16, "~1", "1.3"),
    ("Fang & Murase, CR reservoirs", 6.62, 57.41, "< 10", "6.2"),
    (r"IceCube 9.5-yr $\nu_\mu$, extrapolated", 3.24, 26.90, "< 10", "0.3*"),
    ("Rodrigues et al., HL BL Lacs", 2.79, 24.24, "< 10", "13"),
    ("Rodrigues et al., all AGN (cosmogenic)", 0.13, 0.89, "> 10", "> 20"),
]


def gen2():
    """Slide 23: GRAND10k and IceCube-Gen2 radio on the same benchmark fluxes."""
    import math
    fig, ax = canvas(FIG_GEN2)
    x0, x1 = 9.7, 19.7                     # log axis, 0.1 to 300 events
    lo, hi = -1.0, math.log10(300)

    def X(v):
        return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)

    top, dy, bh = 7.2, 1.02, 0.34
    for i, (lab, g, ic, tg, ti) in enumerate(GEN2_ROWS):
        yc = top - i * dy
        text(ax, 0.2, yc, lab, size=12)
        for yy, v, col in [(yc + 0.19, ic, DEEPBLUE), (yc - 0.19, g, GREEN)]:
            ax.add_patch(FancyBboxPatch(
                (x0, yy - bh / 2), X(v) - x0, bh,
                boxstyle="round,pad=0,rounding_size=0.06", fc=col, ec="none",
                zorder=2))
            text(ax, X(v) + 0.15, yy, f"{v:.3g}" if v < 10 else f"{v:.0f}",
                 size=10.5, color="#444444")
        text(ax, 21.5, yc, tg, size=12, color=GREEN, ha="center", bold=True)
        text(ax, 24.1, yc, ti, size=12, color=DEEPBLUE, ha="center",
             bold=True)
    # axis
    ya = top - 5 * dy - 0.65
    line(ax, [x0, x1], [ya, ya], color=RULE, lw=0.8)
    for v in (0.1, 1, 10, 100):
        line(ax, [X(v)] * 2, [ya, ya - 0.13], color=RULE, lw=0.8)
        line(ax, [X(v)] * 2, [ya, top + 0.5], color="#eeeeee", lw=0.6,
             zorder=0)
        text(ax, X(v), ya - 0.45, f"{v:g}", size=11, color=GREY, ha="center")
    text(ax, (x0 + x1) / 2, ya - 1.0, "Expected neutrino events in 10 years",
         size=12, color=GREY, ha="center")
    # column heads and legend
    text(ax, 22.8, top + 1.05, "Decisive discovery (yr)", size=12,
         color=GREY, ha="center")
    text(ax, 21.5, top + 0.55, "GRAND10k", size=10.5, color=GREEN,
         ha="center")
    text(ax, 24.1, top + 0.55, "Gen2 radio", size=10.5, color=DEEPBLUE,
         ha="center")
    for k, (col, lab) in enumerate([(DEEPBLUE, "IceCube-Gen2 radio"),
                                    (GREEN, "GRAND10k")]):
        lx = x0 + k * 5.2
        ax.add_patch(FancyBboxPatch((lx, top + 0.72), 0.55, 0.3,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=col, ec="none"))
        text(ax, lx + 0.75, top + 0.87, lab, size=11.5)
    text(ax, 0.2, top + 0.87, "Benchmark UHE ν flux", size=12, color=GREY)
    for k, s_ in enumerate([
            r"Gen2 radio: all-sky, all flavours, $10^7$–$10^{10}$ GeV",
            r"GRAND10k: Earth-skimming $\nu_\tau$, $10^8$–$10^{11}$ GeV",
            "* Atmospheric-muon background only"]):
        text(ax, 0.2, 1.3 - k * 0.46, s_, size=10, color=GREY)
    save(fig, "gen2_comparison")


FIGURES = {
    "detection": detection, "ladder": ladder, "km3": km3,
    "timeline": timeline, "roadmap": roadmap, "du_chain": du_chain,
    "pipeline": pipeline, "gantt": gantt, "staircase": staircase,
    "heron": heron, "sites": sites, "gp300": gp300_growth,
    "reach": reach, "gen2": gen2,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    print("faces:", REG.get_name(), "/", BOLD.get_name(), "/", ITAL.get_name())
    for n in names:
        FIGURES[n]()
