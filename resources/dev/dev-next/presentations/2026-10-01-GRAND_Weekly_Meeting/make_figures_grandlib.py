#!/usr/bin/env python
"""Plots for the GRANDlib weekly-meeting talk of 1 October 2026.

Same faces, palette and sizing rule as make_figures.py: drawn at the size the
figure occupies on the slide, in cm, so the font sizes are the projected ones.

    python make_figures_grandlib.py

Numbers are from resources/dev/dev-next/beta-tests/TEST_PLAN.md on dev-next-ipfxhh
(PR #208), 1 October 2026, 11:40.
"""

import matplotlib.pyplot as plt

from make_figures import (
    BLUE, DEEPBLUE, DARKRED, GREEN, GREY, BLACK, RULE, BOLD, REG,
    FONT_SCALE, DPI, save,
)

FIG_SEVERITY = (14.2, 9.6)
FIG_VALIDATION = (25.4, 8.2)

FIXED_C = GREEN
OPEN_C = "#c9c9c9"


def _axes(size, rect):
    w, h = size
    fig = plt.figure(figsize=(w / 2.54, h / 2.54), dpi=DPI)
    ax = fig.add_axes(rect)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=GREY, length=3)
    return fig, ax


def _label(ax, x, y, s, size=12, color=BLACK, bold=False, **kw):
    ax.text(x, y, s, fontproperties=BOLD if bold else REG,
            fontsize=size * FONT_SCALE, color=color, **kw)


def severity():
    """Beta-test issues by severity: fixed (in PR #208) and open."""
    sev = ["Critical", "High", "Medium", "Low"]
    fixed = [11, 5, 0, 0]
    total = [11, 31, 38, 11]
    opened = [t - f for t, f in zip(total, fixed)]
    fig, ax = _axes(FIG_SEVERITY, [0.22, 0.13, 0.75, 0.80])
    y = range(len(sev))[::-1]
    ax.barh(y, fixed, color=FIXED_C, height=0.62, label="fixed (PR #208)")
    ax.barh(y, opened, left=fixed, color=OPEN_C, height=0.62, label="open")
    for yi, f, t in zip(y, fixed, total):
        _label(ax, t + 0.8, yi, f"{t}", size=13, bold=True, va="center")
        if f:
            _label(ax, f / 2, yi, f"{f}", size=12, color="white", bold=True,
                   va="center", ha="center")
    ax.set_yticks(list(y))
    ax.set_yticklabels(sev)
    for t, c in zip(ax.get_yticklabels(), [DARKRED, BLACK, BLACK, BLACK]):
        t.set_fontproperties(BOLD)
        t.set_fontsize(13 * FONT_SCALE)
        t.set_color(c)
    for t in ax.get_xticklabels():
        t.set_fontproperties(REG)
        t.set_fontsize(11 * FONT_SCALE)
    ax.set_xlim(0, 44)
    ax.set_xlabel("issues", fontproperties=REG, fontsize=12 * FONT_SCALE, color=GREY)
    leg = ax.legend(loc="lower right", frameon=False,
                    prop=REG.copy() if False else None)
    for t in leg.get_texts():
        t.set_fontproperties(REG)
        t.set_fontsize(12 * FONT_SCALE)
    save(fig, "grandlib_severity")


def validation():
    """60 bad inputs, before and after the validation PR #179."""
    cats = [
        ("Refused, clear GRANDlib: message", 7, 50, GREEN),
        ("Accepted, with a GRANDlibWarning", 0, 5, BLUE),
        ("Accepted silently, by design", 0, 5, "#9fb9d6"),
        ("Error by accident, or deep crash", 25, 0, "#e3a5a5"),
        ("Accepted silently, wrong result", 27, 0, DARKRED),
        ("exit()", 1, 0, BLACK),
    ]
    fig, ax = _axes(FIG_VALIDATION, [0.10, 0.42, 0.88, 0.55])
    for row, idx in ((1, 1), (0, 2)):
        left = 0
        for name, b, a, c in cats:
            v = (b, a)[idx - 1]
            if v:
                ax.barh(row, v, left=left, color=c, height=0.6)
                if v >= 4:
                    _label(ax, left + v / 2, row, str(v), size=12,
                           color="white", bold=True, ha="center", va="center")
                left += v
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["before", "after"])
    for t in ax.get_yticklabels():
        t.set_fontproperties(BOLD)
        t.set_fontsize(13 * FONT_SCALE)
    for t in ax.get_xticklabels():
        t.set_fontproperties(REG)
        t.set_fontsize(11 * FONT_SCALE)
    ax.set_xlim(0, 60)
    ax.set_xlabel("60 bad inputs given to public functions", fontproperties=REG,
                  fontsize=12 * FONT_SCALE, color=GREY)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, _, c in cats]
    leg = fig.legend(handles, [n for n, *_ in cats], loc="lower center", ncol=2,
                     frameon=False, bbox_to_anchor=(0.54, -0.02), columnspacing=3.0)
    for t in leg.get_texts():
        t.set_fontproperties(REG)
        t.set_fontsize(11 * FONT_SCALE)
    save(fig, "grandlib_validation")


FIGURES = {"severity": severity, "validation": validation}

if __name__ == "__main__":
    for f in FIGURES.values():
        f()
