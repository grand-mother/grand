#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Derives sidebar-friendly variants of the GRANDlib logo.

    python docs/dev/make_logo_variants.py

The supplied logo is dark artwork on an opaque white field: black line work and
"GRAND" lettering, a rust band carrying "lib" in white, and orange rays.  On the
Read the Docs sidebar, which is dark, it can only be shown as a white rectangle
unless it is re-inked.

Simply making white transparent does not work: the "GRAND" lettering is black
and would vanish, and the "lib" lettering is white and would become holes.  So
the pixels are classified by ink and remapped:

===================  ==========================================================
Class                Treatment
===================  ==========================================================
rust band and rays   kept, and lightened slightly so it holds up on dark grey
black line work      turned white
white ground         made transparent, **except** inside the rust band, where it
                     is the "lib" lettering and stays white
===================  ==========================================================

Classification is by chroma before luminance, because the rust band is dark:
RGB (149, 57, 35) has a luminance of 80 and would otherwise be swept up with
the black line work and turned white.

``grandlib_logo_dark.png`` is what the theme uses;
``grandlib_logo_trimmed.png`` remains for light backgrounds.

A third variant with the rays removed was tried and discarded.  Deleting the
rust ray pixels leaves their antialiased edges behind, and those are classified
as line work, so they become white streaks -- worse at every size than keeping
the rays.
"""

import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(HERE, 'source', '_static')


def classify(a):
    r"""Returns boolean masks for the rust, dark and white parts of `a`.

    Parameters
    ----------
    a : ndarray, shape (h, w, 3)
        The image as integers.

    Returns
    -------
    rust, dark, white : ndarray of bool
        Disjoint masks.  Anything in none of them is an antialiased edge.
    """
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = a.mean(axis=2)
    # Chroma first: the rust band is dark, so a luminance test would claim it.
    rust = (r - b > 60) & (r - g > 40)
    dark = (lum < 120) & ~rust
    white = (lum > 215) & ~rust
    return rust, dark, white


def recolour(src):
    r"""Returns `src` re-inked for a dark background, as RGBA.

    Parameters
    ----------
    src : PIL.Image.Image
        The trimmed logo, RGB.

    Returns
    -------
    PIL.Image.Image
        RGBA, with the ground transparent, the line work white and the rust
        band lifted.
    """
    a = np.asarray(src).astype(int)
    rust, dark, white = classify(a)
    h, w, _ = a.shape

    # The rust band is the contiguous run of rows that are substantially rust;
    # white inside it is the "lib" lettering and must stay white.
    rows = rust.mean(axis=1)
    band_rows = np.where(rows > 0.25)[0]
    in_band = np.zeros((h, w), dtype=bool)
    if band_rows.size:
        in_band[band_rows.min():band_rows.max() + 1] = True

    out = np.zeros((h, w, 4), dtype=np.uint8)

    # Rust, lifted towards the orange end so it reads against dark grey.
    lifted = np.array([196, 88, 58], dtype=np.uint8)
    out[rust] = np.concatenate([lifted, [255]])

    # Black line work and "GRAND" become white.
    out[dark] = (255, 255, 255, 255)

    # White ground disappears, except the lettering inside the band.
    lib = white & in_band
    out[lib] = (255, 255, 255, 255)
    ground = white & ~in_band
    out[ground] = (0, 0, 0, 0)

    # Antialiased edges: carry their darkness as opacity so the line work keeps
    # its smooth edge instead of turning into a staircase.
    other = ~(rust | dark | white)
    if other.any():
        lum = a.mean(axis=2)[other]
        alpha = np.clip((235.0 - lum) / 120.0, 0.0, 1.0) * 255
        out[other] = np.stack([np.full(alpha.shape, 255),
                               np.full(alpha.shape, 255),
                               np.full(alpha.shape, 255),
                               alpha], axis=-1).astype(np.uint8)

    return Image.fromarray(out, 'RGBA')


def main():
    r"""Writes the logo variants into ``docs/source/_static``."""
    src = Image.open(os.path.join(STATIC, 'grandlib_logo_trimmed.png')).convert('RGB')
    path = os.path.join(STATIC, 'grandlib_logo_dark.png')
    recolour(src).save(path)
    print('wrote %s' % path)


if __name__ == '__main__':
    main()
