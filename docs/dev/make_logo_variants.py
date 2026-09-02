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


def band_mask(rust, dark, white):
    r"""Returns a mask of the rust band, its outline and its interior.

    The band is a trapezoid whose sides slope outwards, drawn with a black
    outline along its top and bottom edges.  Three things have to be inside
    this mask and nothing else:

    - the rust fill itself;
    - the black outline above and below it, which must become **rust** rather
      than white -- inverted to white it reads as a bright stripe spilling out
      of the band, above and below;
    - the white "lib" lettering, which must stay white.

    Getting the extent wrong is what produced the earlier artefacts.  Taking
    the band as a full-width strip of rows keeps the white *outside* the
    trapezoid's sloping sides, which appears as wedges in the bottom corners.

    Parameters
    ----------
    rust, dark, white : ndarray of bool
        The masks from :func:`classify`.

    Returns
    -------
    ndarray of bool
        True inside the band, following its slope row by row.
    """
    h, w = rust.shape
    rows = np.where(rust.mean(axis=1) > 0.25)[0]
    if not rows.size:
        return np.zeros((h, w), dtype=bool)

    top, bottom = int(rows.min()), int(rows.max())

    # Extend over the outline.  Walk outwards while the row is not yet mostly
    # ground.
    #
    # The test is on ground rather than on line work, and that matters: the row
    # where the outline meets the white above it is almost entirely
    # *antialiased*, counting as neither dark nor white -- 342 of 425 pixels on
    # this artwork.  A "mostly dark" test stops one row short there and leaves
    # a white hairline along the top of the band.
    def is_ground(y):
        return white[y].sum() > 0.5 * w

    while top - 1 >= 0 and not is_ground(top - 1):
        top -= 1
    while bottom + 1 < h and not is_ground(bottom + 1):
        bottom += 1

    mask = np.zeros((h, w), dtype=bool)
    ink = rust | dark
    for y in range(top, bottom + 1):
        xs = np.where(ink[y])[0]
        if xs.size:
            mask[y, xs.min():xs.max() + 1] = True
    return mask


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
        band solid.
    """
    a = np.asarray(src).astype(int)
    rust, dark, white = classify(a)
    h, w, _ = a.shape

    band = band_mask(rust, dark, white)
    lifted = np.array([196, 88, 58], dtype=np.uint8)

    out = np.zeros((h, w, 4), dtype=np.uint8)

    # Outside the band: line work becomes white, ground disappears.
    out[dark & ~band] = (255, 255, 255, 255)

    # Inside the band: the fill and its outline are both rust, so the band is a
    # single solid shape with no bright edge; the lettering stays white.
    out[(rust | dark) & band] = np.concatenate([lifted, [255]])
    out[white & band] = (255, 255, 255, 255)
    out[rust & ~band] = np.concatenate([lifted, [255]])      # the rays

    # Ground is transparent everywhere, inside the band's rows included -- that
    # is what keeps the corners outside the trapezoid clear.
    out[white & ~band] = (0, 0, 0, 0)

    # Antialiased edges: carry their darkness as opacity so the line work keeps
    # a smooth edge instead of turning into a staircase.  Inside the band they
    # are edges of the lettering against rust, so they blend to rust instead.
    other = ~(rust | dark | white)
    if other.any():
        lum = a.mean(axis=2)
        alpha = np.clip((235.0 - lum) / 120.0, 0.0, 1.0) * 255
        edge_out = other & ~band
        out[edge_out] = np.stack([np.full(alpha[edge_out].shape, 255),
                                  np.full(alpha[edge_out].shape, 255),
                                  np.full(alpha[edge_out].shape, 255),
                                  alpha[edge_out]], axis=-1).astype(np.uint8)
        edge_in = other & band
        out[edge_in] = np.concatenate([lifted, [255]])

    return Image.fromarray(out, 'RGBA')


def main():
    r"""Writes the logo variants into ``docs/source/_static``."""
    src = Image.open(os.path.join(STATIC, 'grandlib_logo_trimmed.png')).convert('RGB')
    path = os.path.join(STATIC, 'grandlib_logo_dark.png')
    recolour(src).save(path)
    print('wrote %s' % path)


if __name__ == '__main__':
    main()
