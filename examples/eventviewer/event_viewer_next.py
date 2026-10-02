"""
MIT License.

Copyright (c) 2021 GRAND Collaboration
contact: rkoirala@nju.edu.cn
contact2 (update to root): claire.guepin@lupm.in2p3.fr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
import numpy as np
try:
    import pandas as pd
    # http://holoviews.org/getting_started/index.html
    import panel as pn
    import holoviews as hv
    from bokeh.models import TapTool
    from holoviews import opts, dim
except ImportError as _error:          # a bare ModuleNotFoundError named no remedy (#280)
    raise ImportError("GRANDlib: the event viewer needs its plotting packages (%s): "
                      "pip install -e \".[viewer]\"" % _error.name) from _error

from scipy.signal import hilbert
import scipy.interpolate as scipolate
import mix  # functions written by Valentin Decoene.
import seaborn as sns  # used for color pallettes.

from grand.aoi import EventList
from grand.dataio.xmax_frame import xmax_above_ground


# Defaults at module scope, so that importing this file and constructing an
# EventViewer works.  They used to be assigned only inside the __main__ block
# while methods read them as globals, so `EventViewer()` raised NameError for
# anyone who imported the module instead of running it.
DEFAULT_GEOFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "GP300propsedLayout.dat")
LOGO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "logo_withoutbords.png")

main_width = 750    # width of the main plot
main_height = 700   # height of the main plot
side_width = 350    # width of trace plots
side_height = 300   # height of trace plots
img_width = 380     # width of the side kXB, kX(kXB) image
img_height = 300    # height of the side kXB, kX(kXB) image

color_options = ['Blues', 'Reds', 'RdBu_r', 'RdYlBu_r',
                 'RdYlGn_r', 'Wistia', 'YlGn', 'YlGnBu',
                 'autumn_r', 'cividis_r', 'coolwarm',
                 'copper_r', 'gist_earth_r', 'gnuplot_r',
                 'magma_r', 'mako_r', 'plasma_r', 'rainbow',
                 'seismic', 'summer_r', 'spring', 'terrain_r', 'turbo',
                 'viridis_r', 'vlag', 'winter_r', 'colorblind']


def _tap_only_this_layer(plot, element):
    """Points the figure's tap tool at this element's glyphs alone.

    A HoloViews plot hook.  Called after the Bokeh figure is built, with
    `plot.handles['glyph_renderer']` being the renderer for the element the
    hook is attached to.

    Parameters
    ----------
    plot : holoviews.plotting.bokeh.ElementPlot
        The plot being built.
    element : holoviews.Element
        The element being drawn.  Unused; the hook signature requires it.
    """
    renderer = plot.handles.get('glyph_renderer')
    if renderer is None:
        return
    for tool in plot.state.toolbar.tools:
        if isinstance(tool, TapTool):
            tool.renderers = [renderer]
            