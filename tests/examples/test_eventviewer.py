# -*- coding: utf-8 -*-
r"""The event viewer still builds, against real data.

``examples/eventviewer/`` is a Panel/HoloViews web tool: it draws the GP300
array, colours the antennas that were hit by their peak time, and lets you
click one to see its trace.  It came from Ramesh Koirala's EventViewer and was
rewritten against :mod:`grand.aoi` by Claire Guepin in May 2025.

**Why this file exists.**  Nothing else in the repository tests ``examples/``,
and this viewer had already stopped being runnable by anyone but its author --
a hard-coded event index, a data path on the author's own machine, and a class
that raised ``NameError`` if you imported it rather than running it.  Those are
fixed; without a test they would come back, and nobody would find out until the
next person tried to open it.

The test builds the whole interface and renders it, which is what forces the
interactive callbacks to run.  It does *not* check that the physics on display
is right: the interpolated amplitude maps and the shower-plane projection are
unreviewed, and a green tick here should not be read as saying otherwise.

Skipped unless the optional viewer dependencies are installed::

    pip install -e ".[viewer]"
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
VIEWER = ROOT / 'examples' / 'eventviewer' / 'event_viewer_to_root.py'

#: A run committed to the repository, small enough to build in a second.
SAMPLE = (ROOT / 'sim2root' / 'Common'
          / 'sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000')

needs_viewer = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None
        for name in ('panel', 'holoviews', 'bokeh', 'pandas', 'seaborn')),
    reason='the viewer extra is not installed: pip install -e ".[viewer]"')

needs_sample = pytest.mark.skipif(
    not SAMPLE.is_dir(), reason='the sample run is not present')


def _viewer_module():
    r"""Imports the viewer by path.

    Returns
    -------
    module
        ``examples/eventviewer/event_viewer_to_root.py``.  It lives in
        ``examples/`` rather than in a package, so there is nothing to import
        by name.  ``mix``, which it imports as a bare sibling, is put on the
        path first.
    """
    import sys

    directory = str(VIEWER.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)

    import holoviews as holo
    holo.extension('bokeh')

    spec = importlib.util.spec_from_file_location('event_viewer_to_root',
                                                  VIEWER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_viewer
def test_the_viewer_imports_and_constructs_without_running_as_main():
    r"""Constructing it must not need names that only ``__main__`` defines.

    ``geofile``, ``datadir`` and the plot sizes used to be assigned inside the
    ``if __name__ == '__main__'`` block while methods read them as globals, so
    ``EventViewer()`` raised ``NameError`` for anyone who imported it.
    """
    module = _viewer_module()
    viewer = module.EventViewer(datadir='unused-here')

    assert viewer.geofile.endswith('.dat')
    assert pathlib.Path(viewer.geofile).is_file(), (
        'the default layout file is resolved relative to the script, so it '
        'must not depend on the working directory')
    assert viewer.host == 'localhost', (
        'the default must not publish the viewer to the network')


@needs_viewer
def test_astropy_is_not_a_dependency_of_the_viewer():
    r"""Pinned, because astropy was removed from GRANDlib deliberately.

    The viewer imported it for one line, ``Time(t0).jd``, whose result was
    never read again.  Reintroducing it here would quietly undo that removal
    for the sake of a value nothing uses.
    """
    source = VIEWER.read_text(encoding='utf-8')
    assert 'astropy' not in source, (
        'the viewer imports astropy again; it was dropped in September 2026 '
        'and the value it computed was unused')


@needs_viewer
@needs_sample
def test_the_whole_interface_builds_and_renders(tmp_path):
    r"""The real check: build the layout against a real run, and render it.

    Rendering is what makes this worth having.  Building the ``GridSpec``
    alone would pass even if every plot callback were broken, because they are
    ``DynamicMap``s and do not run until something asks them to draw.
    """
    module = _viewer_module()
    viewer = module.EventViewer(datadir=str(SAMPLE), event=0)

    layout = viewer.view(serve=False)
    assert layout is not None, 'view(serve=False) must return the layout'
    assert len(layout.objects) > 5, (
        'the grid came back nearly empty: %d objects' % len(layout.objects))

    target = tmp_path / 'viewer.html'
    layout.save(str(target))
    assert target.stat().st_size > 100_000, (
        'the rendered page is %d bytes, too small to contain the plots'
        % target.stat().st_size)


@needs_viewer
@needs_sample
def test_an_out_of_range_event_says_so_instead_of_failing_obscurely():
    r"""``--event 999`` on a two-event run must explain itself.

    The index used to be hard-coded to 862, and a shorter run died inside
    numpy with "zero-size array to reduction operation minimum", which names
    neither the event nor the run.
    """
    module = _viewer_module()
    viewer = module.EventViewer(datadir=str(SAMPLE), event=999)

    with pytest.raises(SystemExit) as raised:
        viewer.view(serve=False)

    message = str(raised.value)
    assert '999' in message and 'event' in message.lower()
