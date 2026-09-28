# -*- coding: utf-8 -*-
r"""``sim2root.py`` fills ``tshower.xmax_pos`` and ``tshower.direction``.

Both were written as zeros until 2026-09 (grand-mother/grand#104).  They are
now defined as:

* ``xmax_pos``: Xmax in the site frame, the frame of ``du_xyz`` and
  ``shower_core_pos`` -- the ground-relative ``xmax_pos_shc`` plus the core.
  That is the point ``get_simu_parameters`` already hands its consumers as
  ``FIX_xmax_pos``, so filling the field moves nothing.

* ``direction``: the unit vector along which the shower travels.  ``zenith``
  and ``azimuth`` say where it comes *from* (``tests/geo/test_angle_convention.py``),
  so ``direction`` is minus the vector they name, and the same vector the
  ZHAireS converter already writes as ``primary_inj_dir_shc``.

The oracle is again the ZHAireS ``.sry``, not the converter's arithmetic.
Both committed events are run: 13790, at a zenith of 79.4 degrees, is where
Earth curvature separates the frames most.
"""

import numpy as np
import pytest

from grand.dataio.xmax_frame import GROUND, arrival_direction, xmax_above_ground
from tests.sim2root.test_xmax_frame import (ZHAIRES_EVENTS, _convert_zhaires,
                                            _reader_xmax_z, _run_sim2root, _sry_truth,
                                            needs_root)


def _sry_xmax_above_ground(folder):
    r"""Returns Xmax's ``(x, y, z - ground)`` from the ``.sry``, in metres.

    Parameters
    ----------
    folder : pathlib.Path
        A ZHAireS output directory holding exactly one ``.sry``.

    Returns
    -------
    numpy.ndarray
        Shape (3,), relative to the core, ``z`` above the ground.
    """
    import re

    text = next(folder.glob('*.sry')).read_text(encoding='utf-8', errors='replace')
    pos = re.search(r'Pos\. Max\.:' + r'\s+(-?[0-9.eE+-]+)' * 5, text)
    raw_z, ground = _sry_truth(folder)
    return np.array([float(pos.group(3)) * 1000.0, float(pos.group(4)) * 1000.0,
                     raw_z - ground])


@pytest.fixture(scope='module', params=sorted(ZHAIRES_EVENTS))
def fresh(request, tmp_path_factory):
    r"""Converts one committed event end to end and reads its trees back.

    Returns
    -------
    dict
        ``event_id``, ``folder``, the ``efield`` path, and the event's
        ``tshower``, ``tshowersim`` and ``trun`` fields as NumPy values.
    """
    import uproot

    event_id = request.param
    folder = ZHAIRES_EVENTS[event_id]
    if not folder.is_dir():
        pytest.skip('the ZHAireS fixture %s is not present' % folder.name)

    workdir = tmp_path_factory.mktemp('geometry_%d' % event_id)
    _convert_zhaires(folder, event_id, workdir)
    efield = _run_sim2root(workdir / ('raw_%d.root' % event_id), workdir)

    def first(path_glob, tree):
        (path,) = sorted(efield.parent.glob(path_glob))
        t = uproot.open(str(path))[tree]
        return {name: t[name].array(library='np')[0] for name in t.keys()}

    return {
        'event_id': event_id,
        'folder': folder,
        'efield': efield,
        'shower': first('shower_*.root', 'tshower'),
        'showersim': first('showersim_*.root', 'tshowersim'),
        'run': first('run_*.root', 'trun'),
    }


@needs_root
def test_both_fields_are_filled(fresh):
    r"""Neither field is left at the ``(0, 0, 0)`` of issue #104."""
    shower = fresh['shower']
    for name in ('xmax_pos', 'direction'):
        value = np.asarray(shower[name], dtype=float)
        assert value.shape == (3,)
        assert np.all(np.isfinite(value)), '%s is %r' % (name, value)
        assert np.linalg.norm(value) > 0.5, '%s is still unset: %r' % (name, value)


@needs_root
def test_direction_is_the_propagation_vector_of_the_stored_angles(fresh):
    r"""``direction`` is minus the "comes from" vector of zenith and azimuth."""
    from grand.geo.coordinates import _cartesian_to_spherical

    shower = fresh['shower']
    direction = np.asarray(shower['direction'], dtype=float)
    zenith, azimuth = float(shower['zenith']), float(shower['azimuth'])

    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-6)
    assert direction[2] < 0, 'a downgoing shower travels down'
    assert np.allclose(direction, -arrival_direction(zenith, azimuth), atol=1e-6)

    # Back through GRANDlib's own transform: the reversed vector gives the
    # stored angles, as Xmax's position does in test_angle_convention.py.
    theta, phi, _ = _cartesian_to_spherical(*(-direction))
    assert float(np.ravel(theta)[0]) == pytest.approx(zenith, abs=1e-3)
    assert float(np.ravel(phi)[0]) % 360.0 == pytest.approx(azimuth % 360.0, abs=1e-3)

    # The converter's independent writing of the same vector.
    injected = np.asarray(fresh['showersim']['primary_inj_dir_shc'][0], dtype=float)
    assert np.allclose(direction, injected, atol=1e-5)

    # And the .sry: Xmax lies upstream, against the direction of travel.
    upstream = _sry_xmax_above_ground(fresh['folder'])
    cos = direction @ upstream / np.linalg.norm(upstream)
    assert cos == pytest.approx(-1.0, abs=1e-5)


@needs_root
def test_xmax_pos_is_the_ground_relative_xmax_plus_the_core(fresh):
    r"""``xmax_pos - shower_core_pos`` is ``xmax_pos_shc``, and reads as above ground."""
    shower = fresh['shower']
    xmax_pos = np.asarray(shower['xmax_pos'], dtype=float)
    core = np.asarray(shower['shower_core_pos'], dtype=float)
    shc = np.asarray(shower['xmax_pos_shc'], dtype=float)
    ground = float(fresh['run']['origin_geoid'][2])

    assert np.allclose(xmax_pos - core, shc, atol=0.01)

    # The frame detector classifies the offset from the core unambiguously.
    _, frame = xmax_above_ground(xmax_pos - core, shower['zenith'], shower['azimuth'], ground)
    assert frame == GROUND

    # Measured against the simulation's own record.
    truth = _sry_xmax_above_ground(fresh['folder']) + core
    assert np.allclose(xmax_pos, truth, atol=1.0), (
        'xmax_pos is %r; the .sry puts Xmax at %r in the site frame' % (xmax_pos, truth))


@needs_root
def test_the_reader_places_xmax_where_xmax_pos_says(fresh):
    r"""The readers still read ``xmax_pos_shc``; the answer is unchanged and agrees."""
    from grand.dataio.root_files import get_file_event

    reader = get_file_event(str(fresh['efield']))
    for idx in range(64):
        reader.load_event_idx(idx)
        if int(reader.tt_event.event_number) == fresh['event_id']:
            break
    d_simu = reader.get_simu_parameters()

    assert d_simu['xmax_frame'] == GROUND
    assert np.allclose(d_simu['FIX_xmax_pos'], fresh['shower']['xmax_pos'], atol=0.01)
    assert np.allclose(d_simu['xmax_pos'], fresh['shower']['xmax_pos'], atol=0.01)

    raw_z, ground = _sry_truth(fresh['folder'])
    core_z = float(fresh['shower']['shower_core_pos'][2])
    assert _reader_xmax_z(fresh['efield'], fresh['event_id']) == pytest.approx(
        raw_z - ground + core_z, abs=1.0)
