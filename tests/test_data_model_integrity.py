# -*- coding: utf-8 -*-
r"""Version-controlled data files survive the data-model download.

``data/detector``, ``data/noise`` and ``data/topography`` are filled from a
976 MB archive fetched by ``data/download_data_grand.py``, which
``env/setup.sh`` runs.  That script used to delete those three directories
outright before extracting, on the assumption that the archive owned
everything in them.

It does not.  The galactic-noise :math:`P_L` tables are committed to the
repository instead, deliberately, so that they version with the code that
reads them -- and the archive has never heard of them.  A plain delete
removed them and nothing put them back.

That is not a hypothetical.  It broke the documentation build on the day the
tables landed, with ``FileNotFoundError`` on a file that was plainly in git,
and it would have broken any fresh clone whose data-model version differed
from the installed one.  Locally it was invisible: the downloader
short-circuits when the version already matches, so the delete never ran.

The test below is the invariant that failure violated.
"""

import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The directories the download archive extracts over.
MANAGED = ('detector', 'noise', 'topography')


def _tracked(directory):
    r"""Returns the version-controlled files under ``data/<directory>``.

    Parameters
    ----------
    directory : str
        Name below ``data/``.

    Returns
    -------
    list of pathlib.Path
        Absolute paths, or an empty list outside a git checkout.
    """
    try:
        out = subprocess.run(['git', 'ls-files', '--', 'data/%s' % directory],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        pytest.skip('not a git checkout, so nothing is version-controlled')
    return [ROOT / line for line in out.stdout.split() if line.strip()]


@pytest.mark.parametrize('directory', MANAGED)
def test_tracked_files_in_managed_directories_exist(directory):
    r"""Every file git tracks under a managed directory is actually present.

    This is what the documentation build discovered the hard way.  A file can
    be committed, pass review, and still be absent at runtime, because a setup
    step deleted the directory it lives in and restored only what the archive
    contains.
    """
    missing = [p for p in _tracked(directory) if not p.exists()]
    assert not missing, (
        '%d version-controlled file(s) under data/%s are not on disk: %s. '
        'Something removed them -- most likely data/download_data_grand.py, '
        'which empties this directory before extracting the archive.'
        % (len(missing), directory, [str(p.relative_to(ROOT)) for p in missing]))


def test_the_downloader_preserves_version_controlled_files():
    r"""The download script sets tracked files aside rather than deleting them.

    Read statically: running the script means fetching 976 MB.  The property
    is about the source anyway -- that the delete is bracketed by a save and a
    restore, rather than standing alone as it used to.
    """
    source = (ROOT / 'data' / 'download_data_grand.py').read_text()

    assert 'shutil.rmtree' in source, (
        'the downloader no longer deletes anything; if that is deliberate '
        'this test and its docstring are stale')
    for expected in ('_tracked_files', 'preserved'):
        assert expected in source, (
            'the downloader deletes the managed directories without '
            'preserving version-controlled files: %r is gone' % expected)

    delete_at = source.index('shutil.rmtree')
    save_at = source.index('preserved[rel] = handle.read()')
    restore_at = source.index('for rel, blob in preserved.items()')
    assert save_at < delete_at < restore_at, (
        'the save/delete/restore order is wrong: files are read at %d, '
        'deleted at %d, restored at %d' % (save_at, delete_at, restore_at))


def test_the_galactic_noise_tables_are_among_them():
    r"""The tables that prompted this are tracked, and there are three.

    Named explicitly because they are the case that failed, and because a
    fourth antenna model would want one too.
    """
    names = {p.name for p in _tracked('noise')}
    expected = {'galactic_PL_per_Hz_gp13_GP300.npy',
                'galactic_PL_per_Hz_gp13_GP300_nec.npy',
                'galactic_PL_per_Hz_gp13_GP300_mat.npy'}
    assert expected <= names, (
        'the P_L tables are no longer version-controlled: missing %s'
        % sorted(expected - names))
