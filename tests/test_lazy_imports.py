# -*- coding: utf-8 -*-
r"""Reading GRAND files must not require the physics to be built.

``grand/__init__.py`` used to import the geometry and simulation code
eagerly, so ``import grand.dataio`` loaded 13 ``grand.sim`` modules, 6
``grand.geo`` modules and the compiled C core -- and without TURTLE and GULL
built, failed with ``No module named 'grand._core'``.  The ``grandio_light``
branch proposed deleting the physics to get an I/O-only package; the
collaboration chose lazy loading instead (resources/dev/dev-next/DECISIONS.md,
decided 2026-09-24).

Each check runs in a fresh interpreter, because what is at stake is what an
import *loads*, and a test process has usually loaded everything already.
"""

import pathlib
import subprocess
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Makes ``grand._core`` unimportable, as on a machine where ``env/setup.sh``
#: never compiled the C extensions.
_BLOCK_CORE = textwrap.dedent('''
    import importlib.abc, sys
    class _NoCore(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == 'grand._core':
                raise ModuleNotFoundError("No module named 'grand._core'", name=name)
    sys.meta_path.insert(0, _NoCore())
''')


def _run(code):
    r"""Runs ``code`` in a fresh interpreter and returns the result.

    Parameters
    ----------
    code : str
        Python source to execute, with the repository importable.

    Returns
    -------
    subprocess.CompletedProcess
        The finished process, with text output captured.
    """
    return subprocess.run([sys.executable, '-c', code], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=600)


def test_the_data_layer_imports_without_the_compiled_core():
    r"""``grand.dataio`` loads, and reads its tree classes, with no C core."""
    done = _run(_BLOCK_CORE + textwrap.dedent('''
        import sys
        import grand.dataio
        from grand.dataio import TRun, TShower, TEfield, TADC
        loaded = sorted({m.split('.')[1] for m in sys.modules
                         if m.startswith('grand.')})
        print(','.join(loaded))
    '''))
    assert done.returncode == 0, (
        'import grand.dataio failed without the compiled core; the physics is '
        'being imported eagerly again:\n' + done.stderr[-2000:])
    loaded = done.stdout.strip().split(',')
    assert 'sim' not in loaded and 'geo' not in loaded, (
        'import grand.dataio loaded %s; it should load the data layer only'
        % loaded)


def test_every_public_name_still_resolves():
    r"""Laziness changes when names load, not which names exist."""
    done = _run(textwrap.dedent('''
        import grand
        missing = []
        for name in grand.__all__:
            try:
                getattr(grand, name)
            except AttributeError:
                missing.append(name)
        print(missing)
    '''))
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == '[]', (
        'names in grand.__all__ that no longer resolve: %s' % done.stdout)


def test_star_import_works():
    r"""``from grand import *`` loads everything in ``__all__``.

    Before 2026-09-24 it raised ``AttributeError`` on ``adc``, which was in
    ``__all__`` but never imported.
    """
    done = _run('from grand import *\nprint(Efield2Voltage.__name__, adc.__name__)')
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == ['Efield2Voltage', 'grand.sim.detector.adc']


def test_subpackages_are_reachable_after_a_bare_import():
    r"""``import grand`` then ``grand.sim`` worked before, as a side effect."""
    done = _run('import grand\nprint(grand.sim.__name__, grand.geo.__name__)')
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == ['grand.sim', 'grand.geo']
