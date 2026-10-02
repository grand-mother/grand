# -*- coding: utf-8 -*-
r"""Every directory of code under ``grand/`` is shipped when GRANDlib is built.

``pyproject.toml`` discovers packages with ``packages.find``, which skips a
directory without an ``__init__.py``.  ``grand/analysis/coords/`` had none:
it worked from a checkout, but a built GRANDlib would have shipped without
it, and ``import grand.analysis`` -- which imports it -- would have failed
for anyone who installed the package.  Found and fixed on 2026-09-24.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_every_code_directory_under_grand_is_a_discovered_package():
    r"""Setuptools finds a package for each directory holding ``.py`` files."""
    import tomllib

    from setuptools import find_packages

    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    include = config['tool']['setuptools']['packages']['find']['include']
    found = set(find_packages(where=str(ROOT), include=include))

    code_dirs = {
        '.'.join(path.parent.relative_to(ROOT).parts)
        for path in (ROOT / 'grand').rglob('*.py')
        if '__pycache__' not in path.parts
    }
    missing = sorted(code_dirs - found)
    assert not missing, (
        'directories of code that a built GRANDlib would not contain: %s. '
        'Each needs an __init__.py.' % missing)


def test_the_data_files_the_code_opens_are_package_data():
    r"""#278: vector_filling.C and rf_chain_config.xml were not package data.

    A wheel built where setuptools does not see the git checkout lacked them:
    ``import grand.dataio`` printed a ROOT error and the first vector branch
    written failed, and ``grand.sim.efield2voltage`` did not import.
    """
    import tomllib

    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    listed = set(config['tool']['setuptools']['package-data']['grand'])
    for name in ('dataio/version', 'dataio/vector_filling.C', 'sim/detector/rf_chain_config.xml'):
        assert name in listed
        assert (ROOT / 'grand' / name).is_file()


def test_a_built_wheel_imports_and_writes(tmp_path):
    r"""#278: the wheel, unpacked away from the source tree, imports and writes a tree."""
    import subprocess
    import sys
    import zipfile

    built = subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(ROOT), '--no-deps',
                            '--no-build-isolation', '-q', '-w', str(tmp_path / 'wheel')],
                           capture_output=True, text=True, timeout=600)
    assert built.returncode == 0, built.stderr[-1500:]
    wheel = next((tmp_path / 'wheel').glob('*.whl'))
    names = zipfile.ZipFile(wheel).namelist()
    for name in ('grand/dataio/vector_filling.C', 'grand/sim/detector/rf_chain_config.xml'):
        assert name in names
    zipfile.ZipFile(wheel).extractall(tmp_path / 'site')
    code = ("import numpy as np, grand\n"
            "assert grand.__file__.startswith(%r)\n"
            "from grand.dataio import TEfield\n"
            "t = TEfield('w.root'); t.run_number = 1; t.event_number = 1; t.du_id = [1]\n"
            "t.trace = np.zeros((1, 3, 4), np.float32); t.fill(); t.write()\n"
            "import grand.sim.efield2voltage\n" % str(tmp_path / 'site'))
    env = {k: v for k, v in __import__('os').environ.items() if k not in ('GRAND_ROOT',)}
    env['PYTHONPATH'] = str(tmp_path / 'site')
    done = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-1500:]
