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
