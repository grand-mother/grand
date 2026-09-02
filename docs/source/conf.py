# -*- coding: utf-8 -*-
r"""Sphinx configuration for the GRANDlib documentation.

Build with::

    pip install -r docs/requirements.txt
    cd docs && make html

The result lands in ``docs/build/html``.

GRANDlib imports ROOT, and the compiled TURTLE and GULL bindings, at module
import time.  ``autodoc`` has to import the package to read its docstrings, so
a documentation build would otherwise require the full runtime environment.
The mocks below stand in for it.  One of them cannot be a plain mock:
``grand/dataio/descriptors.py`` evaluates ``ROOT.gROOT.GetVersionInt()>=63600``
while the module is being imported, so the stand-in has to return a real
integer.  That line is also why the package cannot currently be imported for
any purpose without ROOT present.
"""

import os
import sys
import unittest.mock

sys.path.insert(0, os.path.abspath('../..'))


class _ROOTMock(unittest.mock.MagicMock):
    """A MagicMock whose ROOT version compares as a number."""

    @classmethod
    def _version(cls):
        return 63604          # 6.36.04, the version pinned in env/conda

    def __getattr__(self, name):
        if name == 'GetVersionInt':
            return lambda: self._version()
        return super().__getattr__(name)


for _name in ('ROOT', 'grand._core'):
    sys.modules[_name] = _ROOTMock()

autodoc_mock_imports = ['ROOT', 'h5py', 'uproot', 'awkward', 'psycopg2',
                        'sqlalchemy', 'grand._core']

# -- Project information -----------------------------------------------------

project = 'GRANDlib'
copyright = '2019-2026, The GRAND Collaboration'
author = 'The GRAND Collaboration'
# Taken from the installed package rather than written here.  These were
# hard-coded as 0.0.0.dev / 0.0 and drifted: pyproject.toml said 0.1.0.dev0
# while every rendered page said 0.0.  Two places to update is one too many.
#
# Note that this is the *package* version.  It is deliberately distinct from
# grand/dataio/version, which is the ROOT file-format version and moves for
# entirely different reasons.
def _package_version():
    r"""Returns the installed package version, or reads it from pyproject.toml."""
    try:
        from importlib.metadata import version as _version

        return _version('grand')
    except Exception:
        # Not installed -- read the declaration directly so a docs-only
        # checkout still renders the right number.
        import pathlib
        import re

        text = (pathlib.Path(__file__).resolve().parents[2] / 'pyproject.toml').read_text()
        found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        return found.group(1) if found else 'unknown'


release = _package_version()
#: The short form the sidebar shows: major.minor, or the full string if it is
#: a pre-release, where dropping the suffix would be misleading.
version = release if any(mark in release for mark in ('dev', 'a', 'b', 'rc')) \
    else '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',       # API reference, generated from the docstrings
    'sphinx.ext.napoleon',      # tolerates the legacy :param: docstrings
    'sphinx.ext.mathjax',       # renders LaTeX in the docstrings
    'sphinx.ext.viewcode',      # links API entries to highlighted source
    'sphinx.ext.intersphinx',   # cross-links to the Python and NumPy docs
    'sphinx_copybutton',        # copy-to-clipboard on code blocks
    'myst_parser',              # lets changelog.rst include CHANGELOG.md
    'jupyter_sphinx',           # runs the .. jupyter-execute:: blocks
    'sphinxcontrib.bibtex',     # the References page, from refs.bib
]

# One bibliography file, whose entries come from INSPIRE so that keys and
# metadata match what the literature uses.
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'plain'

# CHANGELOG.md is included by changelog.rst; without this, myst also picks it
# up as a page in its own right and warns that it is not in any toctree.
suppress_warnings = [
    'myst.xref_missing',
    # DANTON (Niess:2018opy) is an arXiv-only note with no journal reference.
    # sphinxcontrib-bibtex warns about the absent field; the field is absent
    # because the work was never published, not because the entry is
    # incomplete.  Warnings are errors in the CI docs build, so this cannot
    # be left to stand.
    'bibtex.missing_field',
]

intersphinx_timeout = 10
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# GRANDlib is class-heavy -- Efield2Voltage, RFChain, the tree classes, the
# coordinate frames -- so class members must be documented.
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'

master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ['_build']

# -- HTML --------------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'GRANDlib %s' % version

# The logo carries the wordmark, so the theme shows it in place of the project
# name in the sidebar.  This is the re-inked copy: the supplied artwork is black
# line work on opaque white, which on the theme's dark sidebar can only be shown
# as a white rectangle.  docs/dev/make_logo_variants.py turns the line work
# white, keeps the rust band, and makes the ground transparent.
html_logo = '_static/grandlib_logo_dark.png'
html_favicon = '_static/favicon.png'

html_css_files = ['custom.css', 'version.css']


def _write_version_css(app):
    r"""Writes the one CSS rule that shows the version under the sidebar logo.

    The theme's own version block is suppressed by ``logo_only``, and the logo
    carries the wordmark but not the version.  Generating the rule here rather
    than typing the number into a stylesheet keeps it in step with the package
    on every build.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The running Sphinx application; only its ``srcdir`` is used.
    """
    import pathlib

    target = pathlib.Path(app.srcdir) / '_static' / 'version.css'
    target.write_text('/* Generated by conf.py; do not edit. */\n'
                      '.wy-side-nav-search::after { content: "%s"; }\n' % release)


def setup(app):
    r"""Sphinx entry point: registers the generated stylesheet."""
    app.connect('builder-inited', _write_version_css)

html_theme_options = {
    'logo_only': True,          # the logo already says "GRANDlib"
    'navigation_depth': 3,      # deep enough to reach the handbook subsections
    'collapse_navigation': False,
}
