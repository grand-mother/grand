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
release = '0.0.0.dev'
version = '0.0'

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
suppress_warnings = ['myst.xref_missing']

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
