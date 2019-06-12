import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
sys.path.insert(0, str(Path(__file__).parent.resolve()))


# Project information
project = "GRAND"
copyright = "2019, The GRAND collaboration"
author = "The GRAND collaboration"


# General configuration
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'patch',
]

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    "python": ('https://docs.python.org/3/', None),
    "numpy": ('http://docs.scipy.org/doc/numpy/', None),
    "scipy": ('http://docs.scipy.org/doc/scipy/reference/', None),
    "matplotlib": ('http://matplotlib.org/', None),
    "astropy": ('http://docs.astropy.org/en/stable/', None)
}


# HTML configuration
html_theme = 'python_docs_theme'
html_theme_options = {
    'collapsiblesidebar': True
}

add_module_names = True
html_static_path = ['_static']
