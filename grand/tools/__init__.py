'''Tools for the GRAND package
'''

from typing_extensions import Final

import os
from pathlib import Path

__all__ = ['DATADIR']


# Initialise the package globals
DATADIR: Final = Path(__file__).parent / 'data'
'''Path to the package data'''
