# -*- coding: utf-8 -*-
# Copyright (C) 2019 The GRAND collaboration
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

"""Manage shared libs for the GRAND package
"""

from typing_extensions import Final

import os
from pathlib import Path

__all__ = ["LIBDIR", "DATADIR"]


# Initialise the package globals
DATADIR: Final = Path(__file__).parent / "data"
"""Path to the package data"""


LIBDIR: Final = Path(__file__).parent / "lib"
"""Path to the package shared libraries"""


SRCDIR: Final = Path(__file__).parent / "src"
"""Path to the source for C-extensions"""
