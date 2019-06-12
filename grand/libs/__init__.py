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

import os

__all__ = ["LIBDIR", "DATADIR"]


# Initialise the package globals
DATADIR = os.path.join(os.path.dirname(__file__), "data")
"""Path to the package data"""


LIBDIR = os.path.join(os.path.dirname(__file__), "lib")
"""Path to the package shared libraries"""


SRCDIR = os.path.join(os.path.dirname(__file__), "src")
"""Path to the source for C-extensions"""
