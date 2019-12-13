# -*- coding: utf-8 -*-
"""
Add a brief description

Copyright (C) 2018 The GRAND collaboration

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

from .. import config as _config

__all__ = ["config"]


config = _config.Config(__name__)
config.Config("site", "name", "latitude", "longitude", "obsheight", "origin")
config.array = None
config.Config("magnet", "magnitude", "inclination", "declination")
config.Config("processing", "vrms1", "vrms2", "tsampling")
config.Config("antenna").Config("leff", "x", "y", "z")
