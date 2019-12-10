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

import logging
logger = logging.getLogger("radio_simus")

import astropy.units as u
import numpy as np

__all__ = ["config"]


### Get and set config parameters
class Config:
    _config = {}

    def __getattr__(self, key):
        try:
            return self._config[key]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        if key in self._config:
            raise KeyError(f"{key} already defined")
        else:
            self._config[key] = value

config = Config()


#IMPORTANT: extend it with pa relative path
def load_config(path):
    global config

    path = str(path)
    configfile = open(path, 'r')
    print("..... Loading CONFIG FILE .....: "+path) # XXX use logger?
    for line in configfile:
        line = line.rstrip()
        if 'SITE' in line:
            config.site=str(line.split('  ',-1)[1])
        if 'LONG' in line:
            config.longitude=float(line.split('  ',-1)[1])*u.deg # deg ->astropy.units
        if 'LAT' in line:
            config.latitude=float(line.split('  ',-1)[1])*u.deg # deg ->astropy.units
        if 'OBSHEIGHT' in line:
            config.obs_height=float(line.split('  ',-1)[1])*u.m # m ->astropy.units
        if 'ORIGIN' in line:
            tmp = list(line.split('  ',-1))
            config.origin=np.array([float(tmp[1]),float(tmp[2]), float(tmp[3])])* u.m  # m ->astropy.units

        if 'ARRAY' in line:
            config.arrayfile=str(line.split('  ',-1)[1])

        if 'THETAGEO' in line:
            config.thetageo=float(line.split('  ',-1)[1])*u.deg # deg, GRAND ->astropy.units
        if 'PHIGEO' in line:
            config.phigeo=float(line.split('  ',-1)[1])*u.deg  # deg, GRAND ->astropy.units
        if 'B_COREAS' in line: # B_COREAS  19.71  -14.18
            tmp=list(line.split('  ',-1)) #  Bx  Bz  uT ->astropy.units
            config.Bcoreas=np.array([tmp[0], tmp[1]*u.u*u.T, tmp[2]*u.u*u.T])
        if 'B_ZHAIRES' in line: # B_COREAS  19.71  -14.18
            tmp=list(line.split('  ',-1)) # F in muT, I in deg, D in deg  ->astropy.units
            config.Bzhaires=np.array([tmp[0], tmp[1]*u.u*u.T, tmp[2]*u.deg, tmp[3]*u.deg])

        if 'VRMS1' in line:
            config.Vrms=float(line.split('  ',-1)[1])*u.u*u.V # muV  ->astropy.units
        if 'VRMS2' in line:
            config.Vrms2=float(line.split('  ',-1)[1])*u.u*u.V # muV  ->astropy.units
        if 'TSAMPLING' in line:
            config.tsampling=float(line.split('  ',-1)[1])*u.ns # ns  ->astropy.units

        if 'ANTX' in line:
            config.antx=str(line.split('  ',-1)[1])
        if 'ANTY' in line:
            config.anty=str(line.split('  ',-1)[1])
        if 'ANTZ' in line:
            config.antz=str(line.split('  ',-1)[1])
