# -*- coding: utf-8 -*-
"""
Manage global parameters

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

import importlib

__all__ = ["Config", "load"]


class Config(dict):
    def __init__(self, name, *args, **kwargs):
        self._name = name
        for k in args:
            self[k] = None
        for k, v in kwargs.items():
            self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No such parameter: {self._name}.{key}")

    def __setattr__(self, key, value):
        self[key] = value

    def Config(self, name, *args, **kwargs):
        self[name] = Config(f"{self._name}.{name}", *args, **kwargs)


def load(path):
    spec = importlib.util.spec_from_file_location("config.load", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
