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

import ssl
import urllib.request
import zlib

def _disable_certs() -> None:
    """Disable certificates check"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

_disable_certs()


class InvalidBLOB(IOError):
    """Wrapper for store errors."""
    pass


def get(name: str, tag: str="101") -> bytes:
    """Get a BLOB from the store.
    """
    base = "https://github.com/grand-mother/store/releases/download"
    url = f"{base}/{tag}/{name}.gz"
    try:
        with urllib.request.urlopen(url) as f:
            return zlib.decompress(f.read(), wbits=31)
    except Exception as e:
        raise InvalidBLOB(e) from None
