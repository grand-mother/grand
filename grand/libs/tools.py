# -*- coding: utf-8 -*-
# Copyright (C) 2018 The GRAND collaboration
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

"""Utilities for encapsulating shared libraries
"""

from typing import Generator, Optional, Union
from typing_extensions import Final

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from distutils.command.install import install
from pathlib import Path
from . import LIBDIR

__all__ = ["Meta", "Temporary", "define"]


def git(*args) -> str:
    """System git call
    """
    command = "git " + " ".join(args)
    return subprocess.getoutput(command)


@contextlib.contextmanager
def Temporary(url: str, tag: Optional[str]=None) -> Generator[str, None, None]:
    """Temporary context for building a shared library
    """
    path: Final = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="grand-") as tmpdir:
        try:
            # Clone the repo
            os.chdir(tmpdir)
            git(f"clone {url}")
            os.chdir(Path(url).name)
            if tag is not None:
                git(f"checkout {tag}")

            # Get the hash
            githash = git("rev-parse", "HEAD")

            # Yield back the context
            yield githash

        finally:
            os.chdir(path)


class Meta:
    """Encapsulation of library meta data
    """

    def __init__(self, name: str) -> None:
        path = LIBDIR / f".{name}.json"
        self._path: Final = path

        if path.exists():
            with path.open() as f:
                self._meta: dict = json.load(f)
        else:
            self._meta = {}

    def __getitem__(self, k: str) -> Optional[str]:
        try:
            return self._meta[k]
        except KeyError:
            return None

    def __setitem__(self, k:str, v: Optional[str]):
        self._meta[k] = v

    def update(self) -> None:
        if not LIBDIR.exists():
            LIBDIR.mkdir()

        with self._path.open("w") as f:
            json.dump(self._meta, f)


def define(source, arguments=None, result=None, exception=None):
    """Decorator for defining wrapped library functions
    """

    # Set the C prototype
    if arguments:
        source.argtypes = arguments
    if result:
        source.restype = result

    def decorator(function):
        # Return the (wrapped) function
        if exception is not None:
            def wrapped(*args):
                """Wrapper for library functions with error check"""
                r = source(*args)
                if r != 0:
                    raise exception(r)
                else:
                    return r

            return wrapped
        else:
            def wrapped(*args):
                """Wrapper for library functions without error check"""
                return source(*args)

            return wrapped

    return decorator
