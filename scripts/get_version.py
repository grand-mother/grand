#!/usr/bin/env python3
r"""Prints the ROOT data-format version as ``version=<x.y.z>`` (used by CI).

The file is found from this script's location, so it works from any folder: it
read the relative path ``grand/dataio/version`` and printed ``0.0.0`` outside
the repository root (#246).
"""
from pathlib import Path

versionfile = Path(__file__).resolve().parents[1] / "grand" / "dataio" / "version"
version = versionfile.read_text() if versionfile.exists() else "0.0.0"
print("version=" + version)
