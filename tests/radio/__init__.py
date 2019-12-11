# -*- coding: utf-8 -*-
"""
Unit tests for the radio_simus package
"""

from pathlib import Path
from grand.config import load

path = Path(__file__).parent / "config.py"
load(path)
