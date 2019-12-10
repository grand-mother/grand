# -*- coding: utf-8 -*-
"""
Unit tests for the radio_simus package
"""

from pathlib import Path
from grand.radio import load_config

path = Path(__file__).parent / "test.config"
load_config(path)
