# -*- coding: utf-8 -*-
"""
Run all unit tests for the GRAND package
"""

try:
    from . import main
except ImportError:
    # This is a hack for a bug in `coverage` that does not support relative
    # imports from the __main__
    import os
    import sys

    path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(path)
    from tests import main


if __name__ == "__main__":
    main()
