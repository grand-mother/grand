import os
from pathlib import Path

from grand.geo import gull, turtle
from grand.geo.coordinates import Geodetic


def attempt(label, call):
    try:
        print(label, "->", call())
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


def unknown_model():
    try:
        gull.Snapshot("Unknown")
    except gull.LibraryError:
        return "caught by except gull.LibraryError"


site = Geodetic(latitude=45.0, longitude=3.0, height=0.0)
attempt("1. site.geodetic_to_horizontal()", lambda: site.geodetic_to_horizontal())
attempt("2. gull.Snapshot('Unknown')", unknown_model)
path = Path(os.environ["GRAND_ROOT"]) / "data" / "egm96.png"
attempt("3. turtle.Map(path).path == path", lambda: turtle.Map(path).path == path)
