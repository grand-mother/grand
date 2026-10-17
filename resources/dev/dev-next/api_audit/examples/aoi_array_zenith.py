import numpy as np
from grand import CartesianRepresentation
from grand.aoi import Shower


def attempt(label, call):
    try:
        print(label, "->", call())
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


def assign(value):
    shower = Shower()
    shower.zenith = value
    return repr(shower.zenith)


xmax = CartesianRepresentation(x=np.array([1000.0]), y=np.array([0.0]), z=np.array([5000.0]))
zenith = np.degrees(np.arccos(xmax.z / np.sqrt(xmax.x**2 + xmax.y**2 + xmax.z**2)))   # as in dummy_reconstruction.ipynb
attempt("1. zenith computed from Xmaxpos, type", lambda: type(zenith).__name__ + " " + str(np.shape(zenith)))
attempt("2. shower.zenith = that value", lambda: assign(zenith))
attempt("3. shower.zenith = np.array([11.3])", lambda: assign(np.array([11.3])))
attempt("4. shower.zenith = 11.3", lambda: assign(11.3))
