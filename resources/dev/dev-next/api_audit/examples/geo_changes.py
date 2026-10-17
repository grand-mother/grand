import numpy as np
from grand.geo.coordinates import (ECEF, Geodetic, Horizontal, HorizontalRepresentation,
                                   SphericalRepresentation, geoid_undulation)


def attempt(label, call):
    try:
        print(label, "->", call())
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


site = Geodetic(latitude=40.0, longitude=90.0, height=0.0)
directions = Horizontal(azimuth=np.array([0.0, 90.0]), elevation=np.array([10.0, 20.0]),
                        norm=np.array([1.0, 1.0]), location=site)
attempt("1. Geodetic(latitude=100.0, longitude=90.0, height=0.0)", lambda: np.ravel(Geodetic(latitude=100.0, longitude=90.0, height=0.0).latitude).tolist())
attempt("2. HorizontalRepresentation(azimuth=0.0, elevation=100.0, norm=1.0)", lambda: HorizontalRepresentation(azimuth=0.0, elevation=100.0, norm=1.0).elevation.tolist())
attempt("3. SphericalRepresentation(theta=190.0, phi=0.0, r=1.0)", lambda: SphericalRepresentation(theta=190.0, phi=0.0, r=1.0).theta.tolist())
attempt("4. geoid_undulation(latitude=-35.2, longitude=-69.2)", lambda: round(float(geoid_undulation(latitude=-35.2, longitude=-69.2)), 2))
attempt("5. geoid_undulation(-35.2, longitude=290.8)", lambda: round(float(geoid_undulation(-35.2, longitude=290.8)), 2))
attempt("6. ECEF(site).ecef_to_grandcs(), no location", lambda: type(ECEF(site).ecef_to_grandcs()).__name__)
attempt("7. directions[:, :1].location", lambda: type(directions[:, :1].location).__name__)
