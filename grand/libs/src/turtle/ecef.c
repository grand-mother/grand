/* Vectorization of the TURTLE/ECEF functions */

void turtle_ecef_from_geodetic_v(const double * latitude,
    const double * longitude, const double * altitude, double * ecef, long n)
{
        for (; n > 0; n--, latitude++, longitude++, altitude++, ecef += 3) {
                turtle_ecef_from_geodetic(
                    *latitude, *longitude, *altitude, ecef);
        }
}

void turtle_ecef_from_horizontal_v(const double * latitude,
    const double * longitude, const double * azimuth, const double * elevation,
    double * direction, long n)
{
        for (; n > 0; n--, latitude++, longitude++, azimuth++, elevation++,
            direction += 3) {
                turtle_ecef_from_horizontal(
                    *latitude, *longitude, *azimuth, *elevation, direction);
        }
}

void turtle_ecef_to_geodetic_v(const double * ecef, double * latitude,
    double * longitude, double * altitude, long n)
{
        for (; n > 0; n--, latitude++, longitude++, altitude++, ecef += 3) {
                turtle_ecef_to_geodetic(
                    ecef, latitude, longitude, altitude);
        }
}

void turtle_ecef_to_horizontal_v(const double * latitude,
    const double * longitude, const double * direction, double * azimuth,
    double * elevation, long n)
{
        for (; n > 0; n--, latitude++, longitude++, azimuth++, elevation++,
            direction += 3) {
                turtle_ecef_to_horizontal(
                    *latitude, *longitude, direction, azimuth, elevation);
        }
}
