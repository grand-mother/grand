/* Vectorization of the TURTLE/map functions */

void turtle_map_elevation_v(struct turtle_map * map,
    const double * x, const double * y, double * elevation, long n)
{
        for (; n > 0; n--, x++, y++, elevation++) {
                int inside;
                turtle_map_elevation(map, *x, *y, elevation, &inside);
                if (!inside)
                        *elevation = NAN;
        }
}
