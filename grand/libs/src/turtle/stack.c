/* Vectorization of the TURTLE/stack functions */

void turtle_stack_elevation_v(struct turtle_stack * stack,
    const double * latitude, const double * longitude, double * elevation,
    long n)
{
        for (; n > 0; n--, latitude++, longitude++, elevation++) {
                int inside;
                turtle_stack_elevation(
                    stack, *latitude, *longitude, elevation, &inside);
                if (!inside)
                        *elevation = NAN;
        }
}
