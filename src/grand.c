#include "grand.h"


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


/* Capture error messages */
static char * error_msg = NULL;

const char * grand_error_get(void)
{
        return error_msg;
}

static void capture_turtle_error(enum turtle_return code,
    turtle_function_t * function, const char * message)
{
        free(error_msg);
        const size_t n = strlen(message) + 1;
        error_msg = malloc(n);
        if (error_msg != NULL)
                memcpy(error_msg, message, n);
}

static void capture_gull_error(enum gull_return code,
    gull_function_t * function, const char * message)
{
        free(error_msg);
        const size_t n = strlen(message) + 1;
        error_msg = malloc(n);
        if (error_msg != NULL)
                memcpy(error_msg, message, n);
}

__attribute__((constructor)) void grand_init(void)
{
        turtle_error_handler_set(&capture_turtle_error);
        gull_error_handler_set(&capture_gull_error);
}

__attribute__((destructor)) void fini(void)
{
        free(error_msg);
}


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


/* Vectorization of the GULL geomagnetic field snapshot */
enum gull_return gull_snapshot_field_v(struct gull_snapshot * snapshot,
    double * latitude, double * longitude, double * altitude, double * magnet,
    long n, double ** workspace)
{
        for (; n > 0; n--, latitude++, longitude++, altitude++, magnet += 3) {
                enum gull_return rc = gull_snapshot_field(snapshot,
                    *latitude, *longitude, *altitude, magnet, workspace);
                if (rc != GULL_RETURN_SUCCESS)
                        return rc;
        }

        return GULL_RETURN_SUCCESS;
}


/* Inplace linear transform of Cartesian coordinates */
void grand_cartesian_transform_forward(double m[3][3], double t[3],
    double * x, double * y, double * z, long n)
{
        for (; n > 0; n--, x++, y++, z++) {
                double r0[3] = {*x, *y, *z}, r[3] = {t[0], t[1], t[2]};
                int i;
                for (i = 0; i < 3; i++) {
                        int j;
                        for (j = 0; j < 3; j++) r[i] += m[i][j] * r0[j];
                }
                *x = r[0], *y = r[1], *z = r[2];
        }
}

/* Inplace reverse linear transform of Cartesian coordinates */
void grand_cartesian_transform_backward(double m[3][3], double t[3],
    double * x, double * y, double * z, long n)
{
        for (; n > 0; n--, x++, y++, z++) {
                double r0[3] = {*x - t[0], *y - t[1], *z - t[2]},
                       r[3] = {0, 0, 0};
                int i;
                for (i = 0; i < 3; i++) {
                        int j;
                        for (j = 0; j < 3; j++) r[i] += m[j][i] * r0[j];
                }
                *x = r[0], *y = r[1], *z = r[2];
        }
}
