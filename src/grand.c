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


/* Transform a point from LTP coordinates to ECEF ones */
static void ltp_point_to_ecef(const double * origin, const double * basis,
    const double * local, double * ecef)
{
        int i;
        for (i = 0; i < 3; i++) {
                int j;
                ecef[i] = origin[i];
                for (j = 0; j < 3; j++)
                        ecef[i] += basis[3 * i + j] * local[j];
        }
}


/* Transform a point from ECEF coordinates to LTP ones */
static void ltp_point_from_ecef(const double * origin, const double * basis,
    const double * ecef, double * local)
{
        const double t[] = { ecef[0] - origin[0], ecef[1] - origin[1],
                             ecef[2] - origin[2] };

        int i;
        for (i = 0; i < 3; i++) {
                int j;
                local[i] = 0.;
                for (j = 0; j < 3; j++)
                        local[i] += basis[3 * j + i] * t[j];
        }
}


/* Ground elevation w.r.t. sea level or w.r.t. the ellipsoid */
void grand_topography_global_elevation(struct turtle_stack * stack,
    struct turtle_map * geoid, const double * latitude,
    const double * longitude, double * elevation, long n)
{
        for (; n > 0; n--, latitude++, longitude++, elevation++) {
                int inside;
                turtle_stack_elevation(
                    stack, *latitude, *longitude, elevation, &inside);
                if (!inside) {
                        *elevation = NAN;
                } else if (geoid != NULL) {
                        double undulation = 0.;
                        turtle_map_elevation(
                            geoid, *longitude, *latitude, &undulation, &inside);
                        *elevation += undulation;
                }
        }
}


/* Ground elevation in local coordinates */
void grand_topography_local_elevation(struct turtle_stack * stack,
    struct turtle_map * geoid, const double * origin, const double * basis,
    const double * x, const double * y, double * elevation, long n)
{
        for (; n > 0; n--, x++, y++, elevation++) {
                /* Let us compute the altitude in local coordinates using an
                 * iterative method. We need to find the ground position that
                 * projects at (x, y).  Let us Initialise the search on the
                 * ellipsoid, i.e. at zero altitude
                 */
                double local[3] = { *x, *y, 0. };
                double new[3] = { 0., 0., 0. };
                int _;
                for (_ = 0; _ < 5; _++) {
                        /* Get the ground altitude for the current guess */
                        double ecef[3], lla[3];
                        ltp_point_to_ecef(origin, basis, local, ecef);
                        turtle_ecef_to_geodetic(ecef, lla, lla + 1, lla + 2);

                        int inside;
                        turtle_stack_elevation(
                            stack, lla[0], lla[1], lla + 2, &inside);
                        if (!inside) {
                                new[2] = NAN;
                                break;
                        }

                        double undulation = 0.;
                        turtle_map_elevation(
                            geoid, lla[1], lla[0], &undulation, &inside);
                        lla[2] += undulation;

                        /* Compute the corresponding local coordinates */
                        turtle_ecef_from_geodetic(lla[0], lla[1], lla[2], ecef);
                        ltp_point_from_ecef(origin, basis, ecef, new);

                        /* Check for convergence and update. */
                        const double dx = *x - new[0];
                        const double dy = *y - new[1];
                        if ((fabs(dx) < 1E-03) && (fabs(dy) < 1E-03)) break;
                        local[0] += dx;
                        local[1] += dy;
                }
                *elevation = new[2];
        }
}


/* Intersection with the topography */
void grand_topography_distance(struct turtle_stepper * stepper,
    const double * r, const double * u, double * d, long n)
{   
        for (; n > 0; n--, r += 3, u += 3, d++) {
                int index[2];
                double altitude, elevation[2];
                turtle_stepper_step(stepper, (double *)r, NULL, NULL, NULL,
                    &altitude, elevation, NULL, index);
                if (*index >= 0)
                        *index = altitude > *elevation;

                int medium = *index;
                double dd = 0.;
                while ((*index == medium) && ((*d <= 0) || (dd < *d)) &&
                    (altitude > -11000) && (altitude < 8000)) {
                        double step;
                        turtle_stepper_step(stepper, (double *)r, u, NULL, NULL,
                        &altitude, elevation, &step, index);
                        dd += step;
                        if (*index >= 0)
                                *index = altitude > *elevation;
                }

                if ((*index >= 0) && (*index != medium) &&
                    ((*d <= 0) || (dd < *d))) {
                        *d = (medium == 0) ? -dd : dd;
                } else {
                        *d = NAN;
                }
        }
}
