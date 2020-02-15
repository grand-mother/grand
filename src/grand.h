#include "turtle.h"
#include "gull.h"


/* Vectorization of the TURTLE/ECEF functions */
void turtle_ecef_from_geodetic_v(const double * latitude,
    const double * longitude, const double * altitude, double * ecef, long n);

void turtle_ecef_from_horizontal_v(const double * latitude,
    const double * longitude, const double * azimuth, const double * elevation,
    double * direction, long n);

void turtle_ecef_to_geodetic_v(const double * ecef, double * latitude,
    double * longitude, double * altitude, long n);

void turtle_ecef_to_horizontal_v(const double * latitude,
    const double * longitude, const double * direction, double * azimuth,
    double * elevation, long n);


/* Getter for captured error messages */
const char * grand_error_get(void);


/* Vectorization of the TURTLE/map functions */
void turtle_map_elevation_v(struct turtle_map * map,
    const double * x, const double * y, double * elevation, long n);


/* Vectorization of the TURTLE/stack functions */
void turtle_stack_elevation_v(struct turtle_stack * stack,
    const double * latitude, const double * longitude, double * elevation,
    long n);


/* Vectorization of the GULL geomagnetic field snapshot */
enum gull_return gull_snapshot_field_v(struct gull_snapshot * snapshot,
    double * latitude, double * longitude, double * altitude, double * magnet,
    long n, double ** workspace);


/* Intersection with the topography */
void grand_topography_distance(struct turtle_stepper * stepper,
    const double * r, const double * u, double * d, long n);
