/* Vectorization of the snapshot field getter */
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
