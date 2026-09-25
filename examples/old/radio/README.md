# Star-shape trace interpolation (archived, 2019)

`interpolation.py` estimates the electric-field trace at an arbitrary antenna
position from the traces of a star-shape simulation: it picks neighbouring
simulated antennas and interpolates between them in the shower frame.
**Nothing on `dev-next` does this.** Every other interpolation in GRANDlib is
along time or frequency.

Written by Anne Zilles (`azilles`) and packaged by Valentin Niess, December
2019, on the `radio` branch
(`lib/python/grand/radio/`, commits `ab307229`...`1567c068`). Copied here on
2026-09-24 when that branch was retired; the rest of it is either rewritten on
the trunk or unfinished stubs.

- `interpolation.py`: `interpolate_trace` (two traces, one position) and
  `do_interpolation` (a list of desired positions against a simulated array).
- `frame.py`: the shower (v×B) frame it works in, `get_rotation` and
  `UVWGetter`.
- `utils.py`: the refractive index `getn` it uses, and a few helpers.

## It does not run as it is

- It imports `load_trace` from the branch's `io_utils.py`, not kept: a
  five-line reader of whitespace-separated text traces named
  `a<index>.trace`.
- It reads the geomagnetic angles `phigeo` and `thetageo` from the branch's
  `config` module, also not kept.
- Traces in and out are text files, from before GRAND's ROOT format.

Porting it into `grand/` would mean reading traces through `grand.dataio`,
taking the field direction from `TShower.magnetic_field`, and checking it
against a simulated antenna left out of the star shape. That needs someone
who uses star-shape simulations to want it.
