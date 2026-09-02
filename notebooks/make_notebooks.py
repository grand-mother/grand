# -*- coding: utf-8 -*-
r"""Builds every tutorial notebook in this directory, executes it, and stores its outputs.

**The notebooks are generated.  Edit this file, not the ``.ipynb``** -- anything
written into a notebook by hand is lost the next time this runs.

Why generate them.  Seven notebooks share a title format, a navigation footer
and a set of conventions about units and frames that the reader is meant to
carry from one to the next.  Hand-maintaining those across seven JSON files
means they drift, and a diff of a hand-edited notebook is unreadable because
the outputs move with the source.  Generating them makes the shared structure
structural, and makes review a diff of Python.

Run ``python notebooks/make_notebooks.py`` to rebuild.  Execution is part of
the build, deliberately: a notebook that no longer runs against the current
package is a broken example, and the only way to know is to run it.  The build
also refuses to finish if a notebook came back without stored outputs, because
a notebook whose outputs were stripped renders blank on GitHub.

Options
-------
``--only 03,05``
    Rebuild and execute just those notebooks, matched as a prefix of the file
    name.  The staleness check still covers all of them, so this cannot be used
    to leave the directory half-rebuilt without being told.
``--no-execute``
    Write the notebooks without running them.  Useful while drafting; never
    what you want before committing, since the stored outputs are what a reader
    sees on GitHub.
``--check``
    Report whether the notebooks on disk match this file and carry outputs,
    writing nothing.  This is the one to use in a hook or in CI -- note that
    ``--no-execute`` is *not* a check, because it rewrites every notebook
    without its outputs.

Notes
-----
Some cells need data that is not in version control -- ``data/.gitignore``
excludes the SRTM elevation tiles and the ROOT test files.  Those cells are
written to detect the absence and say so rather than to fail, so that the
notebooks execute on a fresh checkout and in CI.  Notebook 07 is the main
example.
"""

import pathlib
import time

import nbformat as nbf

HERE = pathlib.Path(__file__).resolve().parent

#: Where the rendered documentation lives, for cross-links out of a notebook.
DOCS = 'https://grand-mother.github.io/grand-docs'


def md(text):
    r"""Returns a markdown cell holding `text`.

    Parameters
    ----------
    text : str
        Markdown source.  Trailing whitespace is stripped, so the generator
        can use indented triple-quoted strings without leaving blank lines in
        the rendered notebook.

    Returns
    -------
    nbformat.NotebookNode
        The cell.
    """
    return nbf.v4.new_markdown_cell(text.rstrip())


def code(text):
    r"""Returns a code cell holding `text`.

    Parameters
    ----------
    text : str
        Python source.

    Returns
    -------
    nbformat.NotebookNode
        The cell, with no outputs; :func:`build` fills those in by executing.
    """
    return nbf.v4.new_code_cell(text.rstrip())


def notebook(title, intro, cells):
    r"""Assembles one notebook from a title, an introduction and its cells.

    Parameters
    ----------
    title : str
        Rendered as the level-1 heading.  By convention it starts with the
        notebook's two-digit number, so the heading matches the file name.
    intro : str
        Markdown placed under the heading.  This is what a reader browsing the
        directory on GitHub sees first, so it should say what the notebook is
        for rather than what it will do.
    cells : list
        Cells from :func:`md` and :func:`code`, in order.

    Returns
    -------
    nbformat.NotebookNode
        A complete notebook, with the kernel metadata jupyter needs to execute
        it without being told which kernel to use.
    """
    nb = nbf.v4.new_notebook()
    nb.cells = [md('# %s\n\n%s' % (title, intro))] + cells
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'},
    }
    return nb


def footer(*links):
    r"""Returns the closing "Where next" cell.

    Parameters
    ----------
    *links : str
        Markdown list items, without the leading dash.

    Returns
    -------
    nbformat.NotebookNode
        A markdown cell.
    """
    return md('## Where next\n\n' + '\n'.join('- %s' % item for item in links))


def sources_match(path, nb):
    r"""Returns whether the notebook on disk has the sources this file builds.

    Compares sources only: outputs and execution counts differ after a run and
    are not part of what the generator owns.

    Parameters
    ----------
    path : pathlib.Path
        A notebook on disk.  A missing file counts as not matching.
    nb : nbformat.NotebookNode
        What this file builds for that name.

    Returns
    -------
    bool
        True if every cell type and source agrees, in order.
    """
    if not path.exists():
        return False
    on_disk = nbf.read(path, as_version=4)
    if len(on_disk.cells) != len(nb.cells):
        return False
    return all(a.cell_type == b.cell_type and a.source.rstrip() == b.source.rstrip()
               for a, b in zip(on_disk.cells, nb.cells))


books = {}

# --------------------------------------------------------------- 01_coordinates.ipynb
books['01_coordinates.ipynb'] = notebook(
    r'''01 — Coordinate systems''',
    r'''Getting a simulated electric field onto a real antenna is, more than anything
else, a coordinate problem. Air showers are computed in shower coordinates;
antennas sit at geodetic positions on curved, uneven terrain; the radio
emission is driven by the local geomagnetic field.

This notebook is the long form of the
[coordinates page](https://grand-mother.github.io/grand-docs/coordinates.html):
the frames, the conversions between them, the two conventions that most often
catch people out, and a worked layout at the GRANDProto300 site.

**Prerequisites**: a working `grand-dev` environment.

```bash
conda env create -f env/conda/grand-dev.yml --solver=libmamba
conda activate grand-dev
source env/setup.sh
jupyter lab notebooks/
```''',
    [
    code(r'''import numpy as np
import matplotlib.pyplot as plt

from grand import Geodetic, ECEF, LTP, GRANDCS
from grand.geo.coordinates import Reference, geoid_undulation

# The GRANDProto300 site at Dunhuang, China.
SITE = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)
print("site:", np.asarray(SITE).ravel())'''),
    md(r'''## 1. The four frames

| Frame | What it is |
|---|---|
| `Geodetic` | latitude, longitude, height — degrees and metres |
| `ECEF` | Earth-Centred Earth-Fixed Cartesian; the common pivot |
| `LTP` | a local tangent plane at a chosen origin and orientation |
| `GRANDCS` | the array frame: an `LTP` with GRAND's conventions |

Every conversion between a local frame and geodetic passes through `ECEF`.
There is no direct path, and that is deliberate: one pivot means one place for
the ellipsoid constants to live.'''),
    code(r'''ecef = ECEF(SITE)
print("ECEF (m):   ", np.round(np.asarray(ecef).ravel(), 1))
print("round trip: ", np.round(np.asarray(Geodetic(ecef)).ravel(), 6))'''),
    md(r'''## 2. The trap: `x` is not `x`

`GRANDCS` and `LTP` take the same three numbers and mean different things by
them. `LTP` with `orientation='ENU'` is east-north-up, so its `x` runs **east**.
`GRANDCS` follows the array convention, and its `x` runs **north**.'''),
    code(r'''a = ECEF(GRANDCS(x=1000.0, y=0.0, z=0.0, location=SITE))
b = ECEF(LTP(x=1000.0, y=0.0, z=0.0, location=SITE,
             orientation='ENU', magnetic=False))

separation = np.linalg.norm(np.asarray(a).ravel() - np.asarray(b).ravel())
print("same three numbers, different frame: %.1f m apart" % separation)'''),
    md(r'''No exception, no warning, and both answers are perfectly valid — they answer
different questions. A detector position confused this way lands outside the
array footprint; a shower axis confused this way points at the wrong patch of
sky.

The habit that prevents it is to name the frame in the variable rather than
relying on the constructor alone: `du_grandcs`, `axis_enu`.'''),
    code(r'''for name, orientation in [('ENU', 'ENU'), ('NWU (GRAND)', 'NWU')]:
    step = Geodetic(LTP(x=0.0, y=1000.0, z=0.0, location=SITE,
                        orientation=orientation, magnetic=False))
    lat, lon, _ = np.asarray(step).ravel()
    moved = 'north' if lat > 40.98 else ('west' if lon < 93.95 else 'east')
    print("1 km along y in %-12s -> %s" % (name, moved))'''),
    md(r'''## 3. Magnetic versus geographic north

`magnetic=True` measures the horizontal axes from magnetic north. The
declination at Dunhuang is a few degrees, which over a 10 km array is hundreds
of metres — a choice to make deliberately, not a default to inherit.

> **The date matters, and the shipped model has expired.** `data/geomagnet/IGRF13.COF`
> is IGRF-13, defined to 2025, so any `obstime` from 2025-01-01 onward raises
> `LibraryError: missing data`. The example below uses 2024 for that reason.
> See the known-issues page.'''),
    code(r'''geo = ECEF(LTP(x=1000.0, y=0.0, z=0.0, location=SITE,
               orientation='ENU', magnetic=False))
mag = ECEF(LTP(x=1000.0, y=0.0, z=0.0, location=SITE,
               orientation='ENU', magnetic=True, obstime='2024-06-01'))

print("geographic vs magnetic north, 1 km out: %.1f m apart"
      % np.linalg.norm(np.asarray(geo).ravel() - np.asarray(mag).ravel()))'''),
    md(r'''## 4. Heights need a reference

A height is meaningless without saying what it is measured from. The ellipsoid
is a smooth mathematical figure; the geoid is mean sea level. They differ by up
to about 100 m worldwide.'''),
    code(r'''undulation = geoid_undulation(latitude=40.98, longitude=93.95)
print("geoid - ellipsoid at Dunhuang: %.2f m" % undulation)
print("1200 m above the ellipsoid is %.2f m above sea level" % (1200.0 - undulation))
print("available references:", list(Reference))'''),
    md(r'''## 5. A detector layout, in two frames

A small hexagonal array in `GRANDCS`, then the same units as latitude and
longitude. This is the round trip every simulation performs, drawn.'''),
    code(r'''# a hexagonal layout, 1 km spacing, in the array frame
spacing = 1000.0
positions = [(0.0, 0.0)]
for ring in (1, 2):
    for k in range(6 * ring):
        angle = 2 * np.pi * k / (6 * ring)
        positions.append((ring * spacing * np.cos(angle),
                          ring * spacing * np.sin(angle)))
positions = np.array(positions)

geodetic = np.array([
    np.asarray(Geodetic(GRANDCS(x=float(x), y=float(y), z=0.0, location=SITE))).ravel()
    for x, y in positions])

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
left.scatter(positions[:, 1] / 1000, positions[:, 0] / 1000, s=28)
left.set_xlabel('y  [km]  (west)'); left.set_ylabel('x  [km]  (north)')
left.set_title('GRANDCS'); left.set_aspect('equal'); left.grid(alpha=.3)

right.scatter(geodetic[:, 1], geodetic[:, 0], s=28, color='C1')
right.set_xlabel('longitude [deg]'); right.set_ylabel('latitude [deg]')
right.set_title('Geodetic'); right.grid(alpha=.3)
fig.tight_layout()

print("array spans %.4f deg of latitude, %.4f deg of longitude"
      % (np.ptp(geodetic[:, 0]), np.ptp(geodetic[:, 1])))'''),
    md(r'''Note the aspect: a degree of longitude is shorter than a degree of latitude at
40°N, so the array that is circular in `GRANDCS` is elliptical when plotted
against raw degrees. That is not a bug in the conversion — it is why one works
in a local frame.'''),
    md(r'''## 6. A shower axis in both frames

A shower arriving at 85° zenith from the north, expressed as a direction in the
array frame and then as the geodetic points along its path.'''),
    code(r'''zenith, azimuth = np.radians(85.0), np.radians(0.0)
direction = np.array([np.sin(zenith) * np.cos(azimuth),
                      np.sin(zenith) * np.sin(azimuth),
                      np.cos(zenith)])
print("unit direction in GRANDCS (x north, y west, z up):", np.round(direction, 4))

along = np.array([
    np.asarray(Geodetic(GRANDCS(x=float(d * direction[0]),
                                y=float(d * direction[1]),
                                z=float(d * direction[2]), location=SITE))).ravel()
    for d in np.linspace(0, 20000, 6)])

for d, (lat, lon, h) in zip(np.linspace(0, 20, 6), along):
    print("  %5.1f km along the axis -> lat %.4f, lon %.4f, height %7.1f m"
          % (d, lat, lon, h))'''),
    md(r'''The height climbs as the axis rises, and the latitude increases because the
shower comes from the north — both consistent with `GRANDCS` `x` running north.'''),
    footer(
        r'''[02 — Reading and writing GRAND data](02_data_model.ipynb)''',
        r'''[03 — The antenna response](03_antenna_response.ipynb)''',
        r'''The [coordinates page](https://grand-mother.github.io/grand-docs/coordinates.html)''',
    ),
    ])

# --------------------------------------------------------------- 02_data_model.ipynb
books['02_data_model.ipynb'] = notebook(
    r'''02 — Reading and writing GRAND data''',
    r'''Everything GRAND records or simulates lives in ROOT `TTree`s, and
`grand.dataio` is the layer that reads and writes them. This notebook builds a
file from nothing, reads it back, and works through the conventions that govern
how files are grouped — which are not written down anywhere else and which I
got wrong three times while testing them.

The long form of the [data model page](https://grand-mother.github.io/grand-docs/datamodel.html).''',
    [
    code(r'''import tempfile, os
import numpy as np

from grand.dataio.run_trees import TRun
from grand.dataio.event_trees import TEfield, TVoltage
from grand.dataio.data_handling import DataDirectory, DataFile

workdir = tempfile.mkdtemp()
print("working in", workdir)'''),
    md(r'''## 1. The naming convention

The class names encode what a tree holds, and the convention is worth learning
before anything else:

| Pattern | Meaning |
|---|---|
| `TRun*` | constant for the duration of a run — one entry per run |
| `T*` | one entry per event |
| `T*Sim` | produced **only** by simulators |
| `T*` without `Sim` | hardware-origin; simulation may fill it, leaving unavailable fields empty |

So `TShower` is what a detector would have recorded, and `TShowerSim` is what
only a simulator knows.'''),
    code(r'''path = os.path.join(workdir, 'efield_20260101_000000_RUN0_L0_0000.root')

run = TRun(path)
run.run_number = 0
run.du_id = [0, 1, 2]
run.du_xyz = [[0., 0., 0.], [1000., 0., 0.], [0., 1000., 0.]]
run.t_bin_size = [0.5] * 3          # nanoseconds
run.origin_geoid = [40.98, 93.95, 1200.0]
run.analysis_level = 0
run.fill()
run.write()
print("wrote a run tree with", len(run.du_id), "detection units")'''),
    md(r'''## 2. Fields are descriptors, not plain attributes

A field on a tree class is declared like this:

```python
nutrig_rhox: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
```

`StdVectorListDesc` is a descriptor that translates between Python values and
the C++ type the branch holds. **Adding a field adds a branch to the on-disk
format**, which is why a field addition is a change to a data contract rather
than an implementation detail — and why `tests/dataio/test_schema_snapshot.py`
exists to make such changes visible in a diff.'''),
    code(r'''t = np.arange(256) * 0.5                    # ns
efield = TEfield(path)
for event in range(3):
    pulse = np.exp(-((t - (40.0 + 20.0 * event)) ** 2) / (2 * 4.0 ** 2))
    efield.run_number, efield.event_number = 0, event
    efield.du_id = [0, 1, 2]
    efield.du_nanoseconds = [0, 0, 0]
    efield.du_seconds = [0, 0, 0]
    efield.trace = np.stack([np.stack([pulse, .5 * pulse, .2 * pulse])] * 3).astype(np.float32)
    efield.analysis_level = 0
    efield.fill()
efield.write()

print("events written:", len(efield.get_list_of_events()))'''),
    md(r'''## 3. Reading it back'''),
    code(r'''handle = DataFile(path)
print("trees exposed as attributes:",
      [a for a in ('trun', 'tefield', 'tvoltage', 'tadc') if hasattr(handle, a)])

handle.tefield.get_entry(1)
trace = np.asarray(handle.tefield.trace)
print("event 1 trace shape (units, arms, samples):", trace.shape)
print("peak sample per unit:", [int(np.argmax(trace[i, 0])) for i in range(trace.shape[0])])'''),
    md(r'''## 4. How files are grouped — three traps

`DataDirectory` scans a directory and decides which files belong together. Its
rules are not documented anywhere, and each of them can bite.

**Grouping keys on tree type and analysis level, not on run number.** That is
correct for the layout the converters produce — `sim2root` writes one run per
directory — but it means pointing `DataDirectory` at a directory holding
several runs *silently merges them*.'''),
    code(r'''directory = DataDirectory(workdir)
print("files found  :", [os.path.basename(f) for f in directory.get_list_of_files()])
print("handles      :", len(directory.get_list_of_files_handles()))'''),
    md(r'''**The level in the filename must match the level inside the trees.** The
scanner takes the analysis level from the `_L0_`/`_L1_` marker in the name, then
looks for a tree attribute named for the level recorded *in the tree*. When
they disagree you get `AttributeError: 'DataFile' object has no attribute
'tefield_l1'` — naming something you never wrote, and saying nothing about the
real cause.'''),
    code(r'''bad = os.path.join(workdir, 'mismatch_20260101_000000_RUN0_L1_0000.root')
r = TRun(bad); r.run_number = 0; r.du_id = [0]; r.du_xyz = [[0., 0., 0.]]
r.t_bin_size = [0.5]; r.analysis_level = 0          # name says L1, tree says 0
r.fill(); r.write()

try:
    DataDirectory(workdir).get_list_of_files_handles()
except AttributeError as exc:
    print("AttributeError:", exc)'''),
    md(r'''**Both levels are returned, but the bare attribute follows the highest.** With
an L0 and an L1 file present, `directory.tefield` refers to L1 while
`tefield_l0` and `tefield_l1` name them individually. A script reading
`directory.tefield` therefore changes behaviour the moment someone drops an L1
file beside the L0 one.

## 5. Provenance

`TRun` carries `software_version`, `analysis_level`, `site` and `site_layout`.
This matters more than it sounds: a change to the Galactic-noise normalisation
alters every voltage in a file without changing its shape, and the version
stamp is the only way to tell two such files apart.'''),
    code(r'''handle.trun.get_entry(0)
for field_name in ('run_number', 'site', 'site_layout', 'analysis_level'):
    print("%-16s %r" % (field_name, getattr(handle.trun, field_name, None)))'''),
    footer(
        r'''[01 — Coordinate systems](01_coordinates.ipynb)''',
        r'''[03 — The antenna response](03_antenna_response.ipynb)''',
        r'''The [API reference](https://grand-mother.github.io/grand-docs/api.html)''',
    ),
    ])

# --------------------------------------------------- 03_antenna_response.ipynb
books['03_antenna_response.ipynb'] = notebook(
    r'''03 — The antenna response''',
    r'''The response of a GRAND antenna to an incoming radio wave is its **effective
length** $\boldsymbol{\ell}$: a complex, direction- and frequency-dependent
vector. The open-circuit voltage induced at one arm is its projection onto the
electric field,

$$V_{\rm oc}^{\,p}(\nu) = \boldsymbol{\ell}^{\,p}(\nu,\theta,\phi) \cdot \boldsymbol{E}(\nu),$$

with $p \in \{\rm SN, EW, Z\}$ running over the three arms of a HorizonAntenna.

This is the first stage of the simulation chain and the one that carries the
most instrument-specific knowledge: nothing outside GRAND knows what a
HorizonAntenna is. Everything downstream — the RF chain, the ADC — is generic
electronics; **this** is the physics of the detector.''',
    [
    code(r'''import numpy as np
import matplotlib.pyplot as plt

# AntennaModel loads the tabulated response of one detection unit.  Construction
# reads three data files from data/detector/, one per antenna arm, so it is not
# free -- build it once and reuse it, as we do throughout this notebook.
from grand.sim.detector.antenna_model import AntennaModel

model = AntennaModel()

# The three arms.  "sn" and "ew" are the horizontal arms, laid out south-north
# and east-west; "z" is the vertical one.
#
# These keys are what the simulation uses internally.  The data model calls the
# same three channels X, Y and Z, and the mapping is
#
#     X = sn (south-north)      Y = ew (east-west)      Z = z (vertical)
#
# which is also the order efield2voltage writes them in.  Note that the GRANDlib
# Handbook states the opposite (EW = X, SN = Y); the code is right and the
# Handbook is wrong -- see tests/sim/test_antenna_arm_identity.py, which
# identifies each file by correlating its pattern against the unambiguously
# named HFSS arms.  Following the Handbook swaps two of your three channels.
print("arms:", list(model.d_leff))'''),
    md(r'''## 0. A naming trap, before anything else

`d_leff` is keyed by the *physical* arm — `sn` south-north, `ew` east-west,
`z` vertical. The data model calls the same three channels X, Y, Z, and the
mapping is

| channel | arm | direction |
|---|---|---|
| X | `sn` | south–north |
| Y | `ew` | east–west |
| Z | `z` | vertical |

This is also the order `efield2voltage` writes them in, so `trace[:, 0]` is the
south-north arm.

The GRANDlib Handbook states the opposite — "EW or X arm, South–North denoted
as SN or Y arm". **The Handbook is wrong here.** The HFSS files are named after
the physical arm and carry no ambiguity, so the NEC and MATLAB files can be
identified by correlating their patterns against them; `nec_X` correlates +0.72
with the SN arm and −0.18 with EW. `tests/sim/test_antenna_arm_identity.py`
does that measurement and pins the result.

It matters because following the Handbook swaps two of the three channels for
every `GP300_nec` or `GP300_mat` simulation, and a channel swap looks like a
polarisation measurement rather than a bug.'''),
    md(r'''## 1. What is actually tabulated

Each arm is a `DataTable` holding the response on a regular grid. The two
complex arrays `leff_theta_reim` and `leff_phi_reim` are the $\theta$ and
$\phi$ components of $\boldsymbol{\ell}$ in the local spherical basis — the
antenna is described in its own frame, and it is `process_ant.py` that rotates
the shower direction into that frame before interpolating.

Note the trap in the last two lines below: `leff_theta` / `leff_phi`
(magnitude) and `phase_theta` / `phase_phi` exist as attributes but are `None`.
The tables ship in real/imaginary form and the polar form is never populated,
so code that reaches for `leff_theta` gets `None` rather than an error.'''),
    code(r'''t = model.leff_sn          # one arm's table; the other two have the same layout

# The grid the response is tabulated on.  Frequencies are stored in Hz, which is
# why they are divided by 1e6 for display here and converted throughout below;
# angles are in degrees, as everywhere in GRANDlib.
print("frequency  ", t.frequency.shape,
      "  %.0f - %.0f MHz" % (t.frequency.min() / 1e6, t.frequency.max() / 1e6))
print("phi        ", t.phi.shape, "      %.0f - %.0f deg" % (t.phi.min(), t.phi.max()))
print("theta      ", t.theta.shape, "       %.0f - %.0f deg" % (t.theta.min(), t.theta.max()))

# The response itself: one complex number per (frequency, azimuth, zenith).
print("leff_theta_reim", t.leff_theta_reim.shape, t.leff_theta_reim.dtype)

# ...and the attributes that look like they hold the same thing but do not.
print()
print("polar form populated?  leff_theta =", t.leff_theta,
      "  phase_theta =", t.phase_theta)'''),
    md(r'''## 2. Effective length against frequency

At a fixed arrival direction, $|\ell_\theta|$ and $|\ell_\phi|$ across the
30–250 MHz band. These curves decide which part of the shower spectrum the
antenna is sensitive to; combined with the Galactic background of notebook 05,
they set the band the experiment actually operates in.'''),
    code(r'''freq_mhz = t.frequency / 1e6

# Pick one arrival direction to hold fixed.  np.argmin(|axis - value|) is the
# idiom for "nearest tabulated point", used rather than interpolating because
# the grid is 1 degree and we only want a representative slice.
i_phi   = int(np.argmin(np.abs(t.phi - 45.0)))     # 45 deg azimuth
i_theta = int(np.argmin(np.abs(t.theta - 60.0)))   # 60 deg zenith: a typical
                                                   # inclined shower

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for name, arm in model.d_leff.items():
    # Index order is [frequency, azimuth, zenith], so this slices out a spectrum
    # at one direction.  np.abs() of the complex response is its magnitude in
    # metres; the phase is looked at separately in section 5.
    ax[0].plot(freq_mhz, np.abs(arm.leff_theta_reim[:, i_phi, i_theta]), label=name)
    ax[1].plot(freq_mhz, np.abs(arm.leff_phi_reim[:, i_phi, i_theta]),   label=name)

ax[0].set_ylabel(r'$|\ell_\theta|$ [m]')
ax[1].set_ylabel(r'$|\ell_\phi|$ [m]')
for a in ax:
    a.set_xlabel('frequency [MHz]')
    a.grid(alpha=.3)
    a.legend()
fig.suptitle(r'effective length at $\phi=45^\circ$, $\theta=60^\circ$')
fig.tight_layout()'''),
    md(r'''Two things to read off this figure.

First, the resonance near 100–150 MHz: that is where the arms are electrically
a useful fraction of a wavelength.

Second, **the Z arm is not a scaled copy of the horizontal arms**. It has a
different shape entirely, and it carries almost all of its response in
$\ell_\theta$ while the horizontal arms carry most of theirs in $\ell_\phi$.
That asymmetry propagates all the way to the noise budget and to which arm sees
a given shower — notebook 06 section 4 is the consequence.'''),
    md(r'''## 3. The directional map

Fixing the frequency instead and sweeping direction gives the beam pattern.
Figure 5 of [arXiv:2408.10926](https://arxiv.org/abs/2408.10926) shows the same
quantity.'''),
    code(r'''i_f = int(np.argmin(np.abs(freq_mhz - 150.0)))    # mid-band

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4),
                         subplot_kw=dict(projection='polar'))

# Polar axes want radians for the angular coordinate and take the radial
# coordinate as-is, so azimuth is converted and zenith is not.  indexing='ij'
# keeps the meshgrid in (phi, theta) order to match the table's axis order.
PHI, TH = np.meshgrid(np.radians(t.phi), t.theta, indexing='ij')

for ax, (name, arm) in zip(axes, model.d_leff.items()):
    amp = np.abs(arm.leff_theta_reim[i_f])        # (phi, theta) at this frequency
    pc = ax.pcolormesh(PHI, TH, amp, cmap='viridis', shading='auto')
    ax.set_title(r'%s,  $|\ell_\theta|$ at %.0f MHz' % (name, freq_mhz[i_f]), pad=14)
    ax.set_rlabel_position(135)                   # move the radial labels off the data
    fig.colorbar(pc, ax=ax, pad=.10, label='[m]')
fig.tight_layout()'''),
    md(r'''The radial coordinate is zenith angle, so the **centre of each disc is
straight up** and the rim is the horizon.

The horizontal arms show the two-lobed pattern you would expect, aligned with
the arm. The Z arm is azimuthally symmetric and dark at the centre: a vertical
dipole cannot respond to a wave arriving from directly overhead. The next cell
puts numbers on both statements rather than leaving them to the eye.'''),
    code(r'''# Verify the two claims above numerically rather than by looking at the figure.
print("at 150 MHz:")
for name, arm in model.d_leff.items():
    A = np.abs(arm.leff_theta_reim[i_f])          # (phi, theta)
    at_zenith  = A[:, 0].mean()                   # theta = 0 deg, straight up
    at_horizon = A[:, -1].mean()                  # theta = 90 deg
    # Coefficient of variation across azimuth: ~0 means azimuthally symmetric.
    spread = A[:, 60].std() / A[:, 60].mean()
    print("   %-4s |l_theta| at zenith %.4f m, at horizon %.4f m, "
          "azimuthal spread at 60 deg %.0f%%"
          % (name, at_zenith, at_horizon, 100 * spread))'''),
    md(r'''The Z arm reads 0.004 m at zenith against 1.25 m for the horizontal arms, and
its azimuthal spread is 2 % against 40 %. Both claims hold.

Every arm goes to exactly zero at the horizon, which is a property of the
tabulation rather than of the antenna: the model is not defined below the
horizontal.'''),
    md(r'''## 4. Comparing the arms quantitatively'''),
    code(r'''print("%-5s %10s %10s %10s" % ("arm", "peak", "median", "at MHz"))
for name, arm in model.d_leff.items():
    amp = np.abs(arm.leff_theta_reim)
    # unravel_index turns the flat argmax into (i_freq, i_phi, i_theta) so we can
    # report which frequency the peak sits at.
    i_pk = np.unravel_index(amp.argmax(), amp.shape)
    print("%-5s %9.2f m %8.3f m %9.0f"
          % (name, amp.max(), np.median(amp), freq_mhz[i_pk[0]]))'''),
    md(r'''The horizontal arms peak at 142 MHz and the Z arm at 49 MHz — a different
resonance, not a different scale. An analysis that assumes the three channels
share a band shape is assuming something false.'''),
    md(r'''## 5. Phase matters too

The effective length is complex, so the antenna does not merely scale the field
— it disperses it. Unwrapping the phase along frequency gives a group delay;
ignoring it would smear the sub-nanosecond timing that GRAND's reconstruction
depends on.'''),
    code(r'''z = t.leff_theta_reim[:, i_phi, i_theta]

# np.angle gives the phase in (-pi, pi], which wraps.  np.unwrap removes the
# 2-pi jumps so that the slope below is meaningful.
phase = np.unwrap(np.angle(z))

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(freq_mhz, np.degrees(phase))
ax.set_xlabel('frequency [MHz]')
ax.set_ylabel('phase [deg]')
ax.set_title(r'unwrapped phase of $\ell_\theta$, SN arm')
ax.grid(alpha=.3)
fig.tight_layout()

# A linear phase in frequency is a pure delay: phi = -2 pi nu tau, so the group
# delay is minus the slope over 2 pi.  Fit only over the part of the band where
# the response is large, since the phase is meaningless where the magnitude is
# near zero.
band = (freq_mhz > 50) & (freq_mhz < 200)
slope = np.polyfit(t.frequency[band], phase[band], 1)[0]   # rad per Hz
print("group delay over 50-200 MHz: %.2f ns" % (-slope / (2 * np.pi) * 1e9))'''),
    md(r'''## 6. A note for contributors

`AntennaModel.plot_effective_length()` exists but its body is `pass` — it is a
stub. The plots above are what it was presumably meant to produce. If you fill
it in, the figures in this notebook are a reasonable specification.'''),
    footer(
        r'''[02 — Reading and writing GRAND data](02_data_model.ipynb)''',
        r'''[04 — The RF chain](04_rf_chain.ipynb) — what happens to $V_{\rm oc}$ next''',
        r'''[05 — Galactic noise](05_galactic_noise.ipynb) — why the Z arm matters''',
    ),
    ])

# ---------------------------------------------------------- 04_rf_chain.ipynb
books['04_rf_chain.ipynb'] = notebook(
    r'''04 — The RF chain''',
    r'''Notebook 03 left us with an open-circuit voltage $V_{\rm oc}$ at the antenna
terminals. Between there and the digitiser sits a cascade of two-port networks:
a matching network, a low-noise amplifier, two baluns, a coaxial cable, and a
variable-gain amplifier with a band-pass filter.

Each stage ships as measured **S-parameters**. GRANDlib converts each to an
ABCD matrix, multiplies the matrices in order, and terminates the result in the
ADC load impedance. The product is one complex number per frequency and arm,

$$H(\nu) = \frac{V_{\rm out}(\nu)}{V_{\rm oc}(\nu)}$$

This notebook builds that cascade stage by stage — and ends by showing a bug
that a plot makes obvious and a code review does not.''',
    [
    code(r'''import numpy as np
import matplotlib.pyplot as plt

from grand.sim.detector.rf_chain import RFChain

# The GRAND band, in MHz and on a 1 MHz grid.  Everything in rf_chain.py takes
# and returns MHz -- unlike the antenna model of notebook 03, which stores Hz.
# Mixing the two up is the most common mistake in this module.
freqs_mhz = np.arange(30.0, 251.0)

# 20 dB is the GRANDProto300 default.  Section 5 shows that this argument is
# currently ignored, so every chain built here is a 20 dB chain regardless.
chain = RFChain(vga_gain=20)

# Construction only gathers the stages; this is what evaluates them.  It must be
# called before get_tf(), and it stores its results on the instance rather than
# returning them.
chain.compute_for_freqs(freqs_mhz)

tf = chain.get_tf()          # (3 arms, n_freq), complex
print("transfer function:", tf.shape, tf.dtype)'''),
    md(r'''## 1. The stages

`RFChain` is a facade. Each stage is a `GenericProcessingDU` subclass that
loads its own data file and exposes `s11`, `s21` and an `ABCD_matrix` of shape
`(2, 2, 3, n_freq)` — two by two for the network, then one per arm and
frequency.'''),
    code(r'''# The attribute names on RFChain, in the order the signal meets them
# physically.  Section 3 shows that this is *not* the order they are multiplied
# in, which is worth knowing before reading the source.
stages = ['matcnet', 'lna', 'balun1', 'cable', 'vgaf', 'balun2']

for name in stages:
    s = getattr(chain, name)
    print("%-9s %-22s ABCD %s" % (name, type(s).__name__, s.ABCD_matrix.shape))'''),
    md(r'''## 2. What each stage does on its own

$|S_{21}|$ in dB is the forward gain of a stage measured into a matched load.
Plotting them together shows the division of labour: the LNA supplies the gain,
the cable takes some back, and the VGA/filter stage defines the band edges.'''),
    code(r'''fig, ax = plt.subplots(figsize=(8, 4.2))
for name in stages:
    # s21 is stored per arm; arm 0 is representative and keeps the figure legible.
    s21 = getattr(chain, name).s21[0]
    # 20 log10 because S-parameters are amplitude ratios, not power ratios.  The
    # 1e-30 floor keeps log10 finite where a stage has an exact zero.
    ax.plot(freqs_mhz, 20 * np.log10(np.abs(s21) + 1e-30), label=name)

ax.axhline(0, color='k', lw=.6)       # unity gain, for reference
ax.set_xlabel('frequency [MHz]')
ax.set_ylabel(r'$|S_{21}|$ [dB]')
ax.set_title('forward gain of each stage, arm X')
ax.grid(alpha=.3)
ax.legend(ncol=2)
fig.tight_layout()'''),
    md(r'''## 3. Why ABCD and not just S

S-parameters are defined against a reference impedance and do not compose by
multiplication. ABCD (chain) matrices do:

$$\begin{pmatrix} A & B \\ C & D \end{pmatrix}_{\rm total}
 = \prod_{k} \begin{pmatrix} A & B \\ C & D \end{pmatrix}_{k}$$

so the cascade is a matrix product over stages. `rf_chain.s2abcd` does the
S-to-ABCD conversion and `rf_chain.matmul` the product. The antenna impedance
$Z_{\rm ant}$ enters at the input and the ADC load $Z_{\rm load}$ at the
output, which is why a stage's isolated $|S_{21}|$ above does **not** simply
add up to the total below.

The order the code actually uses is

```
balun1 . matcnet . lna . cable . vgaf . balun2
```

Worth flagging for anyone reading the source: the first factor is the class
named `BalunAfterLNA`, and it is applied *before* the matching network and the
LNA. Either the name or the order is misleading; which one has not been
established here. The cell below reproduces the product by hand, which is how
that order was determined.'''),
    code(r'''from grand.sim.detector.rf_chain import matmul

# The order compute_for_freqs() multiplies in -- not the physical order above.
order = ['balun1', 'matcnet', 'lna', 'cable', 'vgaf', 'balun2']

# matmul() here is rf_chain's own, which contracts the leading 2x2 and broadcasts
# over the trailing (arm, frequency) axes.  It is not np.matmul.
M = getattr(chain, order[0]).ABCD_matrix
for name in order[1:]:
    M = matmul(M, getattr(chain, name).ABCD_matrix)

# Index [0, 0, 0, 70] is the A element, arm X, at 30 + 70 = 100 MHz.
print("hand-cascaded  A(100 MHz), arm X:", M[0, 0, 0, 70])
print("chain.total_ABCD_matrix         :", chain.total_ABCD_matrix[0, 0, 0, 70])
print("agree:", np.allclose(M, chain.total_ABCD_matrix, rtol=1e-4))'''),
    md(r'''## 4. The total transfer function

This is Fig. 8 of [arXiv:2408.10926](https://arxiv.org/abs/2408.10926): the
factor by which the chain multiplies $V_{\rm oc}$.'''),
    code(r'''fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, arm in enumerate('XYZ'):
    ax[0].plot(freqs_mhz, np.abs(tf[i]), label=arm)
    # Unwrapped so the delay shows as a straight line rather than a sawtooth.
    ax[1].plot(freqs_mhz, np.degrees(np.unwrap(np.angle(tf[i]))), label=arm)

ax[0].set_ylabel(r'$|V_{\rm out}/V_{\rm oc}|$')
ax[1].set_ylabel('phase [deg]')
ax[0].set_title('RF chain gain')
ax[1].set_title('RF chain phase')
for a in ax:
    a.set_xlabel('frequency [MHz]')
    a.grid(alpha=.3)
    a.legend()
fig.tight_layout()

gain_db = 20 * np.log10(np.abs(tf))
print("peak gain per arm:", np.round(np.abs(tf).max(axis=1), 1))
print("in dB            :", np.round(gain_db.max(axis=1), 1))'''),
    md(r'''Two features worth noting. The band edges near 30 and 250 MHz come from the
filter, not from the antenna — the antenna of notebook 03 is tabulated over
exactly this range, so the two are matched by construction. And the phase is
smooth and steeply sloped: the chain delays the pulse by a fixed amount, which
the reconstruction has to know about.'''),
    md(r'''## 5. A bug you can see

`RFChain` takes a `vga_gain` argument. S-parameters are shipped for −5, 0, 5
and 20 dB, so four different chains should give four different curves separated
by 25 dB end to end.

They do not.'''),
    code(r'''settings = [-5, 0, 5, 20]
curves = {}
for g in settings:
    # A fresh chain per setting, so nothing is carried over between them.
    c = RFChain(vga_gain=g)
    c.compute_for_freqs(freqs_mhz)
    curves[g] = np.abs(c.get_tf()[0])       # arm X

fig, ax = plt.subplots(figsize=(7.5, 4))
for g, y in curves.items():
    # Thick and semi-transparent, so four distinct curves would be visible as
    # four bands.  They are not, because there is only one curve here.
    ax.plot(freqs_mhz, y, lw=2.5, alpha=.7, label='vga_gain = %+d dB' % g)

ax.set_xlabel('frequency [MHz]')
ax.set_ylabel(r'$|V_{\rm out}/V_{\rm oc}|$')
ax.set_title('four gain settings, one curve')
ax.grid(alpha=.3)
ax.legend()
fig.tight_layout()'''),
    code(r'''# The figure could be hiding four curves that happen to overlap on this scale,
# so check the difference directly rather than trusting the eye.
ref = curves[20]
for g, y in curves.items():
    print("vga_gain = %+3d dB : max relative difference from 20 dB = %.3g"
          % (g, np.max(np.abs(y - ref) / ref)))'''),
    md(r'''All four are bit-identical. The gain argument is accepted, stored, and never
used to select the data file — `VGAFilter._set_name_data_file` ignores it and
loads the 20 dB table regardless.

The consequence is not cosmetic. Any study that varies the VGA setting — a
dynamic-range scan, a saturation study, a comparison against a run taken at a
different gain — silently produces the 20 dB answer. That is worse than an
error, because the output looks plausible.

This is pinned by an expected-to-fail test in
`tests/sim/test_rf_chain_physics.py`, so the day the loader is fixed the suite
will say so:

```python
@pytest.mark.xfail(reason='vga_gain is currently ignored: ...')
def test_vga_gain_changes_the_transfer_function():
    low, high = RFChain(vga_gain=0), RFChain(vga_gain=20)
    ...
```

The fix is a one-line change in the file-name builder. It is left undone
deliberately: nobody has confirmed which of the shipped tables corresponds to
which hardware configuration, and guessing would replace a visible bug with an
invisible one.'''),
    footer(
        r'''[03 — The antenna response](03_antenna_response.ipynb) — where $V_{\rm oc}$ came from''',
        r'''[05 — Galactic noise](05_galactic_noise.ipynb) — what is added alongside it''',
        r'''[06 — From electric field to ADC counts](06_efield_to_adc.ipynb) — the whole chain at once''',
    ),
    ])

# --------------------------------------------------- 05_galactic_noise.ipynb
books['05_galactic_noise.ipynb'] = notebook(
    r'''05 — Galactic noise''',
    r'''Below about 100 MHz the sky is bright. The Galactic synchrotron background is
the dominant noise source for GRAND, larger than the receiver's own noise, and
it is not constant: as the Earth turns, the Galactic plane rises and sets, and
the noise level follows local sidereal time.

That matters for two reasons. The obvious one is the trigger threshold. The
subtler one is that the noise level is a *calibration signal* — it is the one
input whose amplitude is known independently of the instrument, so comparing
measured to predicted Galactic noise is how the absolute response of a
detection unit gets checked in the field.''',
    [
    code(r'''import numpy as np
import matplotlib.pyplot as plt

from grand import grand_add_path_data
from grand.sim.noise.galaxy import galactic_noise

# The tabulated model is defined on a 1 MHz grid from 30 to 250 MHz.  Asking for
# a different grid interpolates; asking outside the range extrapolates silently,
# so stay inside it.
freqs_mhz = np.arange(30.0, 251.0)

# size_out is the length of the padded time trace the spectrum will eventually be
# transformed back into.  It enters the amplitude normalisation, which is the
# subject of section 6.
SIZE_OUT = 1024

# f_lst is the local sidereal time in hours; seed makes the draw reproducible.
v = galactic_noise(f_lst=18.0, size_out=SIZE_OUT, freqs_mhz=freqs_mhz,
                   nb_ant=4, seed=0)
print("shape (n_ant, 3 arms, n_freq):", v.shape, v.dtype)'''),
    md(r'''## 1. The tabulated model

The underlying tables come from LFMap sky brightness temperatures folded
through the antenna response of notebook 03. They are stored per frequency, per
hour of LST, per arm.

The transpose in the next cell is not cosmetic: it is exactly what `galaxy.py`
does internally, and the resulting axis order is what every slice below
assumes.'''),
    code(r'''tab = np.load(grand_add_path_data("noise/Vocmax_30-250MHz_uVperMHz_nec.npy"))
print("as stored :", tab.shape, "  (frequency, LST hour, arm)")

tab = np.transpose(tab, (0, 2, 1))          # what galaxy.py does
print("as used   :", tab.shape, "  (frequency, arm, LST hour)")

# Index 17 is LST 18 h: the code indexes with lst - 1, so hour h is column h-1.
print("one LST slice:", tab[:, :, 17].shape, " -> 221 frequencies x 3 arms at LST 18 h")'''),
    md(r'''## 2. Noise against local sidereal time

This is the diurnal cycle. Each arm sees the Galactic plane differently
because each arm points differently — which is exactly the asymmetry notebook
03 measured.'''),
    code(r'''lst_hours = np.arange(1, 25)

fig, ax = plt.subplots(figsize=(7.5, 4))
for i, arm in enumerate('XYZ'):
    # Quadrature sum over frequency: the band-integrated amplitude, since power
    # adds and the table holds amplitudes.
    y = np.sqrt(np.sum(tab[:, i, :] ** 2, axis=0))
    ax.plot(lst_hours, y, marker='o', ms=3, label='%s arm' % arm)

ax.set_xlabel('local sidereal time [h]')
ax.set_ylabel(r'band-integrated $V_{\rm oc}$ [$\mu$V]')
ax.set_title('Galactic noise over one sidereal day')
ax.set_xticks(np.arange(0, 25, 3))
ax.grid(alpha=.3)
ax.legend()
fig.tight_layout()

for i, arm in enumerate('XYZ'):
    y = np.sqrt(np.sum(tab[:, i, :] ** 2, axis=0))
    print("%s arm: max at LST %2d h, min at LST %2d h, ratio %.2f"
          % (arm, lst_hours[y.argmax()], lst_hours[y.argmin()], y.max() / y.min()))'''),
    md(r'''The maxima land at different sidereal hours for different arms — LST 22 h for
X, 17 h for Y, 15 h for Z — because each arm weights the sky differently. The
Z arm is blind at zenith, as notebook 03 measured, so it peaks when the
Galactic plane is low. Fig. 7 of
[arXiv:2408.10926](https://arxiv.org/abs/2408.10926) shows the same behaviour.

The peak-to-trough swing is modest — 35 % to 45 % depending on the arm — but
the trigger rate depends on the threshold steeply enough that it is not a
detail.'''),
    md(r'''## 3. Spectral shape

The sky brightness temperature follows a steep power law, roughly
$T \propto \nu^{-2.5}$. What the antenna *sees* is much flatter than that,
because the effective length of notebook 03 rises with frequency across most
of the band and partly cancels the sky's fall.'''),
    code(r'''fig, ax = plt.subplots(figsize=(7.5, 4))
for hour in (6, 12, 18, 24):
    ax.loglog(np.arange(30., 251.), tab[:, 0, hour - 1], label='LST %d h' % hour)

ax.set_xlabel('frequency [MHz]')
ax.set_ylabel(r'$V_{\rm oc}$ [$\mu$V/MHz]')
ax.set_title('X arm spectrum at four sidereal times')
ax.grid(alpha=.3, which='both')
ax.legend()
fig.tight_layout()

# Quantify the "much flatter than nu^-2.5" claim: over a factor 8.3 in
# frequency, a nu^-2.5 sky would fall by a factor ~200.
lo, hi = tab[0, 0, 17], tab[-1, 0, 17]
print("LST 18 h, X arm: %.1f uV/MHz at 30 MHz -> %.2f uV/MHz at 250 MHz"
      % (lo, hi))
print("   falls by a factor %.1f, against ~%.0f for a bare nu^-2.5 sky"
      % (lo / hi, (250 / 30.) ** 2.5))'''),
    md(r'''## 4. One realisation

`galactic_noise` returns a *random draw*, not the table: a complex spectrum per
antenna and arm, with amplitude set by the table and phase randomised. Fixing
`seed` makes it reproducible.'''),
    code(r'''v = galactic_noise(18.0, SIZE_OUT, freqs_mhz, nb_ant=200, seed=1)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, arm in enumerate('XYZ'):
    ax[0].plot(freqs_mhz, np.abs(v[0, i]), lw=.8, label=arm)     # one antenna
    ax[1].plot(freqs_mhz, np.mean(np.abs(v[:, i]), axis=0), label=arm)  # averaged

ax[0].set_title('one antenna, one draw')
ax[1].set_title('mean over 200 antennas')
for a in ax:
    a.set_xlabel('frequency [MHz]')
    a.set_ylabel(r'$|V|$ [$\mu$V]')
    a.grid(alpha=.3)
    a.legend()
fig.tight_layout()'''),
    md(r'''Averaging over antennas recovers the smooth tabulated shape, as it must.
Neighbouring detection units get **independent** draws: spatial coherence of
the Galactic background is not modelled, on the argument that the array is
sparse compared with the coherence scale.'''),
    md(r'''## 5. Reproducibility, and one thing to watch

Two calls with the same seed agree exactly; without a seed they do not. The
tests depend on this.'''),
    code(r'''a = galactic_noise(18.0, SIZE_OUT, freqs_mhz, nb_ant=2, seed=42)
b = galactic_noise(18.0, SIZE_OUT, freqs_mhz, nb_ant=2, seed=42)
c = galactic_noise(18.0, SIZE_OUT, freqs_mhz, nb_ant=2)          # no seed
print("same seed identical :", np.array_equal(a, b))
print("no seed identical   :", np.array_equal(a, c))

# f_lst is truncated with int(), not interpolated, so the LST axis has 1 h
# resolution.  Same seed and same hour-bin therefore give the same draw.
d = galactic_noise(18.0, SIZE_OUT, freqs_mhz, nb_ant=2, seed=7)
e = galactic_noise(18.9, SIZE_OUT, freqs_mhz, nb_ant=2, seed=7)
print("LST 18.0 h and 18.9 h identical:", np.array_equal(d, e))'''),
    md(r'''The last line is a real limitation: `f_lst` is truncated with `int()`, so the
LST axis has 1-hour resolution and 18.9 h is silently treated as 18 h. For a
noise level that varies by 35–45 % over a day, hour-quantisation is a
few-percent effect — tolerable, but it should be interpolation, and the `TODO`
in the source says so.'''),
    md(r'''## 6. The normalisation, and how to check it yourself

The amplitude of the returned spectrum is scaled by a constant involving
`size_out`, and there is an open PR (#153) that changes it from `size_out/2`
to `size_out/sqrt(2)` — a factor of 1.41 on every simulated noise voltage, and
therefore on every trigger threshold downstream.

Parseval's theorem decides it, provided the comparison is made against the
table the simulation actually used. **That last clause matters**: the three
`du_type` values read different files, and those files differ in absolute
level by up to a factor of two, so comparing across them measures the model
difference rather than the normalisation.'''),
    code(r'''from scipy import fft

freqs = np.arange(30., 251.)
N_ANT, N = 600, 2048

for du, path in [('GP300_nec', "noise/Vocmax_30-250MHz_uVperMHz_nec.npy"),
                 ('GP300_mat', "noise/Vocmax_30-250MHz_uVperMHz_mat.npy")]:
    # The reference: what the table says the band RMS should be at LST 18 h.
    ref_tab = np.transpose(np.load(grand_add_path_data(path)), (0, 2, 1))
    ref = np.sqrt(np.sum(ref_tab[:, :, 17] ** 2, axis=0))

    # The simulation, transformed back to the time domain.  The 221 in-band bins
    # are placed at indices 30..250 of a length-N/2+1 one-sided spectrum, which
    # is what efield2voltage effectively does; irfft then gives a real trace.
    v = galactic_noise(18.0, N, freqs, nb_ant=N_ANT, seed=0, du_type=du)
    full = np.zeros((N_ANT, 3, N // 2 + 1), dtype=complex)
    full[:, :, 30:251] = v
    sim = np.std(fft.irfft(full, n=N, axis=-1), axis=-1).mean(axis=0)

    print("%-10s simulated/tabulated = %s   (1/sqrt2 = %.4f)"
          % (du, np.round(sim / ref, 4), 1 / np.sqrt(2)))'''),
    md(r'''The ratio is $1/\sqrt{2}$ — to about 0.3 %, for every arm, and (you can check)
independently of `size_out` and of the number of antennas. That is exactly the
relation between the RMS and the peak of a sinusoid.

So the code produces an RMS equal to the tabulated value divided by $\sqrt2$,
and the whole decision reduces to one definitional question:

> **Is `Vocmax_30-250MHz_uVperMHz` an RMS voltage spectral density, or a
> maximum?**

| if the table is a… | then the right constant is |
|---|---|
| **maximum** | `size_out/2` — the current code is right, and PR #153 would break it |
| **RMS** | `size_out/sqrt(2)` — PR #153 is right |

The filename says *max*. Nothing else in the repository states which is meant,
and the question belongs to whoever produced the tables — it is not something
the code can answer.

Until it is answered, quote the version of `grand.sim.noise.galaxy` that
produced any absolute noise level; `TRun.software_version` exists for that.
Relative statements — spectral shape, LST dependence, ratios between arms —
are unaffected, because the disputed factor is a single constant applied to
every model alike.

Pinned by `tests/sim/test_galactic_noise_normalisation.py` and written up in
`docs/dev/issues/galactic-noise-normalisation.md`.'''),
    md(r'''## 7. A trap: `du_type` changes the answer by a factor of two

The three antenna models are not three normalisations of one number. Two of
them are the *same file*, and the default reads neither.'''),
    code(r'''import hashlib

# If two selectors read byte-identical files they cannot be different models.
for kind in ('Vocmax_30-250MHz_uVperMHz', 'Pocmax_30-250_Watt_per_MHz'):
    h = {}
    for variant in ('hfss', 'nec', 'mat'):
        with open(grand_add_path_data("noise/%s_%s.npy" % (kind, variant)), 'rb') as fh:
            h[variant] = hashlib.sha256(fh.read()).hexdigest()[:12]
    print("%-30s nec=%s mat=%s  identical: %s"
          % (kind, h['nec'], h['mat'], h['nec'] == h['mat']))

print()
print("band-integrated V_oc at LST 18 h, per arm:")
for variant in ('hfss', 'nec', 'mat'):
    t = np.transpose(np.load(grand_add_path_data(
        "noise/Vocmax_30-250MHz_uVperMHz_%s.npy" % variant)), (0, 2, 1))
    print("   %-5s %s uV" % (variant,
                             np.round(np.sqrt(np.sum(t[:, :, 17] ** 2, axis=0)), 1)))'''),
    md(r'''`_nec` and `_mat` are byte-identical, so `du_type='GP300_nec'` and
`du_type='GP300_mat'` give the same numbers despite the docstring calling them
different antenna simulations. And the default, `du_type='GP300'`, reads
*neither* — it recomputes the voltage from `PG_ALL_jifen.mat` as
$V_{\rm oc}^2 = 4PR_{\rm ant}$, landing about 40 % below the `nec` tables and
a factor 2.1 below the `hfss` tables, which no `du_type` reaches at all.

**A Galactic noise level quoted without its `du_type` is ambiguous by a factor
of two.** This is separate from the $\sqrt2$ question above; see the
[known issues page](https://grand-mother.github.io/grand-docs/known_issues.html).'''),
    footer(
        r'''[03 — The antenna response](03_antenna_response.ipynb) — the effective length these tables were folded through''',
        r'''[04 — The RF chain](04_rf_chain.ipynb) — what the noise passes through next''',
        r'''[06 — From electric field to ADC counts](06_efield_to_adc.ipynb)''',
    ),
    ])

# --------------------------------------------------- 06_efield_to_adc.ipynb
books['06_efield_to_adc.ipynb'] = notebook(
    r'''06 — From electric field to ADC counts''',
    r'''Notebooks 03–05 covered the pieces. This one runs the whole thing:

```
E-field trace  ->  antenna (03)  ->  RF chain (04)  ->  + noise (05)  ->  ADC
```

`Efield2Voltage` drives it: it reads a ROOT file holding `TRun`, `TEfield` and
`TShower`, applies every stage, and writes a `TVoltage` tree back out.

We build the input here rather than downloading one, so the notebook runs on a
fresh checkout and the contents of the fixture are visible rather than opaque —
the same approach as `tests/sim/test_pipeline_end_to_end.py`.''',
    [
    code(r'''import itertools
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from grand import Efield2Voltage
from grand.dataio.event_trees import TEfield, TShower, TVoltage
from grand.dataio.run_trees import TRun

# Everything this notebook writes goes into a temporary directory that the last
# cell removes.  Nothing is left in the repository.
workdir = tempfile.mkdtemp(prefix='grand-nb06-')

# Each run of the chain needs its own output file: writing two events with the
# same (run_number, event_number) into one tree raises NotUniqueEvent.
counter = itertools.count()

print("working in", workdir)'''),
    md(r'''## 1. Building an input file

Three detection units, each carrying a Gaussian pulse. The three field
components get different amplitudes (1.0, 0.6, 0.2) on purpose: with equal
amplitudes a component swap somewhere in the chain would be invisible.

The peak field is **500 µV/m**, which is a strong but unremarkable shower.
That number matters — see section 3.'''),
    code(r'''N_DU, N_SAMPLES, T_BIN_NS = 3, 512, 0.5
PEAK_NS, WIDTH_NS = 120.0, 4.0
E_PEAK = 500.0                       # microvolts per metre
SITE = [40.98, 93.95, 1200.0]        # GRANDProto300: lat, lon, altitude

path = os.path.join(workdir, 'efield.root')

# --- TRun: everything constant across the events of one run ------------------
# Constructing a tree on a file that has no such tree prints "No valid trun
# TTree ... Creating a new one" -- that is the expected path when writing.
run = TRun(path)
run.run_number = 0
run.du_id = list(range(N_DU))
run.du_xyz = [[0., 0., 0.], [500., 0., 0.], [0., 500., 0.]]   # metres, site frame
run.t_bin_size = [T_BIN_NS] * N_DU                            # nanoseconds
run.origin_geoid = SITE
run.fill()          # commit this entry to the in-memory tree
run.write()         # flush the tree to the file

# --- TEfield: the per-event traces ------------------------------------------
t_ns = np.arange(N_SAMPLES) * T_BIN_NS
pulse = E_PEAK * np.exp(-((t_ns - PEAK_NS) ** 2) / (2 * WIDTH_NS ** 2))

# Shape (du, component, sample).  float32 because that is what the branch holds;
# passing float64 works but round-trips lossily and hides real differences.
trace = np.stack([np.stack([pulse * a for a in (1.0, 0.6, 0.2)])
                  for _ in range(N_DU)]).astype(np.float32)

efield = TEfield(path)
efield.run_number, efield.event_number = 0, 0
efield.du_id = list(range(N_DU))
efield.du_nanoseconds = [0] * N_DU
efield.du_seconds = [0] * N_DU
efield.trace = trace
efield.fill(); efield.write()

# --- TShower: the geometry the antenna response is evaluated for -------------
shower = TShower(path)
shower.run_number, shower.event_number = 0, 0
shower.zenith, shower.azimuth = 85.0, 0.0      # a very inclined shower
shower.energy_primary = 3.98e9                 # GeV
shower.shower_core_pos = [0., 0., 1200.]
shower.xmax_pos_shc = [0., 0., 10000.]         # shower-core frame, not the site frame
shower.fill(); shower.write()

print("wrote %s  (%.1f kB)" % (path, os.path.getsize(path) / 1e3))
print("E-field trace:", trace.shape, " (du, component, sample)")'''),
    md(r'''A note on units and conventions, which is where most first attempts go wrong:

- E-field traces are in **µV/m** (`event_trees.py` says so on the
  peak-to-peak fields).
- `t_bin_size` is in **nanoseconds**, so 0.5 ns is a 2 GHz sampling rate.
- `zenith` is measured from the vertical, so 85° is 5° above the horizon —
  the very inclined geometry GRAND is built for.
- `origin_geoid` is `[latitude, longitude, altitude]` in degrees and metres;
  everything angular in GRANDlib is in **degrees**, never radians.
- `xmax_pos_shc` is in the shower-core frame, not the site frame.'''),
    md(r'''## 2. Running the chain

`Efield2Voltage` takes an input path, an output path and a seed. The seed
makes the noise reproducible, which is what makes the stage-by-stage comparison
below meaningful.'''),
    code(r'''def run_chain(**params):
    """Runs the chain once on the fixture and returns the simulator.

    Any keyword given here overrides an entry of ``sim.params``; the defaults
    are printed below.  Each call writes to a fresh output file.
    """
    out = os.path.join(workdir, 'voltage_%d.root' % next(counter))
    sim = Efield2Voltage(path, out, seed=0)
    sim.params.update(params)
    sim.compute_voltage()
    sim.output_path = out          # remembered so section 7 can read it back
    return sim

full_sim = run_chain()
vout = np.asarray(full_sim.vout)   # (du, arm, sample), microvolts
print("output:", vout.shape, " peak |V| = %.3g uV" % np.abs(vout).max())
print()
print("params in effect:")
for k, v in sorted(full_sim.params.items()):
    print("   %-28s %s" % (k, v))'''),
    md(r'''## 3. Isolating each stage

Two flags switch stages off. Running the chain three times shows what each
contributes.

Because the seed is fixed, `full − chain` is *exactly* the noise that was
added. That gives an honest signal-to-noise number, rather than one estimated
from "the quiet part of the trace" — which does not exist here, because the
band-limited ringing spreads the pulse across the whole window.'''),
    code(r'''bare  = np.asarray(run_chain(add_noise=False, add_rf_chain=False).vout)  # V_oc only
chain = np.asarray(run_chain(add_noise=False).vout)                     # + RF chain
noise = vout - chain                                                    # exactly the noise

fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for a, (data, title) in zip(ax, [
        (bare,  r'antenna only:  $V_{\rm oc}$'),
        (chain, 'antenna + RF chain'),
        (vout,  'antenna + RF chain + Galactic noise')]):
    for i, arm in enumerate('XYZ'):
        a.plot(t_ns, data[0, i], lw=.9, label=arm)     # detection unit 0
    a.set_title(title, fontsize=10)
    a.grid(alpha=.3)
    a.legend(loc='upper right', fontsize=8)
    a.set_ylabel(r'V [$\mu$V]')
ax[-1].set_xlabel('time [ns]')
fig.tight_layout()

print("unit 0, per arm (X, Y, Z):")
print("   V_oc peak       ", np.round(np.abs(bare[0]).max(axis=-1), 1))
print("   + chain peak    ", np.round(np.abs(chain[0]).max(axis=-1), 1))
print("   noise RMS       ", np.round(np.std(noise[0], axis=-1), 1))
print("   signal / noise  ", np.round(np.abs(chain[0]).max(axis=-1)
                                      / np.std(noise[0], axis=-1), 1))'''),
    md(r'''Several things to read off this.

**The RF chain amplifies and rings.** The gain is the factor notebook 04
measured. The sharp input pulse becomes an oscillation because the chain is
band-limited to 30–250 MHz, and a Gaussian narrower than that band cannot
survive it. The ringing is real and appears in data; it is also why you cannot
estimate the noise from "the part of the trace away from the pulse".

**Amplitude decides everything.** At 500 µV/m the X arm has a signal-to-noise
near 10 and is comfortably detectable. Drop the input to 1 µV/m and
$V_{\rm oc}$ falls to 0.3 µV against a noise RMS of several hundred — a
signal-to-noise of 0.03, invisible. GRAND's sensitivity is set by exactly this
ratio, which is why the open $\sqrt2$ question of notebook 05 is not a detail.

**The Z arm output is tiny — and not for the reason you would guess.**'''),
    md(r'''## 4. A trap: arm X is not "the X component of E"

The input field has components in the ratio 1.0 : 0.6 : 0.2, but the output
arms come out closer to 600 : 400 : 1. The Z arm is not receiving 0.2 of the
signal; it is receiving almost none.

The reason is *not* that the Z arm is insensitive at this zenith angle. It is
the most sensitive of the three:'''),
    code(r'''from grand.sim.detector.antenna_model import AntennaModel

model = AntennaModel()
f_mhz = model.leff_sn.frequency / 1e6
i_f = int(np.argmin(np.abs(f_mhz - 100.0)))

# Index 85 of the theta axis is zenith 85 deg, matching shower.zenith above.
# Averaging over azimuth because the shower azimuth enters through the frame
# rotation rather than through this index.
print("mean effective length at 100 MHz, zenith 85 deg (the shower geometry):")
for name, arm in model.d_leff.items():
    lt = np.abs(arm.leff_theta_reim[i_f, :, 85]).mean()
    lp = np.abs(arm.leff_phi_reim[i_f, :, 85]).mean()
    print("   %-4s |l_theta| = %.3f m   |l_phi| = %.3f m" % (name, lt, lp))'''),
    md(r'''The Z arm has by far the largest $|\ell_\theta|$ at this zenith. What is small
is the *projection*: the response is
$V_{\rm oc} = \boldsymbol{\ell}\cdot\boldsymbol{E}$ evaluated in the
$(\theta, \phi)$ basis of the **arrival direction**, not a per-axis gain
applied to the Cartesian components of the field. The Z arm responds through
$\ell_\theta$, so what reaches it is the $\theta$-component of the field for
this particular direction — which for this geometry is small.

The practical rule: **`trace[:, 2]` is the Z antenna arm, not $E_z$.** Any
analysis that treats the three arms as a Cartesian decomposition of the field
will be wrong, and wrong in a direction-dependent way that will not look like
an obvious bug.'''),
    md(r'''## 5. In the frequency domain

The same three stages, seen spectrally, make the band limits obvious.'''),
    code(r'''# rfftfreq needs the sample spacing in seconds; T_BIN_NS is nanoseconds.
freqs = np.fft.rfftfreq(N_SAMPLES, d=T_BIN_NS * 1e-9) / 1e6   # MHz

fig, ax = plt.subplots(figsize=(8, 4))
for data, name in [(bare, r'$V_{\rm oc}$'), (chain, '+ RF chain'), (vout, '+ noise')]:
    ax.semilogy(freqs, np.abs(np.fft.rfft(data[0, 0])), lw=1, label=name)

ax.axvspan(30, 250, color='k', alpha=.06, label='30-250 MHz band')
ax.set_xlabel('frequency [MHz]')
ax.set_ylabel(r'$|V(\nu)|$')
ax.set_xlim(0, 500)          # Nyquist is 1000 MHz; nothing lives above ~300
ax.set_title('X arm, unit 0')
ax.grid(alpha=.3, which='both')
ax.legend()
fig.tight_layout()'''),
    md(r'''The RF chain's band-pass is unmistakable: outside 30–250 MHz the amplified
trace falls away sharply. The noise fills that band and nothing outside it,
because the tables of notebook 05 are only defined there.'''),
    md(r'''## 6. Digitising

The last stage turns microvolts into ADC counts.'''),
    code(r'''from grand.sim.detector.adc import ADC

adc = ADC()
print("ADC:", [a for a in dir(adc) if not a.startswith('_')])

# Read the constants from the class rather than restating them.  An earlier
# draft of this notebook hard-coded "1.8e6 / 2**14", which gives the right
# number for the wrong reason: the ADC is +/-0.9 V over +/-8192 counts, and
# 1.8/2**14 happens to equal 0.9/2**13.  Reading them means the notebook cannot
# quietly disagree with the code.
lsb_uv = adc.max_voltage / adc.max_bit_value
print()
print("full scale         +/-%.2f V over +/-%d counts" % (adc.max_voltage / 1e6, adc.max_bit_value))
print("quantisation step   %.1f uV" % lsb_uv)
print("noise RMS, X arm    %.1f uV  ->  %.1f counts"
      % (np.std(noise[0, 0]), np.std(noise[0, 0]) / lsb_uv))
print("signal peak, X arm  %.1f uV  ->  %.0f counts"
      % (np.abs(chain[0, 0]).max(), np.abs(chain[0, 0]).max() / lsb_uv))'''),
    md(r'''The noise sits several counts above the quantisation step, which is the
condition a well-designed digitiser has to meet: if the LSB were larger than
the noise, quantisation would dominate and information would be thrown away
before the trigger ever saw it.'''),
    md(r'''## 7. Reading the result back

The chain wrote a `TVoltage` tree into its output file. Round-tripping it is
the check that the simulation and the data model actually agree.'''),
    code(r'''voltage = TVoltage(full_sim.output_path)
voltage.get_entry(0)              # load event 0 into the branch buffers
back = np.asarray(voltage.trace)

print("read back:", back.shape, back.dtype)
# float32 on the way in, float64 on the way out, hence the tolerance.
print("matches what compute_voltage held:", np.allclose(back, vout, rtol=1e-5))
print("all finite:", np.isfinite(back).all())'''),
    md(r'''## 8. What this pipeline does and does not do

Worth being explicit, because it is the most common misunderstanding about
GRANDlib.

**It does**: apply a measured instrument response to a given electric field,
add a modelled background, digitise, and record the result in a schema the
rest of the collaboration can read.

**It does not**: simulate the air shower, or the radio emission from it. Those
come from ZHAireS or CoREAS and are converted into `TEfield` by the `sim2root`
tools. GRANDlib is the middleware between a shower simulator and an analysis —
which is why the schema and the frame handling matter as much as the physics.

Three limitations that apply to any number produced here:

- absolute noise levels are subject to the open $\sqrt2$ question, and depend
  on `du_type` by up to a factor of two (notebook 05);
- `vga_gain` is ignored, so this is a 20 dB chain regardless of what is
  requested (notebook 04);
- the arms are not Cartesian field components (section 4).'''),
    code(r'''import shutil

shutil.rmtree(workdir, ignore_errors=True)
print("cleaned up")'''),
    footer(
        r'''[02 — Reading and writing GRAND data](02_data_model.ipynb) — the schema this wrote into''',
        r'''[05 — Galactic noise](05_galactic_noise.ipynb) — the background added above''',
        r'''[07 — Topography](07_topography.ipynb) — where the detection units actually sit''',
    ),
    ])

# ------------------------------------------------------- 07_topography.ipynb
books['07_topography.ipynb'] = notebook(
    r'''07 — Topography''',
    r'''GRAND is built to see showers arriving within a few degrees of the horizon.
For those, the ground is not a flat plane a long way below the antennas — it is
in the way. A ray at 89° zenith would travel eighty kilometres to drop one
kilometre over flat ground, so whether it clears a ridge is decided by terrain
tens of kilometres from the array.

`grand.geo.topography` is the module for this. It wraps
[TURTLE](https://niess.github.io/turtle-pages/), which reads SRTM elevation
tiles and does ray–ground intersection on the real Earth, and it supplies the
geoid undulation needed to convert between the two height conventions GRAND
data uses.

This notebook works with whatever elevation data is on your machine and says
clearly what it could not do.''',
    [
    code(r'''import os

import numpy as np
import matplotlib.pyplot as plt

from grand.geo import topography
from grand.geo.coordinates import Geodetic, CartesianRepresentation

print("elevation model :", topography.model())
print("data directory  :", topography.datadir())

# SRTM tiles are named by the south-west corner of the one-degree square they
# cover, e.g. N41E096.hgt spans 41-42 N, 96-97 E.
datadir = str(topography.datadir())
tiles = sorted(f for f in os.listdir(datadir)
               if f.endswith('.hgt')) if os.path.isdir(datadir) else []
HAVE_TILES = len(tiles) > 0
print("tiles present   :", tiles if tiles else "none")'''),
    md(r'''The SRTM tiles are **not** in version control — `data/.gitignore` excludes
them, and they are several megabytes each. `topography.update_data()` downloads
what a given region needs. Everything below that requires a tile is guarded, so
this notebook runs either way; on a machine with no tiles it still covers the
geoid and the two failure modes, which is most of what there is to get wrong.'''),
    md(r'''## 1. Two definitions of "height"

A GRAND detection unit has an altitude, and that number means one of two
different things depending on who wrote it down.

- **Ellipsoidal height** is measured from the WGS-84 reference ellipsoid, a
  smooth mathematical surface. GPS reports this.
- **Height above mean sea level** is measured from the *geoid*, the surface of
  equal gravitational potential that sea level follows. Maps and altimeters
  report this.

The difference between them — the **geoid undulation** — reaches ±100 m
worldwide, far larger than the vertical precision GRAND needs for timing. The
EGM96 undulation map ships with the package, so this works with no downloads.'''),
    code(r'''# Pass a Geodetic rather than latitude=/longitude= keywords.  Section 2 shows
# why: the keyword form does not normalise the longitude and the Geodetic form
# does, so this is the one that works everywhere.
print("geoid undulation, in metres:")
for name, lat, lon in [('GRANDProto300 site (China)', 40.98,  93.95),
                       ('Auger site (Argentina)',    -35.20, -69.32),
                       ('Greenland',                  72.00, -40.00),
                       ('Indian Ocean low',            0.00,  78.00)]:
    u = topography.geoid_undulation(Geodetic(latitude=lat, longitude=lon, height=0.0))
    print("   %-28s %+8.2f" % (name, float(np.ravel(u)[0])))'''),
    md(r'''At the GRANDProto300 site the undulation is about −7.7 m: sea level sits 7.7 m
*below* the ellipsoid there. Confusing the two conventions moves every antenna
by that much, which at 2 GHz sampling is roughly 50 samples of light travel
time.

The spread across these four points — from −103 m to −1 m — is the reason this
cannot be treated as a constant offset for an array that spans any distance.

The rule in GRANDlib: `Geodetic.height` is **ellipsoidal**, and
`topography.elevation(..., reference='sea')` converts.'''),
    md(r'''## 2. Two silent failures worth knowing about

Neither of these raises. Both return `nan`, which then propagates quietly into
whatever geometry you were computing and stays plausible for several steps.

**First: the keyword form does not normalise longitude.** The shipped EGM96
map is indexed over 0–360°, and the `latitude=`/`longitude=` path passes the
value through unchanged, while the `Geodetic` path normalises it.'''),
    code(r'''lat, lon = -35.20, -69.32          # the Auger site, in the western hemisphere

kw   = topography.geoid_undulation(latitude=lat, longitude=lon)
wrap = topography.geoid_undulation(latitude=lat, longitude=lon + 360.0)
geo  = topography.geoid_undulation(Geodetic(latitude=lat, longitude=lon, height=0.0))

print("geoid_undulation(latitude=..., longitude=-69.32) :", kw)
print("geoid_undulation(latitude=..., longitude=290.68) :", wrap)
print("geoid_undulation(Geodetic(longitude=-69.32))     :", float(np.ravel(geo)[0]))'''),
    md(r'''Same point, three calls, two different answers and one `nan`. Prefer the
`Geodetic` form; if you must use the keywords, wrap the longitude into
$[0, 360)$ yourself.

**Second: a point with no elevation tile also returns `nan`.**'''),
    code(r'''far = Geodetic(latitude=40.98, longitude=93.95, height=0.0)   # GP300 site
print("elevation with no tile for this square:", topography.elevation(far))

# It propagates: the sea-level reference subtracts the undulation from a nan.
print("elevation(reference='sea')            :",
      topography.elevation(far, reference='sea'))'''),
    md(r'''**Check for `nan` after every elevation lookup**, or verify up front that the
tiles covering your region are present. This is the single most common way a
topography calculation goes wrong in practice, and because nothing raises, it
is usually found several steps later.'''),
    code(r'''if not HAVE_TILES:
    print("No SRTM tiles on this machine, so the rest of this notebook is")
    print("descriptive only.  To fetch what a region needs:")
    print()
    print("    from grand.geo import topography")
    print("    from grand.geo.coordinates import Geodetic")
    print("    topography.update_data(Geodetic(latitude=41.5, longitude=96.5,")
    print("                                    height=0.0), radius=50e3)")
else:
    # Work inside whichever square the available tile covers, parsed from its
    # name, so this notebook is not tied to one particular download.
    name = tiles[0]
    lat0 = float(name[1:3]) * (1 if name[0] == 'N' else -1)
    lon0 = float(name[4:7]) * (1 if name[3] == 'E' else -1)
    CENTRE = Geodetic(latitude=lat0 + 0.5, longitude=lon0 + 0.5, height=0.0)
    print("using tile %s -> working near lat %.2f, lon %.2f"
          % (name, lat0 + 0.5, lon0 + 0.5))
    print("elevation there: %.1f m" % topography.elevation(CENTRE))'''),
    md(r'''## 3. The terrain

An elevation map over the tile. This is the surface every ray has to clear.'''),
    code(r'''if HAVE_TILES:
    n = 120
    # Stay just inside the tile edges: a query exactly on the boundary may fall
    # into the neighbouring square, which is not downloaded.
    lats = np.linspace(lat0 + 0.05, lat0 + 0.95, n)
    lons = np.linspace(lon0 + 0.05, lon0 + 0.95, n)
    LON, LAT = np.meshgrid(lons, lats)

    # elevation() is vectorised over a Geodetic holding arrays, which is far
    # faster than looping: one TURTLE call instead of n^2.
    grid = topography.elevation(
        Geodetic(latitude=LAT.ravel(), longitude=LON.ravel(),
                 height=np.zeros(LAT.size))).reshape(LAT.shape)

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    pc = ax.pcolormesh(LON, LAT, grid, cmap='terrain', shading='auto')
    fig.colorbar(pc, ax=ax, label='elevation [m]')
    ax.set_xlabel('longitude [deg]')
    ax.set_ylabel('latitude [deg]')
    ax.set_title('SRTM elevation, tile %s' % name)
    fig.tight_layout()

    print("elevation over the tile: %.0f to %.0f m, median %.0f m"
          % (np.nanmin(grid), np.nanmax(grid), np.nanmedian(grid)))
    print("nan fraction: %.1f%%" % (100 * np.isnan(grid).mean()))
else:
    print("skipped: no tiles")'''),
    md(r'''## 4. Where does a very inclined ray meet the ground?

This is what `topography.distance` answers: given a starting point and a
direction, how far to the terrain? It marches along the ray and intersects the
actual elevation model, not a plane.

The direction is a Cartesian vector in the local ENU frame — x east, y north,
z up — so a zenith angle $\theta$ and azimuth $\phi$ become

$$\hat{d} = (\sin\theta\cos\phi,\; \sin\theta\sin\phi,\; -\cos\theta)$$

with the minus sign on $z$ because we are looking downward, towards the
ground.'''),
    code(r'''if HAVE_TILES:
    ground = topography.elevation(CENTRE)
    origin = Geodetic(latitude=lat0 + 0.5, longitude=lon0 + 0.5,
                      height=ground + 1500.0)     # 1.5 km above the terrain

    print("zenith    to ground [km]    flat-ground estimate [km]")
    for zen in (10.0, 45.0, 80.0, 85.0, 88.0, 89.0):
        th, ph = np.radians(zen), 0.0
        d = CartesianRepresentation(x=np.sin(th) * np.cos(ph),
                                    y=np.sin(th) * np.sin(ph),
                                    z=-np.cos(th))
        # maximum_distance bounds the march; without it a ray that never meets
        # the ground searches a long way before giving up.
        dist = float(np.ravel(topography.distance(origin, d,
                                                  maximum_distance=600e3))[0])
        # What a flat plane 1500 m below would have given, for comparison.
        flat = 1500.0 / np.cos(th)
        print("  %5.1f    %14s    %18.2f"
              % (zen, "%.2f" % (dist / 1e3) if np.isfinite(dist) else "not reached",
                 flat / 1e3))
else:
    print("skipped: no tiles")'''),
    md(r'''The two columns diverge, and they diverge in *both* directions.

At steep angles the real distance is **longer** than the flat-ground estimate —
2.25 km against 1.52 km at 10° zenith — because the ground falls away below the
launch point. The tile spans 1746 m to 2522 m and the centre sits at 2112 m, so
a ray heading almost straight down has further to fall than the local surface
suggests.

Near the horizon the sign flips and the size grows. At 89° the flat-ground
formula gives 86 km, while the real terrain stops the ray at 15 km: the ground
rises into it. **A flat-ground calculation overestimates the path by a factor
of nearly six there**, and puts the intersection point seventy kilometres
wrong — which for an array a few kilometres across is the difference between a
shower landing inside it and outside it.

Neither error is a small correction to be applied afterwards. Which way it goes
depends on the terrain and on the arrival direction, so there is no single
factor to fold in.'''),
    md(r'''## 5. A profile along a shower axis

Following the terrain along the ray gives the profile that decides whether a
shower is visible from a given antenna.'''),
    code(r'''if HAVE_TILES:
    zen, az = 87.0, 30.0
    th, ph = np.radians(zen), np.radians(az)
    east, north = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)

    s = np.linspace(0, 80e3, 300)          # along-track distance, metres

    # Small-angle conversion from metres to degrees.  Fine over 80 km; for
    # anything larger, or nearer the poles, use a proper geodesic.
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0 + 0.5))
    prof_lat = (lat0 + 0.5) + (s * north) / m_per_deg_lat
    prof_lon = (lon0 + 0.5) + (s * east) / m_per_deg_lon

    # Only query inside the tile; outside it every lookup would be nan anyway.
    inside = ((prof_lat > lat0) & (prof_lat < lat0 + 1)
              & (prof_lon > lon0) & (prof_lon < lon0 + 1))
    prof = np.full(s.shape, np.nan)
    prof[inside] = topography.elevation(
        Geodetic(latitude=prof_lat[inside], longitude=prof_lon[inside],
                 height=np.zeros(inside.sum())))

    # The ray, descending at cot(theta) metres per metre of ground track.
    ray = topography.elevation(CENTRE) + 1500.0 - s / np.tan(th)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.fill_between(s / 1e3, prof, color='#8a7a5c', alpha=.65, label='terrain')
    ax.plot(s / 1e3, ray, 'r--', lw=1.4, label='ray, zenith %.0f deg' % zen)
    ax.set_xlabel('distance along track [km]')
    ax.set_ylabel('elevation [m]')
    ax.set_title('terrain profile, azimuth %.0f deg' % az)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()

    hit = np.where(np.isfinite(prof) & (ray < prof))[0]
    print("ray meets the terrain at %.1f km" % (s[hit[0]] / 1e3) if hit.size
          else "ray stays above the terrain over the whole 80 km")
    print("profile leaves the tile after %.1f km" % (s[~inside][0] / 1e3)
          if (~inside).any() else "profile stays inside the tile")
else:
    print("skipped: no tiles")'''),
    md(r'''## 6. Where topography enters the rest of GRANDlib

Three places, and it is worth knowing which is which.

**Detection-unit altitudes.** `TRun.du_xyz` holds positions in the site frame;
turning them into real altitudes needs the terrain, and getting the geoid
convention wrong shifts every one of them together — a systematic that survives
averaging.

**Shower-axis geometry.** Where the axis meets the ground sets the core
position, and section 4 showed that for inclined showers a flat-ground estimate
gets that badly wrong.

**Antenna orientation.** Detection units are levelled to the local vertical,
which on sloping ground is not the same as the site vertical. The effective
length of notebook 03 is tabulated in the antenna frame, so a slope rotates the
response pattern.

Only the first is currently wired through the simulation chain. The other two
are available through this module but are the caller's responsibility.'''),
    footer(
        r'''[01 — Coordinate systems](01_coordinates.ipynb) — the frames the positions above live in''',
        r'''[06 — From electric field to ADC counts](06_efield_to_adc.ipynb) — the chain that uses them''',
        r'''The [API reference](%s/api.html) for `grand.geo.topography`''' % DOCS,
    ),
    ])


def check():
    r"""Reports whether the notebooks on disk match this file, writing nothing.

    ``--no-execute`` is not a check: it rewrites every notebook *without*
    outputs, which is exactly what must not happen to a committed notebook,
    since the stored outputs are what a reader sees on GitHub.  This compares
    instead.

    Returns
    -------
    int
        0 if every notebook matches and carries outputs, 1 otherwise.
    """
    stale, bare = [], []
    for name, nb in books.items():
        path = HERE / name
        if not sources_match(path, nb):
            stale.append(name)
            continue
        on_disk = nbf.read(path, as_version=4)
        if not any(cell.get('outputs') for cell in on_disk.cells
                   if cell.cell_type == 'code'):
            bare.append(name)

    if stale:
        print('  these no longer match the generator: %s' % ', '.join(stale))
        print('  run: python notebooks/make_notebooks.py')
    if bare:
        print('  these carry no stored outputs and would render blank on '
              'GitHub: %s' % ', '.join(bare))
    if not stale and not bare:
        print('  all %d notebooks match the generator and carry outputs'
              % len(books))
        return 0
    return 1


def build(execute=True, only=None):
    r"""Writes every notebook, executes it, and checks the directory is coherent.

    Parameters
    ----------
    execute : bool, optional
        Run each notebook and store its outputs.  True by default, because the
        stored outputs are what a reader sees on GitHub and an example that no
        longer runs is worse than no example.
    only : list of str or None, optional
        Rebuild just the notebooks whose file name starts with one of these.
        The staleness check below still covers all of them.

    Raises
    ------
    SystemExit
        If a notebook fails to execute, comes back with no stored outputs, or
        is left on disk not matching what this file builds.
    """
    selected = dict(books)
    if only:
        selected = {name: nb for name, nb in books.items()
                    if any(name.startswith(prefix) for prefix in only)}
        if not selected:
            raise SystemExit('--only matched no notebooks: %s' % ', '.join(only))

    for name, nb in selected.items():
        nbf.write(nb, HERE / name)
    print('  wrote %d notebook%s%s'
          % (len(selected), '' if len(selected) == 1 else 's',
             '' if not only else ' (of %d)' % len(books)))

    if not execute:
        return

    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    failed = []
    for path in sorted(HERE / name for name in selected):
        nb = nbf.read(path, as_version=4)
        started = time.perf_counter()
        try:
            # Twenty minutes is generous for these -- the slowest is 06, which
            # runs the whole simulation chain three times -- but a machine
            # building the C extensions for the first time is slower still.
            #
            # resources['metadata']['path'] makes the notebook's working
            # directory this one, so relative paths behave as they do when a
            # reader opens it in jupyter lab.
            NotebookClient(
                nb, timeout=1200, kernel_name='python3',
                resources={'metadata': {'path': str(path.parent)}}).execute()
            nbf.write(nb, path)
            print('  executed %-34s %6.1f s'
                  % (path.name, time.perf_counter() - started))
        except CellExecutionError as error:
            failed.append(path.name)
            print('  FAILED   %s\n%s' % (path.name, str(error)[:2000]))

    if failed:
        raise SystemExit('notebooks failed to execute: %s' % ', '.join(failed))

    # A notebook whose outputs were stripped renders blank on GitHub, so this is
    # checked here rather than discovered by a reader.
    bare = [path.name for path in sorted(HERE.glob('*.ipynb'))
            if not any(cell.get('outputs') for cell
                       in nbf.read(path, as_version=4).cells
                       if cell.cell_type == 'code')]
    if bare:
        raise SystemExit('notebooks carry no stored outputs: %s' % ', '.join(bare))

    # Every notebook on disk must still be the one this file builds.  With
    # --only the others were not rewritten, so this is what proves they were
    # already current rather than left over from an earlier edit.
    stale = [name for name, nb in books.items()
             if not sources_match(HERE / name, nb)]
    if stale:
        raise SystemExit('notebooks on disk no longer match this generator: %s\n'
                         'Rebuild them, or run without --only.' % ', '.join(stale))

    print('  %d executed; all %d notebooks match the generator and carry outputs'
          % (len(selected), len(books)))


if __name__ == '__main__':
    import sys

    argv = sys.argv[1:]
    if '--check' in argv:
        raise SystemExit(check())
    chosen = None
    if '--only' in argv:
        chosen = [f.strip() for f in argv[argv.index('--only') + 1].split(',')
                  if f.strip()]
    build(execute='--no-execute' not in argv, only=chosen)
