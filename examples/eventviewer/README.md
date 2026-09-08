# Event viewer

A view of one simulated event: the GP300 array, with the antennas that were hit
coloured by peak time; the electric-field trace and Hilbert envelope of one of
them; the shower parameters; and interpolated peak-amplitude maps in the ground
plane, the shower plane and the angular plane.

It runs as a small web application in your browser.

## Running it

The viewer needs a plotting stack that is **not** part of the normal GRANDlib
environment, so install it first:

```bash
pip install -e ".[viewer]"
```

Then point it at a run directory:

```bash
python examples/eventviewer/event_viewer_to_root.py \
    --datadir sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/
```

It prints a URL; open it. `--event N` chooses which event (default 0), and
`--port` moves it if 46813 is taken.

By default it serves on **localhost**, visible only to you. `--host 0.0.0.0`
publishes it to everyone who can reach your machine, which on a shared cluster
is everyone. That is occasionally what you want and never what you want by
accident.

## Where it came from

Originally [rameshkoirala/EventViewer](https://github.com/rameshkoirala/EventViewer)
(MIT, `rkoirala@nju.edu.cn`), which read HDF5. Claire Guepin rewrote it against
`grand.aoi` in May 2025 so that it reads GRAND ROOT files. The MIT header in
`event_viewer_to_root.py` is Koirala's and stays.

## What is unfinished

Stated plainly so nobody loses an afternoon discovering it:

- **Clicking an antenna does not change the trace.** This is the feature the
  tool most looks like it has. The trace shown is whichever antenna the
  `Selection1D` stream starts on — the middle one of the hit list — and it
  stays there. Verified by clicking, in a browser, on two different hit
  antennas: the title stayed on the same antenna both times. The `tap` tool is
  listed at `dmap_hits_plot.opts(...)`, so the intent is there and something
  further down does not connect; making `tap` the active tool does not fix it
  either. Diagnosing it properly was out of scope for the repair that made the
  viewer runnable.
- **The Play button does nothing.** `animate` is not wired to it and is marked
  in the source as needing an update. The **Browse** file input and the colour
  selector are inert for the same reason.
- **The background array is not your array.** `GP300propsedLayout.dat` is the
  *proposed* 2021 layout, 288 antennas on a 1 km grid. The antennas drawn as
  hit come from your data, but the grey array behind them does not, and the
  two need not agree — for the sample run the data spans about 8 km and the
  background 19 km. Reading the layout from the run's own tree is the obvious
  improvement and has not been done. (The misspelling in the filename is
  original; renaming it would change a default for no gain.)
- **The physics on display is unreviewed.** The interpolated amplitude maps
  and the shower-plane projection have not been checked by anyone during the
  repository overhaul. The test below proves the tool *runs*; it says nothing
  about whether the pictures are right.
- `mix.py` carries a band-pass filter that duplicates
  `grand.basis.signal.get_filter`. Left alone deliberately: consolidating it
  is a decision about what this tool is for.

## The test

`tests/examples/test_eventviewer.py` builds the whole interface against the
sample run committed to this repository and renders it, which is what forces
the interactive callbacks to execute. It skips unless the `viewer` extra is
installed, and CI installs it.

What the test does not do is click anything. It proves the interface builds
and draws; the interactive selection above is broken and a rendering test
cannot see that.

That test is the reason this file can promise the viewer starts and draws. It had stopped
being runnable by anyone but its author — a hard-coded event index of 862, a
data path on her machine, and a class that raised `NameError` if imported
rather than run — and none of that was visible, because nothing exercised it.
