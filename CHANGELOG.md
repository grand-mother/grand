# Changelog

All notable changes to **GRANDlib** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

Note that the **package** version tracked here is distinct from the **ROOT
data-format** version in `grand/dataio/version`, which has its own cycle: a
package release does not imply a schema change, or the reverse.

## [Unreleased]

Work on the `dev-next` integration branch, ahead of the first tagged release.

### Fixed

- **Argument messages in the conversion scripts (#233, items 1–2).**
  `convert_efield2efield.py` failed with a bare `AssertionError` for a
  negative noise, jitter or smearing, a negative rate or duration, and a
  `--target_duration_us` shorter than the traces; each now says what is
  wrong. `--padding_factor 0.5` was reported as "'extend_to_us' = 0 us is
  shorter than the traces"; `Efield2Voltage` now names `padding_factor`.
  The `--lst` message said "> 0h and < 24h" while accepting 0 and 24. (Item
  3, the T1 trigger passing no unit on clean simulations, is for the trigger
  group.)

- **`T1_trigger_offline.py` parses its arguments (#183).** It read
  `sys.argv[1]` directly, so `-h` was opened as a file and no argument gave
  `IndexError`. It now has a usage message, `-o` for where the list goes, and
  `--t1_param KEY=VALUE` as `convert_voltage2adc.py` has; the parser for those
  moved to `grand.sim.detector.trigger.t1_config_from_params`, so both scripts
  share it.

- **Class access to a tree field (#191).** `TShower.zenith`, and so
  `help(TShower.zenith)`, failed with `'NoneType' object has no attribute
  '_zenith'`: the field descriptors did not handle class access. They now
  return themselves, as Python descriptors do.

- **`Efield2Voltage` checks its configuration (#265).** A misspelt key in
  `params` (`add_noise_`) was ignored and the default used; the flags were read
  by truthiness, so the string `'no'` turned the RF chain on; a negative or
  NaN smearing sigma was accepted. Before any computation, unknown keys are
  now refused with the list of valid ones, the four flags must be `True` or
  `False`, `lst` must be 0–24 h and the numbers non-negative and finite.
  `padding_factor` is checked in the constructor, and a single e-field file
  (no run or shower tree) is refused with a message saying to give the
  sim2root folder, instead of `AttributeError: 'DataFile' object has no
  attribute 'trun'`. The defaults are public as `PARAM_DEFAULTS`.

- **A file whose name level disagrees with its trees (#187).** A `run_..._L1_`
  file holding level-0 trees made `DataDirectory` fail with an
  `AttributeError` naming a tree nobody wrote; a file not named after a tree
  type vanished from the directory without a word. The first now warns,
  naming the file and both levels, and uses the trees' level; the second is
  ignored with a warning. Notebook 02 shows the new behaviour.

- **A misspelt tree field is refused (#202).** `t.zenit = 5` was accepted and
  stored nowhere: the guard meant to catch it was assigned to each instance,
  where Python never looks for `__setattr__`. It is now on the class, active
  once the tree is built, and suggests the closest field ("did you mean
  'zenith'?"). Turning it on found the same mistake in the repository, all
  silently losing data:
  - `sim2root.py` wrote the gamma and hadron longitudinal profiles under
    names `TShowerSim` does not have (`long_pd_gammas`, `long_pd_hadr`); they
    now reach `long_pd_gamma` and `long_pd_hadron`. The CoREAS converter
    likewise wrote `long_pd_gamma` where `RawShowerTree` has `long_pd_gammas`.
  - `sim2root.py` also assigned 13 other profiles (all-charged, nuclei,
    neutrino, the low-energy and energy-deposit tables, their depth grid),
    `atmos_refractivity` and `du_x/y/z`, none of which any GRANDROOT tree
    has. They are still not written, now explicitly; whether to map the
    energy profiles onto `TShowerSim`'s `*_elow` / `*_edep` fields is a
    data-model decision left open.
  - `grand.aoi` wrote five monitoring fields to `TVoltage`, which has none;
    the ZHAireS converter set a slant depth with no field; a test set
    `trace_ch` on a `TVoltage`, whose field is `trace`.

- **Tree file patterns and lists (#205).** `TShower("dir/nomatch_*.root")`
  with no match created a file literally named `nomatch_*.root`; it now raises
  `FileNotFoundError`. A chain built from a pattern had no index, so
  `get_event` found nothing; it is now indexed, as `DataFile`'s chains are. A
  list of files is accepted, as `DataFile` accepts it; pattern matches are
  chained in sorted order; and `is_tchain` is true for a chain from
  `DataFile` too.

- **`DataDirectory`: recursive scan and oddly named files (#204).**
  `recursive=True` found nothing below the top folder, because the pattern
  had no `**`. A file of a known type whose name does not end in
  `_L<level>_<serial>.root` (`efield_copy.root`) aborted the whole folder
  with `ValueError: invalid literal for int()`; it is now skipped with a
  warning that names it.

- **CoREAS converter appended to an existing output (#181).** The trees open
  their file for appending, so converting a shower a second time -- or once,
  next to the committed sample -- failed with `NotUniqueEvent`. An existing
  output is now refused with a `GRANDlib:` message, or replaced with
  `--overwrite`; `-o` chooses the folder (or, for one shower, the file). The
  converter exits non-zero when given no option. The README and
  `sim2root.rst` say where the output goes.

- **Declination at Dunhuang overstated (#186).** The coordinates page and
  notebook 01 said "a few degrees ... hundreds of metres over 10 km". It is
  about 0.3° in 2020 (51 m at 10 km) and −0.03° in mid-2024 by the shipped
  IGRF-13 model. Both now give the real size; the notebook computes it. The
  move to IGRF-14 is a separate change.

- **`get_traces_lengths` and `get_list_of_dus` (#200).** `get_traces_lengths`
  looked for branches no tree has (`trace_x`, `trace_0`) and always returned
  `None`; it now gives, for the loaded entry, each unit's channel lengths,
  from `trace` or `trace_ch`. `get_list_of_dus` returned every unit in the
  tree, contrary to its docstring; it now gives the loaded entry's units, and
  `get_list_of_all_used_dus` the whole tree's.

- **Tree datetimes are UTC (#203).** `creation_datetime` was taken in UTC but
  stored as if it were local time, so it was off by the machine's UTC offset
  (8 hours early in China), and read back in local time again. Datetimes are
  now stored and read as UTC, naive datetimes are taken as UTC, and
  `get_metadata_as_dict` gives 1970-01-01 for an unset `source_datetime`, as
  the property does, instead of 0. The `utcnow()` deprecation warning on
  every tree creation is gone.

- **`copy_contents()` emptied the source (#282).** Copying a vector-of-vectors
  field from another tree moved the source's inner vectors instead of copying
  them, so the source's traces were left empty after the first copy, and a
  loop that copied from the same entry twice wrote empty traces. The copy is
  now element by element; the source is unchanged.

- **Tree lookups take NumPy integers (#276).** `get_entry(np.int64(1))`,
  the indices `np.where` gives, and `get_entry_with_index(t.run_number,
  t.event_number)` -- whose values are `np.uint32` -- raised `TypeError` from
  ROOT. Any integer is now accepted; a bool or a float is refused with a
  `GRANDlib:` message.

- **`sin_geomag_angle` on arrays (#214).** An array of angles gave a single
  number, larger than 1: the norm ran over all the directions together. It is
  now taken per direction; a scalar input still gives a float.

- **Output folders and the ADC script's input (#180, #182).** `-od` with a
  folder that did not exist yet failed with `FileNotFoundError` in
  `convert_efield2voltage.py` and `convert_efield2efield.py`, and so did
  `Efield2Voltage(output_directory=...)`; the folder is now created.
  `convert_voltage2adc.py` documented a file but needed the folder; its help
  now says folder, a voltage file in it is accepted too, and the "utput"
  typo is gone.

- **Effective length 1° off for every negative azimuth (#253).** The antenna
  tables hold azimuth 0–360° inclusive (361 points, 360 repeating 0), and the
  periodic wrap used the 361 points, so every direction with a negative
  azimuth read the table one step off: 0.8 % on average, a factor of 3 near a
  null. The wrap now uses the 360 steps of a turn. The pipeline reference
  (`tests/sim/pipeline_golden.npz`) is regenerated for this change: it moved
  by at most 5.8×10⁻⁵ of the trace peak.

- **Documented commands that failed (#257).** `sim2root.rst`,
  `simulation.rst`, `quickstart.rst` and the Handbook's Directory Structure
  page gave commands that failed as written: the CoREAS converter without
  `-d`, `sim2root.py` without `-sl`, and the voltage and ADC steps on single
  files. Each now gives the working form: `-d proton`, `-sl GP300` (with a
  common trace window for the two ZHAireS samples, #222), and the folder
  `sim2root.py` wrote for both conversion steps. A new test reads the commands
  out of the pages and runs them on the committed samples.
  `convert_voltage2adc.py` says what it needs instead of failing with
  `IndexError`, `sim2root.py -o` creates missing parent folders, the
  `Efield2Voltage` docstring describes its input correctly, and the Handbook
  errata list the failing commands.

- **Tests for deliberate bugs the suite missed (#270).** A mutation audit
  found five changes that passed every test. Each is now caught: the azimuth
  of `Horizontal` from an (east, north, up) position, and its round trip
  through ECEF; `final_resample`'s amplitude and output length at 2× and 0.5×
  the rate; exact ADC counts at ±0.5 and ±1.5 LSB and full scale. The order of
  `get_dus_indices_in_run` is pinned by the #199 test. Each new test was
  checked against its mutation.

- **Raw or reduced χ² in `TRecons`, and bounds that read 0 (#211).** The
  committed `recons_CR_candidates.root` holds the raw χ², as `TRecons`
  documents, but `main_AOI.py` and `main_DOI.py` wrote it divided by the
  degrees of freedom, and notebook 11 read the file as reduced: its section 8
  showed raw χ² under a "χ²/ndf" heading. The scripts now write the raw χ²;
  the field docstrings give the right degrees of freedom (`du_count - 2` or
  `- 4`, not `du_count`); notebook 11 divides, and its conclusions are
  rewritten from the new numbers (PWF χ²/ndf 0.1–12, not "tens to over a
  hundred"). Unfilled χ² and Cramér-Rao bound fields read NaN, not 0.0, which
  looked like a perfect fit or no uncertainty. `examples/analysis/README.md`
  records what the committed file holds.

- **`sim2root.py`: one trace window per run, and checked options (#222).**
  A run stores one `t_pre`/`t_post`; events with different windows were
  written under the first event's, with traces of different lengths and no
  warning. `sim2root.py` now checks every input before writing and stops,
  asking for `--trigger_time_ns` and `--target_duration_us` (or `-ss`). The
  window options are checked too: a duration of 0, a trigger at 0, or a
  trigger after the end of the trace wrote all-zero traces or failed with a
  bare error, and now stop with a `GRANDlib: sim2root:` message.

- **`topography.distance` read a local direction as ECEF (#210).** TURTLE
  needs an ECEF direction, the docstrings did not say so, and notebook 07
  passed an (east, north, up) vector: straight down from 1500 m came out as
  2.27 km. `distance` now takes `frame=` (an `LTP`, a `GRANDCS` or `"ENU"`)
  and rotates the direction to ECEF; the docstrings name the frame. Without
  `frame` the direction is still ECEF, as before. Notebook 07's section 4 is
  rebuilt: the terrain changes the inclined path by a few per cent at the
  shipped tile, not by "a factor of nearly six".

- **Coordinates west of Greenwich, and conversions that returned nothing (#251).**
  - `geoid_undulation(latitude=..., longitude=...)` returned NaN for a
    negative longitude: the EGM96 map is indexed 0–360° and only the
    `Geodetic` form wrapped the longitude. Both forms now wrap it and agree.
    The known-issues entry, troubleshooting and notebook 07 are updated.
  - `Horizontal` stored its location, basis and vector on the class, so a
    second `Horizontal` changed the first. They are now per instance.
  - `Geodetic.geodetic_to_horizontal` and the `*_to_grandcs` methods
    returned `None`. They now return the converted coordinates; called
    without a `location`, the GRAND-frame ones warn that the default origin
    is used.

- **Silent failures and global side effects (#256, in part).**
  - The Newton solver behind the Cherenkov angle returned whatever iterate it
    had reached when it did not converge (-610961.3 for x² + 1), without a
    word, and divided by zero when started at 0. It now warns and returns NaN,
    and handles a zero start.
  - `DataFile` raised a string when it could not index a chain, which is
    itself a `TypeError` that lost the message; and a read error was taken
    for an absent tree. Both now raise proper errors.
  - `EventList.get_event` and `Event.fill_event_from_trees` printed their
    errors and returned `None` or `False`, which callers and iteration passed
    on; they now raise `ValueError`, `LookupError` or `FileNotFoundError`.
  - `import grand` turned every `ComplexWarning` in the user's own code into
    an error; the filter now applies to `grand.geo.coordinates` only.
  - The `turtle.Map` cache loaded one file twice under two spellings and
    never freed its C maps; `create_output_for_logger` added handlers on
    every call (each message printed once per call), changed the caller's
    list and truncated its log file each time; `logger.exception` was used
    outside `except` blocks, logging "NoneType: None". All fixed.
  Still open in #256: the fallback chains in `descriptors.py`, and the
  remaining informational prints in `grand.aoi`.

- **RF-chain configuration errors are raised, not printed (#255).** A
  component missing from `rf_chain_config.xml` printed "ERROR: ..." and then
  failed with `NameError: name 'Nonec' is not defined`; an invalid axis
  printed an error and returned `None`, which callers passed on until an
  unrelated `TypeError`. `get_axis_filename` now raises `KeyError`,
  `ValueError` or `FileNotFoundError` naming the component and the axis. The
  duplicate definitions of `read_config` and `get_axis_filename` are gone, the
  configuration is read on first use rather than at import, the VGA gain check
  no longer relies on an `assert`, and a tautological `assert` is removed.

- **`get_files_from_db.py` reports what it does (#245).** Files of 256 kB or
  less that the transfer database lists were moved to a `crap/` folder with no
  message, listed files that did not exist were dropped silently, a database
  whose name did not match `<tag>_<site>_` gave `[]` and exit 0, and a
  mistyped path created an empty database before failing on "no such table"
  (also in `get_files_list.py`). The move is kept (a question for the pipeline
  owners) but every move and every missing file is reported on stderr; a
  misnamed, missing or unreadable database exits 2 with a message; the
  database is opened read-only, and the query takes its tag as a parameter.

- **A wheel carries the data files the code opens (#278).** `package-data`
  listed only `dataio/version`, so a wheel built where setuptools does not see
  the git checkout lacked `dataio/vector_filling.C` and
  `sim/detector/rf_chain_config.xml`: `import grand.dataio` printed a ROOT
  error and the first vector branch written failed with an unrelated
  `AttributeError`, and `grand.sim.efield2voltage` did not import. Both are
  now listed, `grand.dataio` refuses to import without its macro with a
  message saying why, and a test builds the wheel and uses it from outside
  the source tree.

- **The conversion scripts can be run again on the same folder (#240).**
  `convert_voltage2adc.py` deleted its earlier output, then failed with
  `NotUniqueEvent` and left none; `convert_efield2voltage.py` failed the same
  way on a re-run, although `-o`'s help says an existing file is overwritten;
  and a run that failed late, on a parameter checked only after the
  computation, left a stub file that broke every later run on the folder.
  Both scripts now write under a hidden temporary name and move the file into
  place only when it is complete, so the earlier output survives a failed run;
  `Efield2Voltage` checks its parameters before computing and leaves nothing
  behind when it fails; and `DataDirectory` skips a file holding no GRAND tree,
  with a warning. `Efield2Voltage` also records the analysis level of the
  e-field it read: a `voltage_*_L1_*` file said level 0, and the next scan of
  its folder failed.

- **`--seed` makes calibration smearing reproducible (#230).** The smearing
  drew from NumPy's global generator, which the seed never reached, so two runs
  with the same seed differed by up to 18583 µV; it now draws from a generator
  seeded with the seed and the event number. Jitter without a seed crashed
  (`None > 0`); seed 0 counted as no seed, in `Efield2Voltage` and
  `convert_efield2efield.py`; and negative seeds were accepted. Now 0 is a
  seed like any other, no seed means a fresh realisation, and a negative seed
  is refused.

- **`convert_efield2efield.py` writes `du_count`, honours `-o`, and handles
  empty events and inputs (#248).** `du_count` was never set, so every event
  of its output read as having no antenna; `-o` with a directory in it was
  cut to the file name and written into the input folder; an event whose
  shower hit no antenna crashed with `IndexError` (it is now written empty, as
  at every other level, #91); and an input with no events logged "Exiting."
  and went on to write empty output files (it now stops, writing nothing).

- **The sim2root pipeline example runs as documented (#221).** `RunSimPipe.py`
  and `RunSimPipeNoJitter.py` never passed the site layout that sim2root
  requires, so the first step failed and the next ones ran on whichever
  directory was newest (in `sim2root/Common`, a committed sample); the voltage
  step was also given its output path twice over (`DIR/DIR/voltage_...`).
  Both scripts now take a required `-sl`, pass it on, take the directory
  sim2root created rather than the newest one, and stop at the first failed
  step. The sim2root README examples gain `-sl` (and lose a stray `python`).
  `RunSimPipe.py ../ZHAireSRawRoot ZHAireS -sl GP300` was run end to end.

- **`recons_ADF` no longer returns its starting point as a fit (#286).** The
  loss divides by the antenna amplitudes, so a zero amplitude (an antenna
  below one ADC count) made it infinite everywhere and the fit returned the
  input angles and the initial width and amplitude, without a word; a source
  at or below the antennas made the minimizer run for minutes before doing
  the same. Non-positive amplitudes and a source not above the antennas are
  now refused with a `GRANDlib:` error naming the antennas; a fit whose loss
  ends non-finite raises; and a fit the minimizer reports as not converged
  logs a warning.

- **The README quickstart works (#185).** It gave `Efield2Voltage` and
  `convert_efield2voltage.py` a single e-field file, which fails: both need a
  simulation directory holding the run and shower trees as well. The
  quickstart now uses a directory, says what it must hold, and both commands
  were run as written on the committed RUN1 sample.

- **`Handling3dTraces` handles one-antenna events (#287).** `np.squeeze`
  dropped the antenna axis, so `get_tmax_vmax()` and `get_snr_and_noise()`
  crashed on an event with one antenna, and `interpol="no"` returned 0-d
  arrays. They now return one value per antenna for any number of antennas,
  and an unknown `interpol` raises a `ValueError` naming the accepted values
  (it raised "No active exception to reraise").

- **The ADC step no longer turns NaN or huge voltages into the most negative
  count (#239).** Casting a NaN, an inf or a voltage beyond the int64 range to
  an integer gave -9223372036854775808, whose absolute value is negative too,
  so saturation never clipped it: a large positive voltage came out as the
  most negative count, and the failure surfaced only later, as an int16 range
  error from the tree. `ADC.process` now refuses NaN and inf, naming the first
  (unit, channel, sample) index; bounds the value before the integer cast, so
  huge voltages saturate with the right sign; and logs how many samples
  saturated, per unit. `Efield2Voltage` refuses an e-field with NaN or inf
  samples, naming the event and the units.

- **`extract_events.py` no longer deletes the target directory (#244).**
  `-ow` removed the whole target directory with whatever else was in it (with
  target `.`, the current directory). It now replaces only the GRAND files the
  script writes, and refuses `.`, a parent of the current directory, or a
  directory holding a source. The event list is read and checked before the
  target is touched: blank and `#` lines are skipped, a repeated line is
  dropped with a warning, a bad line is reported with its number, and a path
  with a comma can be quoted. An event already in the target is skipped
  instead of aborting the job; a requested event that is not found makes the
  script exit 1; `-c` now stores its comment in the trees written. The trees
  of the two e-field levels share a name, so only one of them was written and
  the other file was left without a tree; both are written now.

- **The `open_grand_*` scripts no longer run the file name as code (#184).**
  `open_grand_file.py`, `open_grand_directory.py` and
  `open_grand_analysis_prompt.py` pasted the name into the Python command they
  start the shell with, so a quote in it broke the session and a crafted name
  ran arbitrary code. The name now reaches the shell through the environment.
  `open_grand_analysis_prompt.py` also printed `sys.argv[1]` as the directory,
  which is the first option when one is given.

- **Appending events slows down less (#283, in part).** Every time a tree was
  opened it listed all run and event numbers, reading the whole tree, and
  `Efield2Voltage` reopened and rewrote its output file for every event, so
  each event was slower than the last (about 16 ms per event at 400 events,
  40 ms at 2000). Opening a tree no longer lists its events: that is done when
  a `fill()` first needs it for the duplicate check. `compute_voltage()` keeps
  its output tree open and writes it once at the end. That also fixes
  `append_file=False`, which deleted the file before every event, so only the
  last event was kept, and crashed when the output directory was a string.
  Still open in #283: `DataDirectory` on thousands of files.

- **Two objects on one tree keep their own values (#273).** Two tree objects
  opened on one file wrapped the same ROOT tree, whose branches read into and
  fill from the buffers of whichever object bound them last: one object's
  `fill()` wrote the other's event number, and reading with one changed the
  other's fields. Each object now takes the branches back before it uses the
  tree, so it reads into and fills from its own fields.

- **Reading events no longer changes input files or crashes at exit (#234).**
  `Event.close_files()` wrote every tree, including those only read, so the
  input files grew a new key cycle each time (and the call could hang); it now
  writes only trees the event filled for writing (with the `Event.write`
  rewrite, #212). And a script that only read an event crashed or hung at exit
  in about a third of the runs (exit 129/139, in ROOT's `EndOfProcessCleanups`):
  ROOT deleted the trees after Python had begun freeing the buffers their
  branches point to. A tree object now detaches its buffers from the tree when
  it is released, and at exit, before ROOT's own cleanup, every tree ROOT still
  holds is detached; nothing is closed there, and no tree that may already be
  gone is touched. 30 of 30 runs exit cleanly.

- **Using a tree after its file was closed raises instead of crashing (#274).**
  `close_file()`, `DataFile.close()` and `DataDirectory.close()` delete the
  trees stored in the file, and any later use of one -- through the object
  that closed it or another object on the same file -- killed the interpreter
  (exit 129, no traceback). Every tree object stored in a file is now marked
  when the file closes, and `get_entry()`, `fill()`, `write()`, `draw()` and
  the other methods that read the tree raise a `RuntimeError` saying the file
  was closed. A tree kept in memory after `write("file.root")` is unaffected.

- **Filled entries are no longer dropped silently (#275).** Entries filled but
  not written were discarded without a word when a `with` block ended or
  `stop_using()` was called. Leaving a `with` block normally now writes them,
  as leaving `with open(...)` flushes a file; if the block ends with an
  exception, or `stop_using()` is called with entries pending, they are
  discarded with a `GRANDlibWarning` giving the count.

- **`EventList` and `DataFile` say what is wrong with an input (#235).**
  Plausible inputs failed deep inside with errors that named nothing useful.
  Now:
  - a shower file without a run tree raises `FileNotFoundError` saying the
    run tree is needed (it was `'NoneType' object has no attribute
    'origin_geoid'`);
  - a tree that is not a GRAND tree is skipped with a warning, and a file
    holding none raises `ValueError` (it was `attribute name must be string`);
  - `entry_number` must be an integer within the input (out of range it was
    `zero-size array to reduction operation minimum`; `True` was read as 1);
  - a closed `ROOT.TFile`, or an empty, text or damaged file, is refused with a
    `GRANDlib:` message, and `Event.file` accepts a path object;
  - data holding only raw voltages is read without `use_trawvoltage=True`;
  - tree types guessed from names no longer take `tvoltage` for
    `TRawVoltage` or `trunvoltage` for `TRun`.

- **`EventList` options take effect (#213).** `start_event` and `start_entry`
  were stored but ignored by iteration; they now set where it starts, and
  refuse values the input does not hold. A per-call `tefield_level` was
  ignored for directories (level 1 came back when level 0 was asked for) and
  stuck to later calls once used; it now applies to that call only, and a
  level the data does not hold raises `ValueError` instead of giving
  `efields=None`. `trawvoltage_channels` must name exactly three channels (it
  crashed with `IndexError` on fewer), and after a `use_trawvoltage=True` call
  the next default call no longer fails: the voltage tree is chosen on every
  call, and data holding only raw voltages is read as such. The docstrings
  now say that every event returned is the same, refilled `Event` object.

- **`Event.write` (#212).** `overwrite=True` with `out_dir` deleted the whole
  output directory, with anything else the user kept there; it now replaces
  only the files of the tree kinds it writes. `write()` with no destination
  crashed with `AttributeError`; it now says to give `out_dir` or file names.
  `common_filename` always failed with `TreeExists`, because `overwrite` was
  not passed on, and the failed write left the `Event`, and the `EventList` it
  came from, reading showers with zenith 0 and Xmax 0: writing now builds its
  own trees and never replaces the ones the event was read from. It adds the
  event to the file, or with `overwrite=True` replaces those trees; parts the
  event does not hold are skipped, and simulated events (no weather data, one
  sampling time) no longer crash the writer.

- **`get_dus_indices_in_run` follows the event's order (#199).** It returned
  the matching indices in the run's order, so whenever an event listed its
  units in another order than the run, antenna positions and sampling times
  (in `Efield2Voltage`, `convert_voltage2adc.py`, `convert_efield2efield.py`
  and `grand.aoi`) were paired with the wrong traces; units missing from the run
  were dropped silently. It now returns the indices in the event's order and
  raises for a unit the run does not hold. Every committed sim2root sample
  lists units in run order, so existing simulation outputs were not affected.

- **`write("other.root")` on a tree stored in a file writes a full copy
  (#198).** It moved the tree to the new file with `SetDirectory()`, which left
  the data already written behind: the new file referenced baskets it did not
  hold, and read back as zeros with zlib errors. The tree is now copied, every
  entry including those filled but not yet written, and the tree object stays
  attached to its own file.

- **`write()` no longer replaces a tree silently (#197).** Writing a new tree
  object into a file that already held a tree of the same name replaced it,
  with `overwrite=False` as with `overwrite=True`, so earlier events were lost;
  and `overwrite=True` opened the file with "recreate", which also deleted every
  other tree in it. Without `overwrite` this is now refused with a message
  saying how to add events (open the file with the tree class and fill that);
  with it, only the tree of that name is replaced.

- **Listing and drawing a tree no longer change the values it holds (#196).**
  `TTree::Draw()` reads every entry into the buffers the tree object is bound
  to, so after `get_list_of_events()`, `draw()`, `get_traces_lengths()` or the
  duplicate check in `fill()`, some fields held the last entry's values and
  others the loaded entry's (`event_number` 3 with entry 1's zenith).
  Worse, values just set for the next `fill()` were replaced too, so the
  wrong event could be written. The fields a draw touches are now saved
  before it and restored after it.

- **`DataDirectory` keeps every file of a level (#195).** Files were grouped
  by a fixed field of their name, so names of different lengths
  (`shower_<date>_<time>_0-0_L1_0000.root` and
  `shower_<date>_<time>_<run>_0-0_L1_0000.root`) of the same level fell into
  two groups, and one replaced the other: `examples/analysis/reconstructed_events_AOI`
  showed 1 of its 10 showers. The level is now read from the `L<n>` field
  before the serial number, wherever it is.

- **Several processes writing one ROOT file no longer lose events or
  corrupt it (#281).** Two batch jobs with the same output, or a resubmitted
  job, interleaved their writes: with three processes appending to one file,
  40 events were reported written and 21 were in the file, or the file was left
  unreadable, while some processes reported success. A process that writes a
  file now holds an exclusive lock on it (`flock` on the file itself, in the new
  `grand.dataio.file_lock`) from the moment it opens it for writing (`fill()`,
  `write()`, or creating the file) until it closes it. Another process that
  tries to write meanwhile, or to open the half-written file, is refused with
  a clear message; one that opened the file before someone else wrote it is
  refused when it tries to write, since its view is out of date. Everything
  reported as written is in the file (checked over 10 runs of 4 writers).
  Reading takes no lasting lock, and one process writing a file is unchanged.
  Where `flock` is unavailable the old behaviour remains. A process waits up to
  2 s for a lock before refusing: processes started together briefly met each
  other's read lock and could all give up.

- **No antenna response from below the antenna's horizon (#285).** The
  effective-length lookup took the zenith row modulo the table size, so a
  source at 91° read the 0° (zenith) row, 95° the 4° row, and so on, at full
  strength: an Xmax below an antenna's plane (near-horizontal showers over
  relief, or a wrong Xmax) gave 777–2500 µV instead of about zero, while 90–91°
  gave exactly zero. Directions outside the table (zenith above 90° for the
  GP300 tables) now get zero response, continuous with the 90° row, and a
  warning naming the angle. Directions inside the table are unchanged.

- **Notebook 05's sky maps show right ascension correctly (#268).** The shipped
  LFMap grid's first axis runs 12 h ahead of right ascension, so the maps put
  the Galactic Centre at 5.8 h instead of 17.8 h. The notebook now shifts the
  maps by 12 h, says why, and marks the Galactic Centre, Cygnus A and
  Cassiopeia A, which land on their bright pixels. The data and
  `galactic_noise` are unchanged.
- **`Handling3dTraces.init_traces` accepts a NumPy sampling rate (#264).** A
  rate given as a NumPy scalar (as read from a tree) was kept as a scalar, and
  `apply_bandpass` then failed with `IndexError: invalid index to scalar
  variable`. One rate for all units or one per unit is now accepted in any
  numeric form; a wrong length, NaN or a bool is refused.
- **A NaN position no longer crashes Python (#262).** libturtle's elevation
  lookups segfaulted on a NaN latitude or longitude, killing the interpreter with
  no traceback: `geoid_undulation`, `topography.elevation` (every reference) and
  `turtle.Map`/`Stack.elevation`. Only finite points now reach the C code; a NaN
  point (or `None`, which has always read as NaN) gives a NaN elevation with a
  `GRANDlibWarning`.
- **`convert_voltage_to_ADC` converts only the channels asked for (#263).** A
  regression from #179: a boolean mask such as `[True, False, True]` was iterated
  as indices 1, 0, 1, converting the wrong rows, and a slice or a single index
  raised `TypeError`. Channels are now selected as NumPy indexing does, as in
  `get_peak_amplitude`, and a mask of the wrong length is refused.
- **The EGM96 geoid map is no longer upside down (#250).** `data/egm96.png`
  stored its rows south-first, while TURTLE reads PNG rows north-first, so every
  geoid undulation was the value at the opposite latitude: the North Pole read
  -29.5 m instead of +13.6 m, and the GP300 site -7.75 m instead of -61.0 m. The
  rows are flipped, and the grid extents corrected (x1 = 360, y1 = 90 for a
  0.25 degree grid; they were one step too far, stretching the grid by up to a
  quarter of a degree). The poles and the model's global minimum and maximum
  now match published EGM96. **Heights given with the default
  `reference="GEOID"` (`Geodetic`, `ECEF`, `topography.elevation(...,
  reference="sea")`) change by the difference, about 53 m at the GP300 site;
  results computed with earlier versions from such heights need recomputing.**
- **The ZHAireS converter refuses a damaged simulation (#242).** A missing trace
  file was dropped, a truncated trace zero-padded, a NaN written as data, a
  missing zenith line read as a vertical shower, an unknown energy unit ignored,
  and a broken `.EventParameters` gave a NaN core, all with exit 0. Before
  writing anything the converter now checks that the trace files match the
  `.sry` antenna list, that antenna names and du_ids are unique and positions
  finite, that each trace is a finite (N, 4) table as long as the `.sry` time
  window (within one bin), and that zenith, azimuth, energy and core are
  present and valid. A failure exits non-zero naming the file and the problem,
  and removes the output file if this run created it. A shower that hit no
  antenna (no antenna in the `.sry`, no trace file) is still written (#91). A
  missing `.EventParameters` still gives core (0, 0, 0), now with a warning.
- **The CoREAS converter checks the antenna list against the trace files (#243).**
  `du_count` came from the trace files and everything else from the list, so a
  truncated list, a missing or extra trace, a NaN or a short trace produced a
  `.rawroot` that looked valid, exit 0. Before writing anything the converter now
  checks that both name the same antennas (no duplicates), that positions and
  traces are finite, traces have 4 columns and equal lengths, and
  `TimeResolution` is positive; otherwise it stops listing every problem.
- **An unknown Xmax no longer gives a crash or near-zero voltages (#228).** The
  CoREAS converter took `DistanceOfShowerMaximum = -1` ("unknown") as a distance
  of -1 cm, putting Xmax at the core, and `convert_efield2voltage` then produced
  V_oc around 1e-13 uV and all-zero ADC traces, exit 0; a NaN Xmax (the `.inp`
  path) crashed in the antenna lookup. The converter now writes NaN for an
  unknown distance or depth, and `Efield2Voltage` refuses an event whose Xmax is
  not finite or lies within 100 m of the core, with a message naming the event.
- **The CoREAS converter no longer mirrors the azimuth (#209).** For a `.reas`
  without the shower block (as the committed sample), the converter reads the
  CORSIKA `.inp` and computed `180 - PHIP`; PHIP is the direction of travel in a
  North/West frame, so the "comes from" azimuth is `PHIP - 180` (13.57° instead
  of -13.57° for the sample, confirmed by a plane-wave fit to its antenna
  timing). Both branches now wrap to [0, 360). The 2024 backward-compatibility
  fixture `sim_Dunhuang_*_CoREAS-NJ_0000` keeps its old, mirrored value on
  purpose; files converted from such inputs should be regenerated.
- **sim2root writes the right antenna latitude/longitude (`du_geoid`) (#220).**
  With several events sharing antennas in one `.rawroot`, `du_geoid` was
  computed from the non-unique antenna list and so paired with the wrong
  antennas (up to 0.1°, about 8 km off). With `-ss` it was assigned to a
  misspelt field and silently dropped, `du_tilt`/`t_bin_size` took the
  cumulative antenna count, and every run kept the first run's first event.
- **Resampling in the voltage step no longer gives the ADC step a wrong rate
  (#229).** `Efield2Voltage` kept the input's `trigger_position` after
  resampling (the ratio compared the input rate with itself), and left the run
  tree's `t_bin_size` at the input rate, so `convert_voltage2adc.py` processed a
  resampled trace as if it were at the input rate (at 750 MHz: 12288 samples
  written as 500 MHz ADC data, exit 0). `convert_efield2voltage.py
  --target_sampling_rate_mhz` is refused, since the voltage file cannot record
  the new rate; resample the e-field with `convert_efield2efield.py`, which
  writes the rate to its run file. The Python API did the same until wave 3
  found it (at 250 MHz the ADC traces came out all zeros):
  `Efield2Voltage.save_voltage` now refuses a voltage resampled with
  `params["resample_to_mhz"]`, before writing anything. Resampling in memory
  (`compute_voltage_event` and `final_resample`) still works, and a value equal
  to the input rate is not a resampling.
- **`Efield2Voltage` reads the run tree at the efield's level (#237).**
  `DataDirectory` picks the highest level of each tree type on its own, so a
  folder with an L0 efield file and an L1 run file (as `convert_efield2efield
  -od` leaves behind, #231) paired 0.5 ns traces with a 2 ns sampling time and
  doubled every voltage, exit 0. The run tree is now taken at the efield's
  level (an error if there is none), the shower tree at that level or the
  closest below, and the files read are logged.
- **The voltage and e-field conversions refuse an input whose shower or run
  tree lacks the event (#247).** A failed shower lookup left the previous
  event's shower loaded, so the antenna response was computed for the wrong
  zenith, azimuth and Xmax, exit 0. `Efield2Voltage.get_event` and
  `convert_efield2efield.py` now check that the loaded shower and run entries
  are the requested ones and raise `KeyError` otherwise.
- **`Efield2Voltage` refuses an event the input does not hold (#238).** A
  missing `(event_number, run_number)` used to leave the previously loaded event
  in place and write it, with an empty trace, under the requested numbers,
  exit 0. It now raises `KeyError` listing the events the input holds. NumPy
  integers are accepted as event and run numbers, and a negative `event_idx`
  is refused instead of wrapping round to the last event.
- **`--rf_chain_nut` / `--rf_chain_gaa` now apply without noise or the main
  chain (#227).** With `--no_noise --no_rf_chain`, the nut or GAA chain was
  multiplied in the frequency domain but never transformed back, so the output
  was V_oc, bit for bit. `final_resample` now inverts the spectrum whenever any
  of noise and the three chains is on.

- **Numbers given as text are accepted again in tree fields (#207, a regression
  from #179).** The input checks refused `"1618"` in a numeric field, which broke
  `ZHAireSRawToRawROOT.py <run>` (the README's one-argument form takes the event
  number from the file name) and `sim2root.py -la/-lo/-al`. Text that reads as a
  number is converted again, now with a `GRANDlibWarning` naming the field; other
  text is still refused. Both callers now pass numbers: `extract_event_number`
  returns an `int`, and the three options are `type=float`. `-al 0` was always
  ignored (the code tested `if clargs.altitude:`) and is now honoured. Tests in
  `tests/sim2root/test_committed_samples.py` run both commands.

- **`EventList` reads a file, an open `TFile` or a `DataDirectory`, not only a
  directory name.** Given a file name, `get_number_of_events()` raised
  `TypeError`, and `get_event()` raised `AttributeError` when the file held no
  run tree; such events now have no antennas, as other missing trees are
  skipped. Given a `ROOT.TFile`, `get_event()` raised `AttributeError`; given a
  `DataDirectory`, iterating raised `TypeError` (the gap left by #94). All four
  forms now list, count and iterate the same events, checked on the committed
  sample run.

- **The ZHAireS example in `sim2root/README.md` works again.** The committed
  `.rawroot` samples predated the current format, and `sim2root.py` stopped on
  them with `IndexError`; they are regenerated with the README's own
  commands. Those commands failed too: `ZHAireSRawToRawROOT.py` (and
  `CoreasToRawROOT.py`) imported `sim2root.*` without the repository on the
  path. Both converters now add it. A test converts the committed samples, so
  it fails if they fall behind again.

- **The simulation-pipeline scripts show each step's output as it comes**
  (#121). `RunSimPipe.py` and `RunSimPipeNoJitter.py` captured every step's
  output with `communicate()`, showed nothing until the step ended, then
  printed stderr as one raw byte string; the streaming version in
  `RunSimPipeADCNoise.py` read it through a pipe and was reported to freeze
  on 1000-event runs. All three now run each step through
  `sim2root/Common/pipeline_step.py`, which lets the step write straight to
  the terminal or job log: no buffering and no pipe to fill. A failed step is
  now reported with its exit status; the pipeline still continues, as before.
  The commands run are unchanged.

- **`convert_voltage2adc.py` writes the ADC file beside its voltage file,
  whatever the directories are called.** Without `-o`, it named the output by
  replacing the first 'voltage' and 'L0' in the whole *path*, so when a
  directory's name contained either, as in `/tmp/test_efield2voltage_x/`, that
  name was changed instead of the file's. The script then stopped with
  `OSError: Failed to open file .../voltage_1-1_L1_0000.root`, or wrote into
  whichever existing directory the changed path named. Only the file name is
  changed now. A test runs the script on such a directory.

- **Seven open issues, fixed together in 2026-09.**
  - **Simulation files record Xmax and the shower direction** (#104).
    `tshower.xmax_pos` and `tshower.direction` were all zeros in `sim2root`
    output. `xmax_pos` is now Xmax in the site frame (that of `du_xyz` and
    `shower_core_pos`): the above-ground `xmax_pos_shc` plus the core, the
    point readers already computed; NaN when unknown. `direction` is the unit
    vector of propagation, minus the "comes from" vector of zenith and
    azimuth, so its z is negative for a downgoing shower; it equals the
    `primary_inj_dir_shc` the ZHAireS converter already wrote. The AOI now
    writes both `xmax_pos` and `xmax_pos_shc`: it wrote only `xmax_pos`, as
    the core-relative point, and read `xmax_pos_shc`, so Xmax was lost on a
    round trip. The `TShower` zenith and azimuth docstrings said "pointing
    to"; they now say "comes from", as decided on 2026-09-24.
  - **Reading many files no longer leaks their memory** (#71). A tree that
    opened its ROOT file from a name never closed it, because PyROOT gives up
    ownership after `TTree.SetDirectory()`. `stop_using()` now closes that file
    once no other tree uses it; a `TFile` the caller passed in is left open.
    Trees are also context managers, `with TADC(path) as tadc: ...`.
    Measured on 200 distinct ADC files: 2.65 MB per file before, 0.15 after.
    About 0.08 MB per file remains inside ROOT 6.36 itself, even from C++.
    Releasing trees is now documented in the data-model and troubleshooting
    pages.
  - **Showers that hit no antenna go through the pipeline** (#91). They are
    kept at every level, with their shower and run information and
    `du_count` 0, so that effective-area studies can count them. Before,
    `sim2root` stopped with "ecef coordinates must be n x 3", and
    `Efield2Voltage` and `convert_voltage2adc.py` with an `IndexError`.
  - **No more `TTreeCache::FillBuffer` errors when appending events** (#89).
    Writing more than about 100 events one at a time, as
    `convert_efield2voltage.py` does, made ROOT report a cache inconsistency
    on every event. Values were right; the message was noise, and hidden
    since 2024-08 by the error level `grand.dataio` sets. The cache's entry
    window is now reset before every full pass over a tree.
  - **The ZHAireS refraction-model reader works** (part of #140).
    `GetRefractionIndexModelFromSry` failed on every call (a misspelt file
    handle) and would otherwise have answered "Constant" for any file.
    Nothing calls it yet; the layered model #140 asks for will need it.

- **The reconstruction examples save their results again.**
  `examples/analysis/main_DOI.py` stored every reconstructed value in a
  `TShower`, and `main_AOI.py` in the event's `Shower`, but neither has those
  fields. Setting an undeclared field on a tree raises nothing, and the file
  was written without it, so nothing was saved; the printout read back the
  in-memory object and looked right. The examples relied on a `TShower`
  extension that was never committed (817bb79e). Both now use `TRecons`,
  which declares every one of these fields and which `display.py` already
  reads. `main_AOI.py` writes it to `reconstructed_events_AOI/recons.root`
  beside its per-event files. A test checks that every value the two scripts
  store is a `TRecons` field. Also fixed there: `sep='\s+'`, an invalid
  escape that Python warns about.

- **Four long-standing issues.**
  - `EventList.get_event` with an event number the input does not hold now
    returns `None`, as documented (#95). It used to crash deep in the reader,
    or, after a valid event, return an event labelled with the requested
    number but carrying the previous event's traces.
  - `convert_voltage2adc.py` finds noise files in a directory given without
    a trailing '/', and stops when it finds none (#122). Before, every ADC
    sample became 8192. A path prefix still works as before.
  - The CoREAS converter reads the hadronic model and CoREAS version from the
    log instead of matching one literal string (#123). Other models and
    versions are no longer written as "n/a"; Sibyll 2.3d and 1.4 read as
    before.
  - A `creation_datetime` or `source_datetime` set on a tree now reads back
    as the datetime that was set, not its integer timestamp. Found while
    triaging #136; an unset value still reads as 0, as its maintainer chose.

- **Xmax is placed right whatever convention a file uses**
  (grand-mother/grand#160). The committed ZHAireS samples store Xmax above sea
  level, 1264 m too high; today's converter stores it above the ground. Before
  this change, only `root_files.py` corrected for it, and it subtracted the
  ground altitude from every file, so freshly converted data came out 1264 m
  low. `gen_shower.py` (the antenna response), `aoi/event.py` and the event
  viewer corrected nothing, so they were wrong on the samples, the viewer's
  angular plane by up to 6.5°. The new `grand/dataio/xmax_frame.py` reads the
  frame from the file: Xmax must lie along the stored arrival direction, and
  only one reading does (0.001° against at least 0.49° on all 12 committed
  events). All four readers use it. A value that fits neither reading is left
  as stored, with a warning. Decided 2026-09-24: detect the frame rather than
  regenerate the samples, since DC2 data was written the old way too.

- **The Handbook PDF warns against `requirements_vers.txt`.** A new erratum,
  because the PDF is compiled from the unchanged LaTeX; the online page was
  already corrected.

- **The recovery paperwork counts branches correctly.** The plan said "5
  still out" when three were: `docs/dev/branch_facts.py` measured branches
  against the local `dev-next`, which can lag the remote, and counted the
  recovery's own working branch as undecided work. It now measures against
  `origin/dev-next` and shows that branch as the working branch. The
  regenerated diagrams also show the 634 tests CI reports, and no longer list
  the reconstruction scope as blocked, since it was decided on 2026-09-24.

- **Installed GRANDlib contains the Galactic noise model.** `grand/sim/noise/`
  had no `__init__.py`, so package discovery skipped it and a built package
  shipped without `galaxy.py`. Found by a new test that checks every
  directory of code under `grand/` is packaged; the same test caught
  `grand/analysis/coords/` before it could ship.

- **The CoREAS site table is in metres and names an unknown site.** It
  stored altitudes in centimetres (kept out of the output only by a later
  override line) and crashed on any other site with "not enough values to
  unpack". Xiaodushan, a real GRAND site, was one of them. It now raises
  `ValueError: unknown site 'Xiaodushan': ... knows only Dunhuang, Lenghu`.

- **The event viewer's vxB axes point the right way.** It treated
  `magnetic_field` -- [inclination, declination, strength] -- as a vector
  and normalised it, giving a field 104° from the real one at Xiaodushan
  and turning its shower-plane and angular-plane axes by 91–114°. It now
  builds the direction from the angles; a new test compares it with the
  geomagnetic model at the site, and fails on the old code.

- **The golden pipeline test no longer fails at random on CI.** It divided
  each difference by that sample's own value, so rounding on a sample near a
  zero crossing read as a large relative error. On 2026-09-24 one leg failed
  at 2.59e-5 on a sample of -1.178 in a trace peaking at 3258, where the
  actual difference was 9.4e-9 of the peak. Differences are now measured
  against each trace's peak; real changes (the √2 noise fix, a one-sample
  shift, a 1e-5 change) still fail it.

- **The CoREAS converter runs on its own fixture.** `CoreasToRawROOT.py -d
  proton` raised `UnboundLocalError: Xmax_NWU` partway through and left a
  447-byte `.rawroot` behind (grand-mother/grand#159). `Xmax_NWU` was computed
  only when the `.reas` carries `ShowerZenithAngle`, and the short
  `SIMxxxxxx.reas` that directory mode reads never does. That branch has no
  `DistanceOfShowerMaximum`, so it now writes NaN for the position; zeros would
  have read downstream as a real Xmax at the array origin. The CoREAS chain now
  runs end to end through `sim2root.py` for the first time, and writes
  `origin_geoid[2] = 1200` in metres, where the 2024 fixture carries `114200`.

  NaN is one of several defensible choices, and the known-issues entry had
  left that choice to the owners of `sim2root/`. Reading the long per-event
  `.reas`, which the fixture has, would give a real position instead. See
  `issue-coreas-xmax-unbound`.

- **A vertical CoREAS shower keeps its own parameters.** The same converter
  tested `if read_params(...ShowerZenithAngle):`, so a zenith of exactly 0.0
  was falsy and fell through to hard-coded Dunhuang values. It now tests for
  presence (`is not None`). This one is reasoned from `read_params`, not
  measured; no fixture has a vertical shower.

- **Tree classes can be constructed again under NumPy 2.** `TRun()`,
  `TADC()` and every other tree raised `ValueError: setting an array element
  with a sequence`. `TTreeScalarDesc.__set__` bound `value` to the same array
  as `inst` when a dataclass field took its default, so the assignment became
  `inst[0] = inst`; NumPy 1 tolerated a one-element array in a scalar slot and
  NumPy 2 does not. `create_default()` had already installed the default, so
  the assignment is now skipped.

  Nothing in the code changed when this appeared — the environment moved. The
  container in which CI last ran successfully dates from January 2022 and
  carried NumPy 1, which is why no test caught it. The suite went from 123
  failed / 216 passed to 15 failed / 336 passed.

- Malformed table in the architecture documentation, where em-dashes pushed
  the first column past its rule.

### Added

- **Files record the code that wrote them** (#137). Every new tree's
  `modification_software` and `modification_software_version` hold
  "GRANDlib" and the package version with the git branch, commit and whether
  tracked files were modified, e.g. `0.1.0.dev1 (git dev-next 1aeac037...)`.
  Both were always empty. Outside a git checkout, only the version is given,
  so an install inside another repository never reports that repository's
  commit. In `grand.provenance`.

- **An offline T1 trigger in `convert_voltage2adc.py`** (#139). The
  DAQ-style T1 logic of `scripts/T1_trigger_offline.py` is now
  `grand.sim.detector.trigger`, and it evaluates every DU: the script looked
  only at the first. The converter gains an opt-in `--t1_trigger` (with
  `--t1_param KEY=VALUE` overrides) that sets `trigger_flag` per DU (1 =
  passed); without it the output is unchanged, checked branch by branch. The
  default parameters are the offline script's and **need confirming by the
  trigger group**: with them, no DU of the committed sample triggers, even
  with pulses of about 1000 ADC, because crossings must be at most 8 ns apart
  (`t_sepmax=10`).

- **Cramér-Rao bounds for the reconstruction** (Sebastián Castro-Isern,
  PR 150). `grand.analysis.cramer_rao_bounds` (also `grand.analysis.crb`)
  gives the smallest uncertainty the PWF, SWF and ADF fits can reach, from
  the timing and amplitude uncertainties. `TRecons` gains ten `crb_*` fields
  to store them, a data-format addition (schema snapshot updated). Files
  written before this still read, with the bounds reading as 0.
  `examples/analysis/main_DOI.py` computes and stores them. Ported by hand
  because the PR targets `dev`; the other files in PR 150 were already on
  `dev-next` from `dev_marion`, apart from its copy of the example, which
  duplicates `main_DOI.py`. Two changes to the code as submitted:
  - an azimuth, or any other parameter, of exactly 0 got a derivative step
    of 0, and every bound came back NaN; it now gets a step of 1e-6;
  - three docstrings are corrected: one named the wrong model and two called
    the zenith the azimuth.
  Tested against 1000 refits of noisy plane-wave times: the PWF bound matches
  their scatter to within 3 %.

- **Branches can be retired without losing them.** `docs/dev/archive_branches.py`
  tags each settled branch as `archive/<branch>-<YYYY-MM>`, with the reason in
  the tag message, before deleting it. Every commit stays in the repository,
  and `git branch <name> <tag>` restores one. `BRANCHES.md` gains an
  "Archived as" column and keeps listing deleted branches from their tags. Its
  "Still out" heading, which counted decided branches too, is now "Not in
  `dev-next`", with the breakdown.

- **Every branch is now settled.** `refact_galaxy`, the last one open, is
  not merged, with luckyjim's agreement: its fixes are already on the trunk
  and the rest re-implements the verified galactic-noise model.

- **Star-shape trace interpolation is kept, in `examples/old/radio/`.** From
  the 2019 `radio` branch (Anne Zilles, Valentin Niess): the only code in the
  repository that interpolates an electric-field trace to a new antenna
  position. Copied with the two helpers it imports; it does not run as it is,
  and the README says what porting would take. The rest of the branch is
  rewritten on the trunk or unfinished, so the branch is retired.

- **The DC1 analysis scripts are kept, in `examples/old/dc1/`.** `beta_dc1`
  (grand-oma, January 2023) was merged to keep its two display scripts and
  their history. They use the pre-2023 API and do not run; the README lists
  the four renames needed to port them.

- **Two notebooks: reconstruction, and the event viewer.**
  `notebooks/11_reconstruction.ipynb` runs `grand.analysis` step by step: plane
  wave, spherical wave, angular distribution function and energy proxy. It
  starts on showers it generated, so the answer is known, and ends on the ten
  GP13 candidates in `examples/analysis/`. There it shows what the tests could
  not: most of those events have timing χ²/ndf far above 1, and 8 of 10
  amplitude fits stop on the upper bound of the ring width, which nothing
  flags. `notebooks/12_event_viewer.ipynb` shows how to run the viewer, redraws
  each panel as a static figure so it reads on GitHub, and measures the
  angular-plane error from #160 on both sample events (6.5° and 1.0°), which
  the Xmax detection below now corrects.
  Both are generated by `notebooks/make_notebooks.py` and listed in the
  documentation. The weekly notebook job now installs the viewer extra.

- **The ADF fit has a round-trip test.** Amplitudes generated by
  `ADF_parameters` fit back to their direction, width and scale. It is the
  only fit independent of timing, and it had no test. Checked to fail when
  the fit's azimuth bound is narrowed.

- **Reconstruction: `grand.analysis`** (Marion Guelfand, from `dev_marion`).
  Plane-wave, spherical-wave and ADF fits of arrival direction and Xmax
  distance, an electromagnetic-energy proxy, signal extraction, footprint
  geometry and a Cherenkov-angle model, with results in a new `TRecons` tree.
  Needs `iminuit` (`pip install -e ".[analysis]"`; in the conda
  environment). Tested for self-consistency -- each fit recovers what its
  forward model generates -- not yet against simulated or measured showers.
  Examples in `examples/analysis/`.

- **GRAND's angle convention is stated and pinned.** A shower's zenith and
  azimuth name where it comes from; the coordinates page now says so, and
  `tests/geo/test_angle_convention.py` checks the core transform against the
  ZHAireS summaries. Chosen over `snonis_sim2root_test_merge`, which flipped
  the transforms and would have put every computed angle at odds with every
  stored one.

- **Decision material for the three branches waiting on the collaboration**
  (`dev_marion`, `grandio_light`, `snonis_sim2root_test_merge`), in
  `resources/dev/dev-next/DECISIONS.md`: what each changes, what merging it
  would also do, and the question it asks, each measured.

- **The schema snapshot notices new trees.** It compared only a hard-coded
  list of tree classes, so a new tree -- the largest format change there is
  -- passed unseen: `dev_marion`'s 27-field `TRecons` left it green. A new
  test finds every tree class in the data modules and fails on any that is
  not pinned. It also found `TRunRawVoltage`, on the trunk and never pinned;
  it is now in the snapshot.

- **A `Tests gate` CI check** that is green only when the test suite ran and
  passed, or when a commit changed documentation only, in which case it says
  that nothing ran. A skipped test job used to read as a pass. Designed to be
  the required check for `dev-next`; making it required is an admin setting.

- **Acknowledgement of the NCN OPUS grant** (no. 2022/45/B/ST2/02889) in
  `README.md`, copied verbatim from `dev`, where it was added to `README.rst`
  on 2026-09-15 after `dev-next` had replaced that file.

- **One environment for everything.** `env/conda/grand-dev.yml` consolidates
  four dependency lists that had drifted apart: the previous runtime file, the
  pip-installed test and lint tools, a third set under `env/docker_*/` that was
  the only one carrying `numba` and `lmfit`, and documentation dependencies
  that nothing declared. Build with `--solver=libmamba`.

- **The package is installable.** `pyproject.toml` makes `pip install -e .`
  work, so `import grand` no longer needs `PYTHONPATH`. Scope is deliberately
  narrow: the C extension is still built by `env/setup.sh`, and no console
  entry points are declared yet.

- **Documentation, rebuilt.** A single Sphinx tree at `docs/source/`, replacing
  five scattered sources. Narrative pages for installation, quickstart,
  coordinates, the data model, architecture and known issues, with examples
  that execute against the real library when the page is built. Appendix A of
  the paper is ported into the coordinates page.

- **Known issues page** recording the Galactic-noise normalisation, the NUTRIG
  field-name collision, and the import-time ROOT dependency, each with what is
  measured, what is not, and what would settle it.

- **Tests.** Schema snapshot pinning the ROOT tree layout, with a guard against
  the NUTRIG name collision; Galactic-noise normalisation, asserting what holds
  under any convention and recording what does not; descriptor-default
  regression covering every tree class.

- **The converters are run, not just read.** `tests/sim2root/test_xmax_frame.py`
  runs the ZHAireS and CoREAS converters on the repository's own fixtures and
  checks Xmax against the ZHAireS `.sry` summaries. Until now `tests/sim2root/`
  only parsed the sources, which is how a crash on the committed CoREAS
  fixture went unnoticed. Each test was checked to fail with its defect put
  back.

- **Tools.** `quality/premerge_check.py` reports what a branch adds and flags
  fields that duplicate an existing field's meaning; `quality/docstring_coverage.py`
  measures the numpydoc conversion; `docs/dev/make_recovery_diagram.py` and
  `make_frames_diagram.py` generate the documentation figures.

- Numpydoc docstrings, in progress, enforced by a ruff `D` ratchet whose
  ignore list may shrink and must never grow.

### Changed

- **Public functions check their input, with messages that start with
  `GRANDlib:`.** A review found that only 18 % of the 328 public functions
  checked their input at all; of 60 bad inputs, 27 were accepted silently,
  13 failed deep inside with an unrelated exception, and one called
  `exit()`. Now 50 of the 60 are refused with a message naming the function,
  the argument, what was expected and what was given, for example
  `ValueError: GRANDlib: Geodetic: 'latitude' must be between -90 and 90
  degrees, got 200.0`. The exceptions are the standard `TypeError`,
  `ValueError` and `FileNotFoundError`. Suspicious but usable values give a
  `GRANDlibWarning` instead. The helpers are in the new
  `grand.basis.validate`; the convention is in the contributing guide and
  the troubleshooting page. In particular:
  - coordinates: latitude within ±90°, all of latitude, longitude and height
    given, arrays of equal length; NaN warns. `Geodetic` no longer shifts a
    negative longitude inside the caller's own array;
  - data-tree fields: no silent change on storage (`run_number = 1.7` was
    stored as 1, `-1` as 4294967295, a numeric string converted); fixed
    shapes enforced; `du_xyz` rows must be (x, y, z). Physical ranges
    (zenith, energies, Xmax depth, time-bin size) warn rather than raise,
    because existing files hold placeholders such as `xmax_grams = -201`;
  - files: a missing file or directory, or a file that is not ROOT, is named
    as such; `pathlib.Path` is accepted. Creating a new file by opening a
    tree on it still works; only a missing directory is refused;
  - `EventList` raises instead of calling `exit()`, accepts `os.PathLike`,
    and refuses an empty directory; `Event.write` likewise;
  - reconstruction (`grand.analysis`, which had no checks): antenna positions
    must be (N, 3) and finite, with enough antennas for the fit (3 for the
    plane wave, 4 for the spherical and ADF fits); times and amplitudes one
    per antenna; uncertainties positive. `PWF_semianalytical` and the loss
    functions raised instead of printing and returning `None`. The Cramér-Rao
    bounds warn when the antennas barely constrain the parameters (2
    antennas used to give bounds of 10^5 rad). `recons_energy_from_voltage`
    now takes arrays, as documented, and refuses `sin_alpha = 0`;
  - simulation: `AntennaModel` and `galactic_noise` refuse an unknown
    `du_type`; the VGA gain must be one of the four tabulated values;
    frequencies must not be negative; `assert`s on user input became errors;
  - two `raise` of a plain string (itself a `TypeError`) fixed.
  The CoREAS converter passed its event number as the string `"004100"`; it
  now converts it explicitly. `tests/test_input_validation.py` (28 tests)
  keeps all this.

- **The Handbook's "Directory Structure" page is maintained by hand.**
  `docs/dev/build_handbook.py` rewrote every Handbook page from the LaTeX
  source, and in September 2026 a regeneration silently undid the 2026-09-08
  correction on that page, which removed the instruction to install
  `requirements_vers.txt`: a 2022 pin set including a Pillow with a critical
  advisory. The correction was restored at the time; the decision now (the
  repository owner's) is that the hand edit wins. The generator keeps any page
  listed in `HAND_MAINTAINED`, and refuses, before touching anything, if such
  a page no longer matches a section of the source. The PDF is built from the
  LaTeX and still carries the old instruction.

- **`grand.recon` removed.** It held two classes with a constructor and
  nothing else; reconstruction is `grand.analysis`.

- **Reading GRAND files no longer loads the physics.** `grand/__init__.py`
  loads its public names on first use instead of importing the geometry and
  simulation code eagerly. `import grand.dataio` now loads the data layer
  only, and works without the compiled C core; `import grand` and
  `grand.geo.coordinates` no longer need ROOT. Every public name still
  resolves, and `from grand import *` now works (it raised on `adc`, listed
  in `__all__` but never imported). Chosen over the `grandio_light` branch,
  which reached the same goal by deleting the physics.

- Merged from the outstanding queue: `dev_fix_root_warnings_lwp` (ROOT 6.38
  warnings), `dev_nutrig_fields`, `dev_reprocessing` (Snakemake pipeline),
  `dev_Event_write`, and `dev_aoi_unittest` (~3000 lines of tests, stripped of
  ten generated summary documents and three stray fixtures).

### Known

- `magnetic_field` stores [inclination, declination, strength] with the
  strength in µT from ZHAireS and in gauss from CoREAS, and no unit
  recorded. See `issue-magnetic-field-units`.
- The Galactic-noise normalisation does not reproduce the tabulated model: the
  simulated RMS is 0.33 of the Parseval value with the current `size_out/2`,
  and would be 0.47 with the proposed `size_out/sqrt(2)`. Neither is 1. See
  the known-issues page.
- Two branches add the same NUTRIG correlation fields to `TADC` under
  different names; the schema decision is unresolved.
- CI has not completed a run in a long time, for three independent reasons
  recorded in `docs/dev/FINDINGS_CI.md`.
