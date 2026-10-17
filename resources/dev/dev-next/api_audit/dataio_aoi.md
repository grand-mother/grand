# API audit, step 2: `grand.dataio` and `grand.aoi`

What a caller written for `dev` (frozen as `archive/dev-2026-09`) sees on dev-next (86680cb0), for every public function whose code changed. The method and the setup to repeat the examples are in [#293](https://github.com/grand-mother/grand/issues/293).

Each function was read by an agent, old and new code side by side, and classified with the codes below. Every row with a behavior change came with an example, run on both versions. The rows below were then reviewed against those outputs, and each change that needs a decision became an issue. The functions of `ShowerEvent` (`grand.sim`) are included because they share the checks of `grand.aoi`.

## Summary

A function can count in several columns: most changes that break a `dev` call also fix a bug.

| | `grand.dataio` | `grand.aoi` |
|---|---|---|
| Changed public functions | 73 | 26 |
| (1) Incompatible: raise, return or write something else for input that worked in `dev` (A, W) | 48 | 18 |
| of which also fix a bug | 44 | 18 |
| (2) Improvements: checks, units, new options, messages (C) | 36 | 18 |
| (3) Non-critical fixes (B2) | 36 | 11 |
| Serious fixes: wrong results, lost data, crashes (B1) | 21 | 14 |
| No visible change | 10 | 2 |
| Examples run on both versions | 45 | 18 |

The incompatible changes are grouped by the decision they need in 13 issues, #312 to #324, besides the 11 of step 1 (#294 to #304). #312 is a crash in dev-next that `dev` does not have.

## Changes that only fix filed bugs

These change what a `dev` caller sees, but only where `dev` was wrong, so they need no decision. Each is shown by an example: a script in `examples/`, or a reader's snippet with the output of both versions.

**`Event.write(out_dir=..., overwrite=True)` deletes only its own files, not the whole folder (#212)**

Example `examples/aoi_write_dir.py`. Output with `dev`:

```text
output holds ['notes.txt']
No valid trun TTree in the file output/run_00000_L1_0000.root. Creating a new one.
Writing trun
No valid tvoltage TTree in the file output/voltage_20261017_150653_0-0_L0_0000.root. Creating a new one.
Writing tvoltage
No valid tefield TTree in the file output/efield_20261017_150653_0-0_L1_0000.root. Creating a new one.
Writing tefield
No valid tshower TTree in the file output/shower_20261017_150653_0-0_L0_0000.root. Creating a new one.
Writing tshower
after event.write(out_dir='output', overwrite=True), output holds ['efield_20261017_150653_0-0_L1_0000.root', 'run_00000_L1_0000.root', 'shower_20261017_150653_0-0_L0_0000.root', 'voltage_20261017_150653_0-0_L0_0000.root']
```

With dev-next:

```text
output holds ['notes.txt']
after event.write(out_dir='output', overwrite=True), output holds ['efield_20261017_150658_0-0_L1_0000.root', 'notes.txt', 'run_00000_L1_0000.root', 'shower_20261017_150658_0-0_L0_0000.root', 'voltage_20261017_150658_0-0_L0_0000.root']
```

**`EventList` starts at `start_entry` or `start_event`, which `dev` ignored (#213)**

Example `examples/aoi_iter.py`. Output with `dev`:

```text
EventList(SAMPLE)                   -> [13790, 1618]
EventList(SAMPLE, start_entry=1)    -> [13790, 1618]
EventList(SAMPLE, start_event=1618) -> [13790, 1618]
EventList(SAMPLE, start_entry=5)    -> [13790, 1618]
```

With dev-next:

```text
EventList(SAMPLE)                   -> [13790, 1618]
EventList(SAMPLE, start_entry=1)    -> [1618]
EventList(SAMPLE, start_event=1618) -> [1618]
EventList(SAMPLE, start_entry=5)    -> ValueError: GRANDlib: EventList: start_entry 5 is out of range: the input holds 2 events
```

**`Event.fill_antennas(gp300_workaround=False)` works for GP13; `dev` raised a string**

Example `examples/aoi_antennas.py`. Output with `dev`:

```text
GP300 workaround: calculating all antennas positions
TypeError: exceptions must derive from BaseException
```

With dev-next:

```text
44 antennas; the first: [(46, 6062.1), (55, 6928.1)]
```

**`Event.write_run` works; `dev` raised `ValueError` (see case 1 of the example)**

Example `examples/aoi_write_attach.py`. Output with `dev`:

```text
No valid tefield TTree in the file built.root. Creating a new one.
1. built in memory, after write_efields: event.tefield is a TEfield in built.root
No valid trun TTree in the file run.root. Creating a new one.
   write_run("run.root") -> ValueError: Incorrect type for t_bin_size <class 'int'>. Either a list, an array or a ROOT.vector required.
2. read from files: event.tefield is a TEfield in input/efield_1618-13790_L1_0000.root
   write_efields("read.root", overwrite=True) -> TreeExists: The tefield TTree already exists!
   then event.tefield is a TEfield in input/efield_1618-13790_L1_0000.root
```

With dev-next:

```text
1. built in memory, after write_efields: event.tefield is None
   write_run("run.root") -> written
2. read from files: event.tefield is a TEfield in input/efield_1618-13790_L1_0000.root
   write_efields("read.root", overwrite=True) -> written
   then event.tefield is a TEfield in input/efield_1618-13790_L1_0000.root
```

**Reading with one tree object no longer changes the fields of another object on the same file (#273)**

```python
import os, shutil, tempfile
from grand.dataio.event_trees import TEfield
S = "/home/user/grand/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"
p = shutil.copy(os.path.join(S, "efield_1618-13790_L0_0000.root"), tempfile.mkdtemp())
a = TEfield(p)
b = TEfield(p)  # a second object on the same file
b.get_entry(0)
before = int(b.event_number)
a.get_entry(1)
print(repr((before, int(b.event_number))))
```

Output with `dev`:

```text
(13790, 1618)
```

With dev-next:

```text
(13790, 13790)
```

**A second process writing to the same file gets an error instead of corrupting it (#281)**

```python
import os, shutil, subprocess, sys, tempfile
from grand.dataio.event_trees import TEfield
S = "/home/user/grand/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"
p = shutil.copy(os.path.join(S, "efield_1618-13790_L0_0000.root"), tempfile.mkdtemp())
t = TEfield(p); t.get_entry(0); t.event_number = 1; t.fill()
child = ("from grand.dataio.event_trees import TEfield\n"
         f"t = TEfield({p!r}); t.get_entry(0); t.event_number = 2; t.fill(); print('child filled')")
r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
print(repr((r.stdout.strip() or r.stderr.strip()).splitlines()[-1]))
```

Output with `dev`:

```text
'child filled'
```

With dev-next:

```text
'OSError: GRANDlib: TEfield: cannot open /tmp/tmp1zzw4y1d/efield_1618-13790_L0_0000.root: another process is writing it; wait for it to finish'
```

**`DataDirectory.get_list_of_files(recursive=True)` lists subfolders (#204)**

```python
import os, shutil, tempfile
from grand.dataio.data_handling import DataDirectory
S = "/home/user/grand/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"
d = tempfile.mkdtemp()
os.mkdir(os.path.join(d, "sub"))
shutil.copy(os.path.join(S, "run_1_L0_0000.root"), d)
shutil.copy(os.path.join(S, "run_1_L1_0000.root"), os.path.join(d, "sub"))
dd = DataDirectory(d)
print(repr([os.path.relpath(f, d) for f in dd.get_list_of_files(recursive=True)]))
```

Output with `dev`:

```text
['run_1_L0_0000.root']
```

With dev-next:

```text
['run_1_L0_0000.root', 'sub/run_1_L1_0000.root']
```

**`v += x` on a vector field appends; with ROOT 6.36 `dev` replaced the contents (#201)**

```python
from grand.dataio.descriptors import StdVectorList
v = StdVectorList("float", [1.0, 2.0])
v += [3.0]
print(repr(len(v)))
```

Output with `dev`:

```text
1
```

With dev-next:

```text
3
```

## Codes

| Code | Meaning |
|---|---|
| `A1` | a public name or parameter removed or renamed |
| `A2` | a call binds its arguments differently, or fails, because of the signature |
| `A3` | a call that completed in `dev` now raises |
| `A4` | same input, different result, side effect or printed output |
| `A5` | different contents written to files |
| `W` | a new warning; the call otherwise completes as before |
| `B1` | fixes a serious bug: wrong results, lost data or a crash on valid input |
| `B2` | fixes a minor bug: unusual input, misleading message, resource leak |
| `C1` | a new input check, on input `dev` handled badly |
| `C2` | units, limits or metadata added |
| `C3` | a new optional parameter or feature |
| `C4` | refactor or cleanup |
| `C5` | log or message text changed |
| `same` | nothing a caller can observe |

## `grand.dataio`

| Function | Classes | What a caller sees on dev-next | Issue |
|---|---|---|---|
| `dataio.data_handling.DataDirectory.__getattr__` | A3, B2 | Only exact tree attribute names (trun, ftadcs, tefield_l1...) return None when absent; other names merely containing them, like tadc_L1 or my_tefield, now raise AttributeError. | #322 |
| `dataio.data_handling.DataDirectory.__init__` | A3, W, B2, C1, C3 | Missing folder raises FileNotFoundError (dev: empty object); analysis_level not an int (even 1.0) or absent from the files raises ValueError (dev: trees None); unrecognised files warn. | #322 |
| `dataio.data_handling.DataDirectory.get_list_of_files` | A4, B1 | recursive=True now also returns .root files in subfolders; dev's pattern had no '**', so it listed only the top folder. | fix, #204 |
| `dataio.data_handling.DataDirectory.get_list_of_files_handles` | A4, B2, C5 | Files holding no GRAND tree are dropped with a logged warning (dev kept empty handles); names with fewer than four '_' fields no longer crash; grouping prefers the _L<n>_ level. | #322 |
| `dataio.data_handling.DataDirectory.init_exp_structure` | A4, W, B2 | Files not named *_L<n>_<serial>.root are skipped with a warning (dev parsed another field or crashed); a name/tree level mismatch now warns and uses the tree's level. | #322 |
| `dataio.data_handling.DataDirectory.init_sim2root_structure` | A4, W, B2 | Files not named *_L<n>_<serial>.root are skipped with a warning (dev parsed another field or crashed); a name/tree level mismatch now warns and uses the tree's level. | #322 |
| `dataio.data_handling.DataDirectory.init_structure` | A4, W, B2 | Files not named *_L<n>_<serial>.root are skipped with a warning (dev parsed another field or crashed); a name/tree level mismatch now warns and uses the tree's level. | #322 |
| `dataio.data_handling.DataDirectory.print` | same | No visible change. |  |
| `dataio.data_handling.DataFile.__init__` | A3, A4, B2, C1, C3, C5 | Raises ValueError for files without GRAND trees or closed TFiles, FileNotFoundError for root:// URLs (dev returned empty/opened them); accepts Path; chained trees gain metadata; info['dus'] computed when first read. | #322 |
| `dataio.data_handling.DataFile.close` | W, B2 | Marks every tree object of that file (even ones opened separately) as closed, so later use raises RuntimeError instead of crashing; warns about filled but unwritten entries. | #320 |
| `dataio.data_handling.DataFile.get_max_list_of_events` | C4 | Same result; an exception raised while listing a tree's events now propagates instead of being swallowed by a bare except. |  |
| `dataio.data_tree.DataTree.__post_init__` | A3, A4, B2 | After construction, setting an unknown public attribute (t.zenit = 5) raises AttributeError; dropping the last reference to a tree now closes the file it opened. | #318, #320 |
| `dataio.data_tree.DataTree.analysis_level.setter` | same | No visible change. |  |
| `dataio.data_tree.DataTree.close_file` | B2 | Also marks every tree object stored in that file as closed (later calls raise RuntimeError instead of crashing) and releases its write lock; no error without a file. | #320 |
| `dataio.data_tree.DataTree.comment.setter` | same | No visible change. |  |
| `dataio.data_tree.DataTree.create_branch_from_field` | C5 | A string branch missing from an older file is logged once per file at INFO instead of a WARNING on every read. |  |
| `dataio.data_tree.DataTree.create_metadata` | A4, A5, B1, C2 | New trees record modification_software 'GRANDlib' and its version (dev: empty); creation time is true UTC; creation/source_datetime return datetimes (dev: ints, source 0). | #324 |
| `dataio.data_tree.DataTree.creation_datetime.setter` | A4, A5, B1 | After setting a datetime the getter returns it (dev returned the int timestamp); naive datetimes are stored as UTC (dev: local time); ints read back as naive UTC. | #324 |
| `dataio.data_tree.DataTree.draw` | A4, B1, C1 | Fields named in varexp/selection keep their values after drawing (dev left the last entry's values in them); raises RuntimeError on a closed file instead of crashing. | fix, #273 |
| `dataio.data_tree.DataTree.fill` | A3 | The base fill() raises RuntimeError once the tree's file was closed (dev: silent no-op); concrete trees override fill(). | #320 |
| `dataio.data_tree.DataTree.get_entries` | B2, C1 | On a tree whose file was closed, raises a clear RuntimeError instead of crashing or failing obscurely. | #320 |
| `dataio.data_tree.DataTree.get_entry` | A3, A4, B1, B2, C1 | Out-of-range entry raises IndexError (filed). Also: reading no longer overwrites the fields of another tree object on the same file (#273); NumPy ints accepted; RuntimeError after close. | #294, #273 |
| `dataio.data_tree.DataTree.get_entry_with_index` | A2, A3, A4, B1, B2, C1 | Positional arguments raise TypeError; missing event raises LookupError (filed). Also: reading no longer overwrites another tree object's fields on the same file (#273); NumPy ints accepted. | #295, #294, #273 |
| `dataio.data_tree.DataTree.get_metadata_as_dict` | A4, B2 | Datetime entries become naive UTC datetimes (dev: local time) and a stored 0 becomes datetime(1970, 1, 1) instead of 0; text values under *datetime* keys no longer crash. | #324 |
| `dataio.data_tree.DataTree.get_number_of_entries` | B2, C1 | On a tree whose file was closed, raises a clear RuntimeError instead of crashing or failing obscurely. | #320 |
| `dataio.data_tree.DataTree.modification_history.setter` | same | No visible change. |  |
| `dataio.data_tree.DataTree.modification_software.setter` | same | No visible change. |  |
| `dataio.data_tree.DataTree.modification_software_version.setter` | same | No visible change. |  |
| `dataio.data_tree.DataTree.scan` | A4, B2, C1 | Flushes ROOT's output after scanning, so the table appears before later Python prints when output is piped (dev: it surfaced at exit); RuntimeError on closed files. *Review:* Same output in the example; the difference is only in when the table appears in piped output. |  |
| `dataio.data_tree.DataTree.source_datetime.setter` | A4, A5, B1 | After setting a datetime the getter returns it (dev returned the int timestamp); naive datetimes are stored as UTC (dev: local time); ints read back as naive UTC. | #324 |
| `dataio.data_tree.DataTree.stop_using` | A3, A4, W, B2, C3 | Now closes the file the tree opened itself (close_file=True default) and frees the tree, so later use raises; warns about filled-but-unwritten entries; a second call no longer raises. | #320 |
| `dataio.data_tree.DataTree.type.setter` | same | No visible change. |  |
| `dataio.data_tree.DataTree.write` | A3, A4, A5, W, B1, C1 | overwrite=True replaces only this tree (dev recreated the file, losing other trees); same-name tree without overwrite raises FileExistsError; other-file targets get a copy; locks; unfilled absent tree warns. | #323 |
| `dataio.descriptors.StdStringDesc.__get__` | A4, B2 | Accessing the field on the class (TRun.site) returns the descriptor instead of raising AttributeError, so hasattr(TRun, 'site') is now True. | #321 |
| `dataio.descriptors.StdStringDesc.__set__` | C1, C5 | Non-string values raise TypeError instead of ValueError, with a reworded message (filed). | #296 |
| `dataio.descriptors.StdVectorList.__delitem__` | B2 | del v[i], del v[a:b], pop() and remove() now work, also with negative indices; dev passed the index to std::vector::erase, which needs an iterator, raising TypeError. | fix |
| `dataio.descriptors.StdVectorList.__eq__` | A4, B2 | Comparison with an array or tuple returns one bool (dev: elementwise array, or False for tuples); char elements compare as ints; 3-D and deeper vectors compare correctly. | #321 |
| `dataio.descriptors.StdVectorList.__getitem__` | A4, B2 | Elements of char vectors come back as ints (dev: 1-character strings); slices return plain lists, also on empty vectors; ragged 4-D vectors convert correctly. | #321 |
| `dataio.descriptors.StdVectorList.__iadd__` | A4, B1 | v += x now always appends; on ROOT >= 6.36 dev replaced the contents. Adding a StdVectorList no longer empties the source's inner vectors; a failed add leaves v unchanged. | fix, #201 |
| `dataio.descriptors.StdVectorList.__init__` | A3, C3 | The initial value goes through the assignment conversion: None and NumPy arrays are accepted, but an int (dev: that many zeros) now raises TypeError. | #318 |
| `dataio.descriptors.StdVectorList.__repr__` | A4, B2 | repr shows char elements as ints and vectors deeper than 2-D as nested lists (dev: characters, and cppyy vector objects below the second level). *Review:* The example failed on both versions (a `char` vector built from strings); not confirmed. | #321 |
| `dataio.descriptors.StdVectorList.__setitem__` | A3, B2, C3 | Out-of-range assignment raises IndexError (dev silently ignored it); negative indices and slices now work; NumPy values accepted. | #318 |
| `dataio.descriptors.StdVectorList.asnumpy` | A4, B2 | Char-typed vectors now give a numeric int8/uint8 array; dev gave numpy's conversion of the character vector. *Review:* Not confirmed: the example failed on dev-next while building a `char` vector from strings. See #321 for the field case. | #321 |
| `dataio.descriptors.StdVectorList.insert` | B2 | insert() and therefore append() now work, also with negative indices; dev passed an int where std::vector::insert needs an iterator, raising TypeError. | fix |
| `dataio.descriptors.StdVectorListDesc.__get__` | A4, B2 | Accessing the field on the class (TRun.du_id) returns the descriptor instead of raising AttributeError, so hasattr(TRun, 'du_id') is now True. | #321 |
| `dataio.descriptors.StdVectorListDesc.__init__` | C2, C3 | New optional limits and unit (filed) and inner_length; existing calls are unaffected. | #300, #301 |
| `dataio.descriptors.StdVectorListDesc.__set__` | A3, W, B1, B2, C1, C2, C5 | Limit warnings and TypeError/None refusal (filed). Also: fractions or out-of-range values in integer vectors raise, TRun.du_xyz rows need 3 values, numeric text warns, sources not emptied (#282). | #296, #300, #318 |
| `dataio.descriptors.TTreeArrayDesc.__get__` | A4, B2 | Accessing the field on the class (TRun.origin_geoid) returns the descriptor instead of raising AttributeError, so hasattr(TRun, 'origin_geoid') is now True. | #321 |
| `dataio.descriptors.TTreeArrayDesc.__init__` | C2, C3 | New optional component_limits (filed); existing calls are unaffected. | #300, #301 |
| `dataio.descriptors.TTreeArrayDesc.__set__` | A3, W, B2, C1, C2, C5 | Limit warnings and TypeError for strings (filed). Also: a scalar no longer fills every component (size must match); fractions, out-of-range or float32-overflowing values raise; numeric text warns. | #296, #300, #318 |
| `dataio.descriptors.TTreeScalarDesc.__get__` | A4, B2 | Accessing the field on the class (TRun.run_number) returns the descriptor instead of raising AttributeError, so hasattr(TRun, 'run_number') is now True. | #321 |
| `dataio.descriptors.TTreeScalarDesc.__init__` | C2, C3 | New optional limits and unit (filed); existing calls are unaffected. | #300, #301 |
| `dataio.descriptors.TTreeScalarDesc.__set__` | A3, W, B1, C1, C2, C5 | Limit warnings, None/TypeError refusal (filed); NumPy 2 default fix. Also: fractions in integer fields and float32 overflow raise (dev truncated/stored inf); numeric text warns; overflowing ints ValueError. | #296, #300, #318 |
| `dataio.descriptors.chars_to_uint8_array` | same | No visible change. |  |
| `dataio.event_trees.MotherEventTree.build_index` | B2, C1 | Clears the read cache before indexing, so grown trees no longer print TTreeCache 'Inconsistency' errors; raises RuntimeError if the file was closed. | fix |
| `dataio.event_trees.MotherEventTree.fill` | A3, B1, C1 | Takes an exclusive file lock (OSError if another process writes or changed the file); writes this object's values even if another object rebound the branches (#273); RuntimeError after close. | fix, #281, #273 |
| `dataio.event_trees.MotherEventTree.fill_entry_list` | A4, B1 | run_number and event_number keep their current values after the call (dev left the last entry's values in them); no spurious TTreeCache errors. | fix, #273 |
| `dataio.event_trees.MotherEventTree.get_dus_indices_in_run` | A3, A4, B1 | Indices follow the event's unit order and units missing from the run raise ValueError (filed). | #297 |
| `dataio.event_trees.MotherEventTree.get_event` | A3, A4, B1, B2, C1 | Missing event raises LookupError; numbers must be integers; stale index rebuilt (filed). Also: reading no longer overwrites another tree object's fields on the same file (#273); RuntimeError after close. | #294, #303, #319 |
| `dataio.event_trees.MotherEventTree.get_list_of_all_used_dus` | A4, B2, C1 | Same list; du_id values set but not yet filled now survive the call (dev overwrote them from the tree); RuntimeError on a closed file. | fix, #273 |
| `dataio.event_trees.MotherEventTree.get_list_of_dus` | A4 | Returns the current entry's units as a list (dev: unique units over the whole tree, as an array) (filed). | #298 |
| `dataio.event_trees.MotherEventTree.get_list_of_events` | A4, B1, C1 | Same list; the event_number and run_number fields keep their current values (dev left the last entry's values); RuntimeError on a closed file. | fix, #273 |
| `dataio.event_trees.MotherEventTree.get_traces_lengths` | A4 | Returns per-unit channel lengths of the current entry from trace/trace_ch (dev: per-channel lengths over all entries from trace_x/trace_0) (filed). | #299 |
| `dataio.event_trees.MotherEventTree.has_event` | A3, A4, B1, C1 | Float or string numbers raise TypeError (dev applied int()); events filled but not yet indexed are now found (dev: False until build_index/write); RuntimeError after close. | #319 |
| `dataio.event_trees.MotherEventTree.print_list_of_events` | A4, B1, C1 | Same printout; the event_number and run_number fields keep their current values (dev left the last entry's values); RuntimeError on a closed file. | fix, #273 |
| `dataio.protocol.get` | C1 | A non-string or empty name raises TypeError before downloading (dev built a URL and failed with InvalidBLOB). |  |
| `dataio.root_files.get_file_event` | C1, C5 | A file without tefield/tvoltage/tadc raises ValueError (dev: bare AssertionError); errors carry messages instead of a separate ERROR log line. |  |
| `dataio.run_trees.MotherRunTree.add_proper_friends` | B2 | Clears the read cache before indexing, so growing run trees no longer print TTreeCache 'Inconsistency' errors. | fix |
| `dataio.run_trees.MotherRunTree.build_index` | B2, C1 | Clears the read cache before indexing, so grown trees no longer print TTreeCache 'Inconsistency' errors; raises RuntimeError if the file was closed. | fix |
| `dataio.run_trees.MotherRunTree.fill` | A3, B1, C1 | Takes an exclusive file lock (OSError if another process writes or changed the file); writes this object's values even if another object rebound the branches (#273); RuntimeError after close. | fix, #281, #273 |
| `dataio.run_trees.MotherRunTree.get_run` | A3, A4, B1, B2, C1 | Missing run raises LookupError; numbers must be integers; stale index rebuilt (filed). Also: reading no longer overwrites another tree object's fields on the same file (#273); RuntimeError after close. | #294, #303, #319 |
| `dataio.run_trees.MotherRunTree.has_run` | A3, A4, B1, C1 | Float or string numbers raise TypeError (dev applied int()); runs filled but not yet indexed are now found (dev: False until write); RuntimeError after close. | #319 |
| `dataio.run_trees.MotherRunTree.is_unique_event` | same, C4 | No visible change. |  |

## `grand.aoi` (and `ShowerEvent`)

| Function | Classes | What a caller sees on dev-next | Issue |
|---|---|---|---|
| `aoi.event.Event.close_files` | A4, B1 | Writes and closes only trees queued by write_* with auto_file_close=False; the event's other trees (read from input files, or set or filled directly) are no longer written or closed. | #315 |
| `aoi.event.Event.directory.setter` | C3 | When the new Event.tefield_level is set, picks that level's efield tree (tefield_l<level>) from the directory; unset, behaves exactly as before. |  |
| `aoi.event.Event.file.setter` | C1, C3 | Accepts any os.PathLike. A missing file raises FileNotFoundError, a non-ROOT or damaged one OSError, with a clear message instead of ROOT's generic error. |  |
| `aoi.event.Event.fill_antennas` | A4, B1, B2, C1, C3, C5 | gp300_workaround=False is now honoured for GP80/GP13 sites (dev still used GPS positions). Without a run tree it logs a warning and leaves antennas empty instead of crashing. *Review:* Example: with site GP13, dev raised (`raise` of a string); dev-next fills the antennas. | fix |
| `aoi.event.Event.fill_efield_tree` | same, C4 | No visible change. |  |
| `aoi.event.Event.fill_event_from_efield_tree` | same, C4 | No visible change. |  |
| `aoi.event.Event.fill_event_from_shower_tree` | A4, W, B1, C1, C3 | Xmaxpos z is now height above ground: sea-level files are shifted by the site altitude; off-axis values kept with a logged warning. Also fills primary_type, magnetic_field. | #313, #333 |
| `aoi.event.Event.fill_event_from_trees` | A3, A4, B1, B2, C1, C3, C5 | Raises instead of printing: no input files (dev returned False), entry_number with event/run numbers, or only one of event/run. Honours gp300_workaround; status prints now debug logging. | #317 |
| `aoi.event.Event.fill_event_from_voltage_tree` | A3, B2, C1 | With use_trawvoltage, trawvoltage_channels must be exactly three integers: four or more channels, or bools, now raise ValueError (dev silently used the first three, or bools as 0/1). | #317 |
| `aoi.event.Event.fill_run_tree` | B1, C4 | Now completes: t_bin_size is stored as a one-element vector. Dev always raised ValueError assigning the scalar sampling time to TRun's vector branch. | fix |
| `aoi.event.Event.fill_shower_tree` | A3, A4, B1 | Raises TreeExists when the event already has a tshower and overwrite is False. The tree now gets xmax_pos_shc=Xmaxpos and xmax_pos=Xmaxpos+core; analysis_level is no longer set to 1. | #314 |
| `aoi.event.Event.fill_voltage_tree` | B2, C4 | No longer copies the antennas' atm_*, battery_level and firmware_version (TVoltage never stored them), so events without antennas no longer crash with TypeError. | fix |
| `aoi.event.Event.write` | A4, A5, B1, C1, C5 | With out_dir and overwrite, deletes only the <tree>_*.root files of the event's trees, not the whole directory. The shower is copied from tshower, not rebuilt from event.shower. | fix, #212 |
| `aoi.event.Event.write_efields` | A4, B1 | The written tree is no longer attached as event.tefield; with auto_file_close=False it is queued for close_files instead. | #315 |
| `aoi.event.Event.write_run` | B1, C4 | Now writes the run tree; dev always raised (TreeExists if the event had a trun, else ValueError storing the scalar t_bin_size). The tree is not attached as event.trun. | fix, #315 |
| `aoi.event.Event.write_shower` | A4, A5, B1 | The file now holds xmax_pos_shc=Xmaxpos and xmax_pos=Xmaxpos+core; analysis_level is no longer forced to 1. The written tree is not attached as event.tshower. | #314, #315 |
| `aoi.event.Event.write_voltages` | A4, B1 | The written tree is no longer attached as event.tvoltage; with auto_file_close=False it is queued for close_files instead. | #315 |
| `aoi.event_list.EventList.__init__` | A3, A4, B1, B2, C1, C3 | Path, DataFile, TFile and DataDirectory inputs now get a filled event_list; el.file is a DataFile. Other types raise TypeError (dev built an unusable list); exit() calls became exceptions. | #317 |
| `aoi.event_list.EventList.__iter__` | A3, A4, B1 | Iteration now starts at start_entry or start_event (dev ignored both and yielded every event); an out-of-range start_entry or unknown start_event raises ValueError. | fix, #213 |
| `aoi.event_list.EventList.get_event` | A3, B2, C1, C3 | Raises instead of printing and returning None for entry_number with event/run numbers, or run_number alone; out-of-range entry_number and unknown event/run pairs raise IndexError/LookupError. | #317 |
| `aoi.event_list.EventList.get_number_of_events` | B1, C1, C5 | Works for lists opened from a file name (dev crashed wrapping the existing DataFile again); no input raises RuntimeError instead of exit(); the no-tree message is a logged warning. | fix |
| `aoi.antenna.Antenna` | A3, B2, C2 | Setting id to a non-integer (e.g. 1.5, '7', an array) raises TypeError, and to a value below -1 raises ValueError; dev stored anything. None is still accepted. | #316 |
| `aoi.event.Event` | A3, B2, C2 | event_number, run_number and tefield_level must be integers >= 0: non-integers raise TypeError, negative values ValueError; dev stored any value. None stays accepted. | #316 |
| `aoi.shower.Shower` | A3, W, B2, C2 | Non-string primary_type, non-numeric energies/Xmax/angles and magnetic_field without 3 values raise; negative energies or Xmax, azimuth outside 0-360 and zenith outside 0-180 warn. | #316 |
| `aoi.timetrace.Timetrace3D` | A3, W, B2, C2 | On Voltage and Efield too: non-integer n_points or du_id raise TypeError, n_points < 0 or du_id < -1 ValueError; non-numeric time_step/t_bin_size raise, negative ones warn. | #316 |
| `sim.shower.gen_shower.ShowerEvent` | A3, W, B2, C2 | primary other than a name or ParticleCode (e.g. an int PDG code), non-numeric energy/angles and origin_geoid without 3 numbers raise; negative energy, out-of-range angles or latitude/longitude warn. | #316 |

## Examples

The scripts in `examples/` are the ones shown in the issues, each with the output of both versions (`.out`). Run them as described in #293.

- `aoi_antennas.py`
- `aoi_array_zenith.py`
- `aoi_efield_ids.py`
- `aoi_fields.py`
- `aoi_iter.py`
- `aoi_raises.py`
- `aoi_write_attach.py`
- `aoi_write_dir.py`
- `aoi_write_shower.py`
- `aoi_xmax.py`
- `aoi_zenith_read.py`
- `basis_changes.py`
- `crash_min.py`
- `crash_reader.py`
- `dio_assign.py`
- `dio_char.py`
- `dio_closed.py`
- `dio_inputs.py`
- `dio_meta.py`
- `dio_numbers.py`
- `dio_write.py`
- `event_generation_dev.py`
- `geo_changes.py`
- `list_file.py`
- `noise_labels.py`
- `root640_grand.py`
- `root640_plain.py`
- `scripts_changes.py`
- `sim_checks.py`
- `sim_numbers.py`
- `xmax_dataio.py`
- `xmax_prepare.py`
