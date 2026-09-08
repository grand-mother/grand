# -*- coding: utf-8 -*-
r"""Everything both branch documents know about branches.

``make_branch_diagram.py`` draws the family tree and
``make_branch_inventory.py`` writes the branch-by-branch reference. They used
to be one script and a plan to write the other; keeping the facts here means a
branch renamed, described or judged is renamed, described or judged in both.

Three kinds of data live here, and the difference matters when editing:

**Measured.** ``collect()`` asks git. Dates, authors, commit counts, parentage
and whether a branch is contained in the trunk are never typed by hand, so they
cannot go stale.

**Described.** ``DESCRIPTIONS`` and ``SHORT`` -- two to four words per branch,
and an abbreviation where the real name will not fit a box. A branch with no
entry still appears, with an empty description, so a new one shows up as soon
as it is pushed rather than waiting for somebody to notice.

**Judged.** ``VERDICTS`` -- whether an unmerged branch should be merged, and
why. This is the one part that is an opinion, and it is dated: see
``resources/dev/dev-next/BRANCHES.md`` for when each was last reviewed.
"""
import datetime
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The branch everything is measured against.
TRUNK = "dev-next"

#: Two-to-four words per branch. Git knows the rest.
DESCRIPTIONS = {
    "dev-next": "the new trunk",
    "dev": "the old trunk",
    "master": "old default branch",
    "main": "abandoned 2023 trunk",
    "event-viewer": "event viewer",
    "dev_fix_root_warnings_lwp": "ROOT 6.38 warnings",
    "dev_fix_root_warnings_lwp_new_fields": "NUTRIG name clash",
    "dev_fix_root_warnings_aoi_levels_lwp": "levels, +40% speed",
    "dev_nutrig_fields": "NUTRIG fields in TADC",
    "dev_reprocessing": "Snakemake pipeline",
    "dev_Event_write": "tshower writing",
    "dev_aoi_unittest": "aoi unit tests",
    "dev_snonis": "noise sqrt2 fix",
    "dev_database": "data catalogue",
    "dev_io_root": "ROOT I/O layer",
    "dev_io_root_testmerges": "I/O test merges",
    "dev_sim2root": "sim2root converters",
    "dev_sim2root_merge": "sim2root merge",
    "dev_sim2root_merge__merge_with_dev": "sim2root into dev",
    "dev_sim2root_merge__merge_with_dev_fix_fields": "sim2root field fixes",
    "dev_imports": "import cleanup",
    "ci/docker-test": "docker CI trial",
    "copilot/add-color-coded-diagram": "diagram experiment",
    "radio": "2020 lib/ work",
    "refact_galaxy": "rival galaxy refactor",
    "dev_marion": "reconstruction package",
    "grandio_light": "the package split",
    "dev_downsample_and_ADCconversion_Jelena": "ADC conversion",
    "masterkastner": "docstrings, old docs",
    "beta_dc1": "DC1 analysis scripts",
    "dc2_debug_xmax": "DC2 xmax debugging",
    "dev_leisos": "recursive coreas pipeline",
    "dev_event_viewer": "event viewer examples",
    "snonis_sim2root_test_merge": "galaxy test notebook",
    "147-add-option-to-read-in-non-parallel-coreas-sims-in-sim2root":
        "non-parallel CoREAS",
    "tian-conda-arm": "ARM install notes",
    "no-astropy": "drop astropy",
    "dependabot/pip/binder/pillow-9.3.0": "abandoned auto-PR",
    # Merged and since deleted. Descriptions read off the paths each branch
    # touched, not guessed from the name alone.
    "73_dev_efield2voltage": "efield to voltage (#73)",
    "73_dev_for_merge": "issue 73 merge staging",
    "74_dev_sim2root": "sim2root converters (#74)",
    "debug_dev_io_root": "ROOT I/O debugging",
    "dependabot/pip/binder/pillow-9.0.1": "auto-PR: pillow in binder",
    "dependabot/pip/quality/lxml-4.6.5": "auto-PR: lxml in quality",
    "dependabot/pip/quality/lxml-4.9.1": "auto-PR: lxml in quality",
    "dev_28-factory-event": "docker env (#28)",
    "dev_binder": "Binder notebooks",
    "dev_dc1": "data challenge 1",
    "dev_dc1_EfieldROOT": "DC1 efield in ROOT",
    "dev_dc1_fleg": "DC1 database work",
    "dev_dc1_merge": "DC1 merge staging",
    "dev_docker": "docker environment",
    "dev_fix_dc2": "DC2 figure scripts",
    "dev_Jelena": "CoREAS proton sims",
    "dev_jupyter": "Jupyter examples",
    "dev_noAppim": "drop AppImage",
    "dev_sim1root": "sim2root Common",
    "dev_sim2root_efield2efield": "efield to efield",
    "dev_tool4qual": "quality tooling",
    "dev_update_readme_for_paper": "README for the paper",
    "dev_update_trace_ci32": "trace basis update",
    "dev_vs_code": "VS Code setup",
    "dev_xdu_rf": "Xiaodushan RF chain",
    "grand_coordinates": "coordinate system",
    "infra_conda": "conda infrastructure",
    "merge_grand_coordinates": "coordinates merge staging",
    "multievent_73_dev_efield2voltage": "multi-event efield2voltage",
    "mypy_correction": "mypy typing",
    "update": "docs and tests update",
}

#: Names too long for a box. Display only -- git is always asked about the real
#: name, so an abbreviation here cannot break a lookup, which is a mistake the
#: first version of the diagram made.
SHORT = {
    "dev_fix_root_warnings_lwp_new_fields": "dev_fix_root_..._new_fields",
    "dev_fix_root_warnings_aoi_levels_lwp": "dev_fix_root_..._aoi_levels",
    "dev_sim2root_merge__merge_with_dev": "dev_sim2root_..._with_dev",
    "dev_sim2root_merge__merge_with_dev_fix_fields": "dev_sim2root_..._fix_flds",
    "dev_downsample_and_ADCconversion_Jelena": "dev_downsample_..._Jelena",
    "147-add-option-to-read-in-non-parallel-coreas-sims-in-sim2root":
        "147-non-parallel-coreas",
    "dependabot/pip/binder/pillow-9.3.0": "dependabot/...pillow",
    "copilot/add-color-coded-diagram": "copilot/add-color-coded",
}

#: When the verdicts below were last reviewed, end to end.
VERDICTS_REVIEWED = "2026-09-08"

#: What to do with each branch that still carries patches of its own, and why.
#: ``action`` is one of: ``merge`` (clean, take it), ``merge-hand`` (take it,
#: expect to resolve a conflict), ``cherry-pick`` (take part, never the
#: branch), ``no`` (nothing to take), ``decide`` (a question for the
#: collaboration, not a merge), ``ask`` (needs its author first).
VERDICTS = {
    "dev_event_viewer": ("merge",
        "Four new files under examples/eventviewer/, May 2025. Test-merges "
        "clean and touches nothing else. The 864-line viewer enters lint "
        "scope on merge, so expect to baseline or clean it."),
    "dev_marion": ("merge",
        "The reconstruction package: 38 new files under grand/analysis/. "
        "Test-merges clean. Merging is easy; the open question is whether "
        "reconstruction belongs in GRANDlib at all, which is phase 5."),
    "beta_dc1": ("merge",
        "scripts/ADanalysis.py and TDAnalysis.py, 312 lines, absent from the "
        "trunk. Test-merges clean. From 2023, so confirm DC1 analysis is "
        "still wanted before taking it."),
    "tian-conda-arm": ("absorb",
        "Ten lines of ARM install notes, and **nothing else**: one commit "
        "whose parent is the merge-base itself, touching one file, adding ten "
        "lines and removing none. No code of any kind."
        "\n\n"
        "The content was worth having -- it is the only written record of "
        "getting GRANDlib running natively on ARM -- but not as written. Its "
        "opening sentence, \"ARM cpu is compatible with the grandlib "
        "environment for conda\", contradicts the warning twenty lines above "
        "it in the same file, which says the environment is amd64 only with "
        "arm64 work in progress. Both are describing the same thing from "
        "different ends: the trouble starts at the TURTLE and GULL "
        "compilation, and Tien's step 2 is the workaround for exactly that. "
        "The link also pointed at `master`, which Phase 9 retires."
        "\n\n"
        "So it was rewritten into `docs/source/installation.rst` as a `tip` "
        "admonition under the existing ARM paragraph, which until then said "
        "only that there were known problems and pointed at a readme. "
        "Attributed to Tien with the date, and marked as not re-tested -- this "
        "machine is x86-64, so nobody here can verify it."
        "\n\n"
        "The branch is therefore **absorbed, not merged**. `git cherry` will "
        "report it outstanding for ever, because a paragraph rewritten into "
        "another file has no patch in common with the original."),
    "147-add-option-to-read-in-non-parallel-coreas-sims-in-sim2root": ("no",
        "One commit, one file, one hunk: eight lines wrapping the PARALLEL "
        "read in `try/except` so that a non-parallel CoREAS run, which writes "
        "no PARALLEL card, gets -1 rather than an error."
        "\n\n"
        "**The fix works.** An earlier reading here said it could not, on the "
        "grounds that a missing keyword raised nothing to catch. That was "
        "wrong: `read_list_of_params` held its result in a local named "
        "`list`, and assigning that name anywhere in the body makes it local "
        "throughout, so the not-found path raised `UnboundLocalError` rather "
        "than returning the builtin. The bare `except:` catches it. The wrong "
        "claim came from testing a scratch function that really did return "
        "the builtin, instead of the branch's own code."
        "\n\n"
        "**What was tested, end to end.** Two working trees identical except "
        "for the two converter files, both running the real "
        "`CoreasToRawROOT.py` over the repository's own fixture at "
        "`sim2root/CoREASRawRoot/proton/`, with the `PARALLEL` card deleted "
        "from `SIM004100.inp`. Reading the values back out of the ROOT trees "
        "that each produced:"
        "\n\n"
        "| | output | `parallel_ectcut` | `parallel_ectmax` |\n"
        "|---|---|---|---|\n"
        "| branch | `Coreas_004100.root` | -1.0 | -1.0 |\n"
        "| `dev-next` | `Coreas_004100.rawroot` | -1.0 | -1.0 |"
        "\n\n"
        "Identical, in the written file rather than in a reconstruction of "
        "the code. Both also print the same warning, because the trunk's "
        "wording was taken from this branch."
        "\n\n"
        "Two confounds had to be cleared first, both of which would have given "
        "a wrong answer. `Coreas_004100.rawroot` is **committed in the "
        "repository**, so it was copied into both scratch trees and the first "
        "read picked up the stale file for the branch, reporting the original "
        "PARALLEL values of 1000 and 100000. And the two versions write "
        "different names -- `.root` against `.rawroot` -- so a glob for one "
        "reported the branch as having produced nothing when it had exited 0."
        "\n\n"
        "**Why superseded rather than merged.** The trunk fixed the reader "
        "itself in September 2026, so absence is a value: `PARALLEL` is "
        "handled on an explicit `is None`, and `ECUTS`, `THIN` and `THINH` "
        "raise an error naming the keyword and the file. The branch guards "
        "only `PARALLEL`, leaving the other three raising the original "
        "`UnboundLocalError`, which names neither. Its bare `except:` also "
        "catches every other failure -- a missing or unreadable .inp included "
        "-- and writes -1 for all of them. The two edits are in the same "
        "block, so this is either/or, and the trunk's is the one to keep."
        "\n\n"
        "The branch's contribution was identifying a real problem, and that "
        "is worth crediting when issue #147 is closed against the trunk."),

    "dev_leisos": ("cherry-pick",
        "A real fix in get_antenna_position (an `or` that should be an `and`, "
        "plus guarded assignment) under 35,000 lines of committed CORSIKA "
        "input, three 1.4 MB .rawroot binaries and an Emacs autosave file "
        "named #test#. Take the two .py hunks, never the branch."),
    "masterkastner": ("no",
        "**In plain terms.** Three commits from mid-2024 adding Sphinx "
        "documentation. Of 201 files, nearly all are a rebuilt docs tree plus "
        "committed build artefacts -- generated `html/` and "
        "`doctrees/environment.pickle`. What touches live code is 34 lines, "
        "and they read as though an IDE's documentation assistant wrote them: "
        "one begins \"The above code defines...\" on line 1, with nothing "
        "above it; another pastes the same sentence twice; a third promises "
        "\"Here's a breakdown of what the class does:\" and then stops."
        "\n\n"
        "**Corrects an earlier reading here.** This was previously listed as a "
        "cherry-pick, on the grounds that it carried 34 lines of docstrings "
        "worth taking. The line count is right and the characterisation was "
        "not."
        "\n\n"
        "**In detail**, all verified. Strip docstrings and the AST of all five "
        "`grand/` files is identical to the merge-base: **no executable change "
        "at all**. The 34 insertions are 19 docstring lines and **15 `#` "
        "comments**. Three of the five hunks no longer apply to the trunk "
        "(`du_network.py`, `type_trace.py`, `root_files.py`)."
        "\n\n"
        "The content is wrong or stale where it is not redundant. The "
        "`closest_node` docstring describes it as returning the closest node; "
        "it returns `np.argmin(...)`, an index, and the trunk's numpydoc "
        "already says so. A comment names `Handling3dTracesOfEvent`, renamed "
        "to `Handling3dTraces` -- zero occurrences on the trunk. The "
        "`root_files.py` insert says the factory returns \"either FileEfield "
        "or FileVoltage\"; it also returns `FileAdc`."
        "\n\n"
        "Its `src_outlib/` edits are to the abandoned duplicate tree. The file "
        "there carries literal `<<<<<<<<` conflict markers on the trunk at "
        "lines 17, 23 and 27; the branch comments two of them out and keeps "
        "both sides, turning a SyntaxError into a ModuleNotFoundError. "
        "`widget.py`'s 102-line diff is pure whitespace -- `git diff -w` is "
        "empty."
        "\n\n"
        "**One line of overlap, recorded so nobody later thinks it was "
        "mislabelled.** The branch fixes `Magneitc` to `Magnetic` once in "
        "`examples/geo/geomagnet_tutorial.ipynb`. That typo occurs **eight** "
        "times there, four in sources and four in the matching outputs, and "
        "the branch fixes one -- leaving source and output disagreeing. All "
        "eight were fixed independently on 2026-09-08, consistently. That is "
        "an incidental coincidence, not content taken: the branch is red "
        "rather than purple because none of its purpose survives. Purple is "
        "for a branch whose *purpose* was realised elsewhere."
        "\n\n"
        "Two trunk defects surfaced while checking it, neither to the branch's "
        "credit and both fixed the same day: `grand/basis/type_trace.py` had "
        "no module docstring, and `get_file_event`'s summary omitted the ADC "
        "reader. A third was found in passing -- `Voltage`'s class docstring "
        "claimed a coordinate frame the class does not have, its only fields "
        "being `t` and `V`."),

    "radio": ("decide",
        "**Held open deliberately, 2026-09-08.** The branch itself will not be "
        "merged: 27 files and 7,410 lines from December 2019 under "
        "lib/python/grand/, a layout the project abandoned -- lib/ tracks only "
        "readme.md today -- and most of it is superseded (computevoltage by "
        "grand/sim/efield2voltage.py, AiresInfoFunctions and "
        "CoreasInfoFunctions by sim2root/, and shower, detector, frame and "
        "signal_processing by grand/sim/, grand/geo/ and grand/basis/)."
        "\n\n"
        "What is open is one file. `lib/python/grand/radio/interpolation.py` "
        "(808 lines, Anne Zilles, December 2019) implements `interpolate_trace` "
        "and `do_interpolation`: interpolating an electric-field trace to an "
        "arbitrary antenna position from a star-shape simulation. **There is "
        "no equivalent anywhere on the trunk.** Every interpolation call in "
        "grand/ and sim2root/ -- five of them, in basis/signal.py, "
        "sim/noise/galaxy.py and sim/detector/rf_chain.py -- is one-dimensional "
        "along frequency or time: signal resampling, galactic-noise spectra, "
        "RF-chain response. sim2root knows about star-shapes (`--star_shape`, "
        "a separate run per event) but only to package them into ROOT, never "
        "to interpolate between antennas."
        "\n\n"
        "So deleting this branch would delete the only spatial trace "
        "interpolation the repository has. The question to answer before "
        "closing it is whether that capability is wanted, and if so who ports "
        "it -- Anne Zilles wrote it, and it is star-shape work. Until then the "
        "branch stays amber: this is an open question, not a settled no."),
    "no-astropy": ("no",
        "Two independent reasons, either sufficient. All three files it edits "
        "are gone -- grand/simulation/ was renamed to grand/sim/ in d1ac041, "
        "so the test-merge is three modify/delete conflicts. And its goal has "
        "already been met by another route: astropy appears nowhere on the "
        "trunk, not in pyproject.toml, not in the conda environment, and in "
        "no import under grand/, granddb/ or sim2root/. What is left is "
        "scaffolding for a finished migration -- a grand_astropy flag "
        "threaded through code that no longer exists."),
    "dependabot/pip/binder/pillow-9.3.0": ("no",
        "One line, bumping Pillow in binder/requirements.txt. binder/ was "
        "deleted from the trunk in January 2025 by b9a540b, \"clear not "
        "used\", and survives only on master and main. The test-merge is a "
        "modify/delete conflict, so taking it would resurrect a directory "
        "removed on purpose. It does not answer the security alert either: "
        "that alert is against master, which merging here does not touch."),
    "dev_downsample_and_ADCconversion_Jelena": ("no",
        "**In plain terms.** A day's work from January 2024 on turning "
        "simulated voltages into ADC counts, left unfinished. It does not "
        "run: the code reads settings under names that were never defined, "
        "so it stops with an error on the first event. The conversion "
        "function it adds also fails when called the way its own "
        "documentation describes, and where it does run it gets the physics "
        "wrong in two ways -- it never clips signals that are too loud for "
        "the hardware, and it throws away samples without filtering first, "
        "which turns high frequencies into fake low-frequency signal."
        "\n\n"
        "None of this needs fixing, because somebody already did the job "
        "properly. Five weeks later, in February 2024, Pablo Correa wrote the "
        "finished version: an ADC module and a conversion script that clips "
        "correctly, resamples correctly, and splices in real measured noise "
        "instead of a flat placeholder. That is what is on the trunk today, "
        "with eight tests covering exactly the two things this branch got "
        "wrong. Nothing is lost by letting the branch go."
        "\n\n"
        "**In detail.** Four commits, 2024-01-18, Jelena Petereit, adding "
        "`grand/sim/ADCconverter.py` and wiring it into `efield2voltage.py`."
        "\n\n"
        "*The wiring cannot execute.* `self.params` is initialised with "
        "`add_ADC_conversion` and `add_downsampler_and_paddings`, while the "
        "code reads `add_adc_conversion` and `add_downsampler_and_padding`. "
        "Run against the branch's own dict, both raise `KeyError`, and they "
        "sit in `get_event` (every event) and `compute_voltage_du` (every "
        "antenna). It also calls `self.voltage_to_adc`, which is never "
        "assigned -- the functions are module-level imports -- and then does "
        "`self.vout_f[du_idx] += self.voltage_to_adc`, adding a function "
        "object to a numpy array. The author's own comment reads "
        "`#! TODO: is += correct here?`."
        "\n\n"
        "*The converter is broken independently.* `voltage_to_adc` computes "
        "`downsampling_factor = 1 / adc_sampling_rate`, so its documented "
        "default of 2 gives 0.5 and it raises. A factor of four needs "
        "`adc_sampling_rate=0.25`, inverting the parameter's stated meaning. "
        "Measured: 1e9 uV in gives **9,102,222 counts** out, where a 13-bit "
        "ADC saturates at 8191; and `trace[:, ::4]` with no anti-alias filter "
        "moves a 700 MHz tone to **199 MHz**, inside the band and "
        "indistinguishable from signal. `padding()` uses `padding_value` as "
        "the sample count while its docstring calls it the fill value, which "
        "is hard-coded to 800."
        "\n\n"
        "*What replaced it.* `grand/sim/detector/adc.py` saturates at 8192 "
        "and resamples in the Fourier domain; `scripts/convert_voltage2adc.py` "
        "(38050fe, 2024-02-21, Pablo Correa; fadaaee, 2024-02-27, pcorcam) is "
        "the conversion stage, at `n_samples=2048` with measured noise. Both "
        "verified working here: the script's CLI loads and "
        "`tests/sim/test_adc.py` passes eight tests on saturation and "
        "downsampling."
        "\n\n"
        "*One design note.* The ADC is not wired into `efield2voltage.py` on "
        "the trunk either. A separate pipeline stage was chosen instead, so "
        "the branch's architectural idea was considered and not taken -- it is "
        "not an oversight waiting to be corrected."),

    "dc2_debug_xmax": ("no",
        "**In plain terms.** A debugging session saved to a branch. To chase a "
        "discrepancy between a polarisation simulation and DC2, luckyjim made "
        "the simulation lie in two deliberate ways, and both are still "
        "switched on. Merging it would silently corrupt every simulation run "
        "afterwards. But he was chasing something real, and it is still there "
        "-- see the note at the end, which matters far more than the branch."
        "\n\n"
        "**In detail.** Two commits, luckyjim, December 2024. The hacks, both "
        "live at the tip and both flagged by the author's own warnings:"
        "\n\n"
        "- `for du_idx in range(2)` under `logger.warning('Reduce DU to 2')`, "
        "where the trunk has `range(self.nb_du)`: two antennas per event "
        "instead of all of them.\n"
        "- an `if True:` block replacing each antenna's position with the "
        "shower core position. `if True:` appears zero times on the trunk."
        "\n\n"
        "The rest is what a debug session leaves behind: `logger.debug` to "
        "`logger.info`, a `plt.show()`, `# DEBUG JMC` prints. Of 534 changed "
        "lines in efield2voltage.py, 261 vanish under `git diff -w` and most "
        "of the remainder is black rewrapping one line into several. A "
        "line-by-line read of the whole substantive diff found no bug fix and "
        "no algorithm change hiding in it. The second commit, \"typo\", "
        "fixes a bug *inside* the debug hack -- `shower_core_pos[1]` to `[2]` "
        "-- so only the later version tests what its comment claims."
        "\n\n"
        "`scripts/extract_rf_chain.py` (47 new lines) has no functions, no "
        "CLI and no `__main__` guard: it runs on import, hard-codes "
        "`np.save(\"TF_RF_Chain\")` and ends in `plt.show()`. A scratch file."
        "\n\n"
        "*The one salvage candidate does not survive.* The `-r/--range_idx` "
        "flag converts only events N..M, and the trunk's CLI has no "
        "equivalent -- but `compute_voltage` already accepts lists of "
        "`event_number`/`run_number` (efield2voltage.py:797), so the "
        "capability exists and only the CLI lacks it. The branch implements it "
        "with instance state plus a setter, which is clunkier than passing the "
        "list. Better rebuilt in ten lines than extracted from here."
        "\n\n"
        "**What the branch was actually chasing, and why it matters.** The "
        "`if True:` block puts the antenna at the core so that the direction "
        "to Xmax should reduce to the file's own `azimuth` and `zenith`. That "
        "is a coordinate-consistency probe, and running it against the "
        "repository's own sample data shows it fails: `xmax_pos_shc[2]` "
        "carries an altitude above sea level while x and y are relative to the "
        "core, so Xmax sits **1264 m too high -- exactly `origin_geoid[2]` -- "
        "in all 14 events of every ZHAireS-derived sample**, across zeniths "
        "from 51.6 to 84.2 degrees. Subtract the site altitude and the "
        "geometry becomes self-consistent to a tenth of a metre. The direction "
        "feeding the antenna response is wrong by 1 degree for a distant Xmax "
        "and 7 degrees for a close one. Filed as grand-mother/grand#160; the "
        "branch itself records no conclusion and there is no third commit. It "
        "was cut ten days after issue #106 reported the symptom."),

    "snonis_sim2root_test_merge": ("decide",
        "Not the small fix it looks like. coordinates.py here redefines the "
        "angular convention across all four core transforms -- theta to "
        "180-theta, phi to phi+180, azimuth and elevation redefined -- so "
        "merging moves every angle in the codebase. It also cannot run on "
        "arrays: `if phi==360` raises on the normal call. The other half "
        "(du_type through Efield2Voltage) is already on the trunk. The "
        "convention is a collaboration decision, not a cherry-pick."),
    "grandio_light": ("decide",
        "594 edits to live files and 295,000 deletions: this is the proposal "
        "to split GRANDlib into a light I/O package, not a change to review. "
        "It needs an answer to the split question before it needs a merge."),
    "refact_galaxy": ("ask",
        "A parallel galactic-noise implementation -- it adds galaxy_new.py "
        "beside galaxy.py rather than replacing it -- and touches "
        "efield2voltage.py and signal.py, which conflict. It overlaps the "
        "noise work already verified on the trunk. luckyjim's call."),
}


#: Dispositions that have been *agreed*, with the date. A branch listed here
#: whose verdict is ``no`` will never be merged, and both diagrams colour it
#: apart from the rest so that a reader can see the difference between a
#: question nobody has answered and one that has been closed.
#:
#: A verdict in :data:`VERDICTS` is a recommendation. An entry here is a
#: decision. Only add one when it has actually been taken.
#:
#: Each value is ``(date, one line saying why)``. The one-liner is written out
#: rather than taken from the first sentence of the verdict, which turned out
#: to be a preamble -- "Two independent reasons, either sufficient." is not a
#: reason.
DECIDED = {
    "dependabot/pip/binder/pillow-9.3.0": ("2026-09-08",
        "binder/ was deleted from the trunk in January 2025, so merging would "
        "resurrect it; and the security alert is against master, which this "
        "would not touch"),
    "no-astropy": ("2026-09-08",
        "astropy is already gone from the trunk entirely, and all three files "
        "the branch edits were renamed away in d1ac041"),
    "masterkastner": ("2026-09-08",
        "no executable change, and its 34 live lines are 15 auto-generated "
        "comments plus 19 docstring lines that are wrong on return semantics, "
        "stale on renamed classes, or redundant with the 2026 numpydoc pass; "
        "three of five hunks no longer apply"),
    "dc2_debug_xmax": ("2026-09-08",
        "a debug branch whose two hacks are still live -- two antennas per "
        "event, and every antenna moved to the shower core -- and which "
        "records no conclusion; but the discrepancy it was chasing is real and "
        "still on the trunk, filed as its own issue"),
    "dev_downsample_and_ADCconversion_Jelena": ("2026-09-08",
        "unfinished January 2024 work that cannot run -- KeyError on the main "
        "path -- and whose converter never saturates and aliases; Pablo "
        "Correa's finished ADC module and conversion script landed five weeks "
        "later and is on the trunk"),
    "tian-conda-arm": ("2026-09-08",
        "content taken: rewritten as a tip admonition in "
        "docs/source/installation.rst, attributed and marked untested. The "
        "branch's own patch is not merged and will not be"),
    "147-add-option-to-read-in-non-parallel-coreas-sims-in-sim2root":
        ("2026-09-08",
         "the fix works, but the trunk repaired the reader at the root and "
         "produces the identical stored result -- verified end to end, both "
         "converters writing -1 into parallel_ectcut and parallel_ectmax"),
}


def display_state(name, entry):
    r"""Returns the colour key for a branch.

    Parameters
    ----------
    name : str
        Branch name.
    entry : dict
        Its record from :func:`collect`.

    Returns
    -------
    str
        One of ``trunk``, ``merged``, ``absorbed``, ``retired``, ``unmerged``
        or ``gone``.

    Notes
    -----
    Four of these describe a branch that git still reports as unmerged, and
    they are kept apart because they answer different questions:

    ``unmerged``
        Carries patches of its own, and what to do about it is open.
    ``retired``
        Decided against. It will never be merged.
    ``absorbed``
        Its *content* is in the trunk, rewritten or moved, but its patch is
        not and never will be -- ``git cherry`` compares patch identity, so a
        paragraph rewritten into another file matches nothing. Calling this
        "merged" would claim the patch is contained, and calling it "retired"
        would claim the work was rejected. Both are false, and the question a
        reader actually has is whether the work was lost.
    ``gone``
        Merged and since deleted, so the question does not arise.
    """
    if name == TRUNK:
        return "trunk"
    if not entry.get("live", True):
        return "gone"
    if entry["state"] == "merged":
        return "merged"
    if name in DECIDED:
        action = VERDICTS.get(name, ("", ""))[0]
        if action == "no":
            return "retired"
        if action == "absorb":
            return "absorbed"
    return "unmerged"


def git(*args):
    r"""Returns the stdout of a git command, stripped.

    Runs in the repository root, so these scripts work from any directory.
    """
    return subprocess.run(["git"] + list(args), capture_output=True,
                          text=True, cwd=ROOT).stdout.strip()


def _lines(text):
    r"""Splits git output into non-empty lines."""
    return [line for line in text.split("\n") if line.strip()]


def _is_ancestor(earlier, later):
    r"""True when ``earlier`` is contained in ``later``.

    ``git()`` returns output, and this question is answered by an exit status.
    """
    return subprocess.run(["git", "merge-base", "--is-ancestor", earlier, later],
                          capture_output=True, cwd=ROOT).returncode == 0


def _merges_by_branch():
    r"""Every branch name ever merged into the trunk, and the merges that did it.

    Reads the subject lines of merge commits, which is the only record left of
    a branch that has since been deleted.  A name can appear more than once:
    several of these were merged repeatedly over years.

    Returns
    -------
    dict
        Branch name to a list of merge commit SHAs, oldest first.
    """
    out = {}
    for entry in _lines(git("log", "--merges", "--format=%H%x1f%s", TRUNK)):
        sha, _, subject = entry.partition("\x1f")
        name = ""
        if "Merge pull request #" in subject:
            _, _, tail = subject.partition(" from ")
            name = tail.partition("/")[2].strip()
        elif "Merge branch '" in subject or "Merge remote-tracking branch '" in subject:
            name = subject.partition("'")[2].partition("'")[0]
        for prefix in ("refs/remotes/origin/", "refs/heads/", "origin/"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name:
            out.setdefault(name, []).insert(0, sha)
    return out


def _first_last(shas):
    r"""Earliest and latest (date, author) among a set of commits.

    Parameters
    ----------
    shas : list of str
        Commit SHAs, in any order.

    Returns
    -------
    tuple
        ``(created, creator, last, author)``, dates as ``YYYY-MM-DD``.
        Empty strings when ``shas`` is empty.
    """
    if not shas:
        return "", "", "", ""
    result = subprocess.run(
        ["git", "log", "--no-walk", "--format=%ad%x1f%an", "--date=short",
         "--stdin"],
        input="\n".join(shas), capture_output=True, text=True, cwd=ROOT)
    rows = sorted(line.split("\x1f") for line in _lines(result.stdout))
    if not rows:
        return "", "", "", ""
    return rows[0][0], rows[0][1], rows[-1][0], rows[-1][1]


def collect(include_historical=False):
    r"""Reads every branch from git and works out how they are related.

    Parameters
    ----------
    include_historical : bool, optional
        Also report branches that were merged and have since been deleted,
        recovered from the second parent of the merge that took them.  The
        diagram leaves these out -- it would draw seventy boxes -- and the
        inventory wants them.

    Returns
    -------
    dict
        Branch name to a dict carrying ``last``, ``author``, ``created``,
        ``creator``, ``commits``, ``ahead``, ``state``, ``live``,
        ``merged_on``, ``merged_by``, ``merge_named``, ``parent``,
        ``children`` and ``generation``.

        ``merge_named`` is False when the branch is contained in the trunk but
        no merge commit names it -- merged by fast-forward, squash or rebase.
        For those, ``merged_by`` is empty and ``created``, ``creator`` and
        ``commits`` cannot be recovered: nothing in the history distinguishes
        the branch's commits from the trunk's.
    """
    merges = _merges_by_branch()

    heads = [h for h in git("for-each-ref", "--format=%(refname:short)",
                            "refs/remotes/origin").split()
             if h not in ("origin/HEAD", "origin")]

    info, tip = {}, {}
    for head in heads:
        name = head.replace("origin/", "")
        tip[name] = git("rev-parse", head)
        info[name] = dict(live=True)

    if include_historical:
        for name, shas in merges.items():
            if name in info:
                continue
            second = git("rev-parse", "-q", "--verify", "%s^2" % shas[-1])
            if not second:
                continue
            tip[name] = second
            info[name] = dict(live=False)

    for name, entry in info.items():
        last, author = (git("log", "-1", "--format=%ad|%an", "--date=short",
                            tip[name]).split("|") + ["?"])[:2]
        ahead = sum(1 for line in git("cherry", TRUNK, tip[name]).split("\n")
                    if line.startswith("+"))
        entry["state"] = ("trunk" if name == TRUNK
                          else "unmerged" if ahead else "merged")
        entry["ahead"] = ahead
        entry["last"], entry["author"] = last, author

        # Which merge or merges took this branch. Worked out before the
        # commit count, because it is also the only reliable way to count.
        took = list(merges.get(name, []))
        entry["merged_on"] = entry["merged_by"] = ""
        entry["merge_named"] = False
        if entry["state"] == "merged":
            # For a branch whose ref still exists, the merge that introduced
            # it is the first one on the ancestry path from its tip to the
            # trunk. That is exact. Reading the merge subjects instead would
            # find the last merge that mentions the *name*, which for a branch
            # merged repeatedly over years is a different commit.
            path = [x for x in git("log", "--ancestry-path", "--merges",
                                   "--reverse", "--format=%h|%ad",
                                   "--date=short",
                                   "%s..%s" % (tip[name], TRUNK)).split("\n")
                    if x.strip()] if entry["live"] else []
            # A merge named for the branch only settles the question if it
            # actually contains the branch's tip. dev_sim2root was merged in
            # January 2025 and then committed to again in March, so the merge
            # bearing its name is not the one that brought in what it now
            # holds. Naming it would be a plausible, wrong answer.
            named = [sha for sha in took if _is_ancestor(tip[name], sha)]
            # A merge that *names* the branch is the one that took it, and
            # its second parent is the branch. A merge merely found on the
            # ancestry path is only the first merge that contains the branch,
            # which is a different claim: for ci/docker-test that is the merge
            # of dev_snonis, and counting commits from it would attribute
            # dev_snonis's work to ci/docker-test. So the date is taken from
            # either, and the commits only from a named merge.
            if named:
                entry["merged_by"] = git("log", "-1", "--format=%h", named[0])
                entry["merged_on"] = git("log", "-1", "--format=%ad",
                                         "--date=short", named[0])
                entry["merge_named"] = True
                took = named
            elif path:
                _, entry["merged_on"] = path[0].split("|")
            else:
                # Contained, but nothing to point at: merged by fast-forward,
                # squashed, or rebased, all of which leave no marker at all.
                entry["merged_on"] = "ancestor"

        # Commits belonging to the branch. For one that was merged, that is
        # everything its merges brought in -- a branch merged repeatedly over
        # years is undercounted by looking at the last merge alone.
        own = []
        if entry["state"] == "merged":
            for sha in took:
                if git("rev-parse", "-q", "--verify", "%s^2" % sha):
                    own += _lines(git("rev-list", "%s^1..%s^2" % (sha, sha)))
        if not own:
            # An unmerged branch's own commits are simply what the trunk does
            # not have. Reading them from merges that name the branch would
            # describe an older state of it: beta_dc1 was merged in 2022 and
            # committed to again in 2023, and counting from those merges dated
            # its creation a year before its oldest surviving commit.
            base = git("merge-base", TRUNK, tip[name])
            if base and base != tip[name]:
                own = _lines(git("rev-list", "%s..%s" % (base, tip[name])))
        own = list(dict.fromkeys(own))
        entry["commits"] = len(own)
        created, creator, seen_last, seen_author = _first_last(own)
        entry["created"], entry["creator"] = created, creator
        if seen_last and not entry["live"]:
            # A deleted branch has no ref to ask, so the last commit it
            # contributed is the last one there is. A live branch is asked
            # directly -- overriding it here would report the branch as of its
            # merge rather than as it stands, and the diagram lays out its
            # columns by this date.
            entry["last"] = seen_last
            entry["author"] = seen_author

    by_sha = {sha: name for name, sha in tip.items()}
    for name in info:
        contained = [by_sha[s] for s in set(git("rev-list", tip[name]).split())
                     & set(by_sha) if by_sha[s] != name]
        info[name]["parent"] = (max(contained, key=lambda n: info[n]["last"])
                                if contained else None)

    for name in info:
        info[name]["children"] = sorted(
            other for other in info if info[other]["parent"] == name)

    depth = {}

    def generation(name, guard=0):
        if name in depth:
            return depth[name]
        parent = info[name]["parent"]
        depth[name] = (0 if parent is None or guard > 40
                       else generation(parent, guard + 1) + 1)
        return depth[name]

    for name in info:
        info[name]["generation"] = generation(name)
    return info


def merge_events():
    r"""Every merge in the trunk's history: what went in, and into what.

    Returns
    -------
    list of dict
        One entry per merge commit reachable from the trunk, oldest first,
        with ``source``, ``target``, ``date`` and ``sha``.  ``target`` is None
        when the subject does not name one -- a pull request records only the
        head branch, and a plain ``git merge`` on a checked-out branch records
        neither.  Callers decide what to do with that; the history diagram
        routes it to whichever trunk was current at the time.

    Notes
    -----
    Read from merge *subjects*, which is the only record of a branch that has
    since been deleted, and is therefore as good as the person who wrote them.
    A subject somebody edited by hand is a subject this cannot parse.
    """
    out = []
    for entry in _lines(git("log", "--merges", "--reverse",
                            "--format=%H%x1f%ad%x1f%s", "--date=short", TRUNK)):
        sha, date, subject = (entry.split("\x1f") + ["", ""])[:3]
        source = target = ""
        if "Merge pull request #" in subject:
            source = subject.partition(" from ")[2].partition("/")[2].strip()
        elif "branch '" in subject:
            source = subject.partition("'")[2].partition("'")[0]
            after = subject.partition("'")[2].partition("'")[2]
            if " into " in after:
                target = after.partition(" into ")[2].strip().strip("'\"")
        for prefix in ("refs/remotes/origin/", "refs/heads/", "origin/"):
            if source.startswith(prefix):
                source = source[len(prefix):]
            if target.startswith(prefix):
                target = target[len(prefix):]
        if source:
            out.append(dict(source=source, target=target or None, date=date,
                            sha=git("log", "-1", "--format=%h", sha)))
    return out


def provenance(script):
    r"""Returns ``(stamp, commit)`` describing this generation run.

    Parameters
    ----------
    script : str
        Repository-relative path of the calling generator.

    Returns
    -------
    tuple of str
        A UTC timestamp, and the script's commit with ``(modified since)``
        appended when the working copy differs from it.  A generated file that
        does not say which version of its generator made it cannot be checked.
    """
    commit = git("log", "-1", "--format=%h", "--", script) or "uncommitted"
    if git("status", "--porcelain", "--", script):
        commit += " (modified since)"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    return stamp, commit
