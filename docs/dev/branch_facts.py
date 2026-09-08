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
    "tian-conda-arm": ("merge-hand",
        "Ten lines of ARM install notes in env/conda/readme.md. Conflicts "
        "only because the trunk added 52 lines to the same file; the content "
        "does not overlap. The readme already mentions ARM twice, so fold "
        "the steps in rather than appending."),
    "147-add-option-to-read-in-non-parallel-coreas-sims-in-sim2root": ("no",
        "The problem is real and the fix works. A non-parallel CoREAS run "
        "writes no PARALLEL card, and the reader raised UnboundLocalError -- "
        "its result was held in a local named `list`, which made the name "
        "local throughout, so the not-found path returned a variable that had "
        "never been assigned. The branch's bare try/except does catch that."
        "\n\n"
        "It is superseded rather than wrong. The trunk fixed the reader itself "
        "in September 2026: absence now returns None, PARALLEL is handled on "
        "an explicit `is None`, and ECUTS, THIN and THINH raise an error "
        "naming the keyword and the file instead of the same unhelpful "
        "UnboundLocalError. The branch's bare `except:` catches every other "
        "failure too -- an unreadable file included -- and writes -1 for all "
        "of them. The two conflict, and the trunk's version is the one to "
        "keep. Close the branch and the issue."),
    "dev_leisos": ("cherry-pick",
        "A real fix in get_antenna_position (an `or` that should be an `and`, "
        "plus guarded assignment) under 35,000 lines of committed CORSIKA "
        "input, three 1.4 MB .rawroot binaries and an Emacs autosave file "
        "named #test#. Take the two .py hunks, never the branch."),
    "masterkastner": ("cherry-pick",
        "Of 201 files, 192 are the old sphinx_docs/ tree, rebuilt since. "
        "What is left is 34 lines of docstrings across five live files under "
        "grand/basis/ and grand/dataio/. Worth taking; the rest is not."),
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
        "Superseded on physics, not on line count. Its voltage_to_adc "
        "decimates with trace[:, ::4] and no anti-alias filter, and never "
        "saturates; grand/sim/detector/adc.py resamples in the Fourier domain "
        "and clips. Its padding() also uses padding_value as the sample count "
        "while documenting it as the fill value."),
    "dc2_debug_xmax": ("no",
        "A debug branch, and it says so: `for du_idx in range(2)` under a "
        "'Reduce DU to 2' warning, and an `if True:` block that puts every "
        "antenna at the shower core. Merging would silently corrupt every "
        "simulation. The half of efield2voltage.py that is not a debug hack "
        "is black reformatting. Only the -r/--range_idx flag and "
        "set_idx_du_range are worth anything, about 20 lines."),
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
        One of ``trunk``, ``merged``, ``retired``, ``unmerged`` or ``gone``.
        ``retired`` is a branch that has been decided against: it still
        carries patches of its own and never will be merged. ``unmerged`` is
        the same situation with the decision still open, which is why the two
        cannot share a colour.
    """
    if name == TRUNK:
        return "trunk"
    if not entry.get("live", True):
        return "gone"
    if entry["state"] == "merged":
        return "merged"
    if name in DECIDED and VERDICTS.get(name, ("", ""))[0] == "no":
        return "retired"
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
