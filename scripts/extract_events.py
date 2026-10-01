#!/usr/bin/python
# Created by Lech Wiktor Piotrowski at 16/05/2025
# Extracts events from provided directories and stores in a target directory

import argparse
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict
import grand.dataio
from grand.dataio import DataDirectory, NotUniqueEvent

#: Names this script writes: <tree type>_..._L<level>_<serial>.root.  With -ow
#: only files of this form are removed from the target (#244).
GRAND_FILE = re.compile(r"^(run|runvoltage|runrawvoltage|runefieldsim|runshowersim|runnoise|"
                        r"efield|voltage|rawvoltage|adc|shower|showersim)_.*_L\d+_\d+\.root$")


def read_event_list(path):
    r"""Reads ``dir_path,run_number,event_number`` lines; exits 2 on a bad line.

    Blank lines and lines starting with ``#`` are skipped, and a repeated line
    is dropped with a warning.  A path containing a comma can be quoted.
    """
    events, seen = [], set()
    with open(path, newline="") as f:
        for number, row in enumerate(csv.reader(f), start=1):
            if not row or not "".join(row).strip() or row[0].lstrip().startswith("#"):
                continue
            if len(row) != 3:
                sys.exit("extract_events.py: %s line %d: expected dir_path,run_number,event_number, "
                         "got %r" % (path, number, ",".join(row)))
            try:
                key = (str(Path(row[0].strip()).resolve()), int(row[1]), int(row[2]))
            except ValueError:
                sys.exit("extract_events.py: %s line %d: run and event numbers must be "
                         "integers, got %r" % (path, number, ",".join(row)))
            if key in seen:
                print("Warning: line %d repeats %s, run %d, event %d; skipped" % ((number,) + key))
                continue
            seen.add(key)
            events.append(key)
    return events


def check_target(target, sources):
    r"""Refuses a target that -ow must not clear: ., .., or a folder holding a source."""
    target = target.resolve()
    if target == Path.cwd().resolve() or target in Path.cwd().resolve().parents:
        sys.exit("extract_events.py: refusing to use %s, the current directory or one of its "
                 "parents, as the target" % target)
    for source in sources:
        source = Path(source).resolve()
        if target == source or target in source.parents:
            sys.exit("extract_events.py: the target %s holds the source folder %s" % (target, source))


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Extract events from provided directories and store in a target directory.')

    # Add the command-line options
    parser.add_argument('source_events_list_file', metavar='<source_events_list_file>', type=str, help='A file with a list of source events to extract. The format is dir_path,run_num,event_num')
    parser.add_argument('target_dirname', metavar='<dirname>', type=str, help='The target directory to store the extracted events in')
    parser.add_argument("-c", "--comment", help="Comment stored in the metadata of every tree written", default=None)
    parser.add_argument("-ow", "--overwrite", action='store_true',
                        help="Replace the GRAND files already in the target directory (other files are kept)",
                        default=False)

    # Parse the arguments
    args = parser.parse_args()

    # Read and check the whole list before touching the target
    source_events_list = read_event_list(args.source_events_list_file)
    if len(source_events_list) == 0:
        print('No events in the source file.')
        sys.exit(1)

    target_dir_path = Path(args.target_dirname)
    check_target(target_dir_path, {dp for dp, _, _ in source_events_list})

    # -ow removes only the GRAND files this script writes: it removed the whole
    # directory, with whatever else was in it -- the current directory for "."
    if target_dir_path.is_dir() and args.overwrite:
        for old in target_dir_path.iterdir():
            if old.is_file() and GRAND_FILE.match(old.name):
                old.unlink()

    # Create the target directory if it doesn't exist
    target_dir_path.mkdir(exist_ok=True)

    # Init the target DataDirectory
    target_dir = DataDirectory(args.target_dirname)

    # Currently opened directory
    cur_dir = None

    # Dict of generated trees
    dict_of_trees = {}

    # List of run numbers
    list_of_runs = defaultdict(set)

    copied_event_num = 0
    already_num = 0
    missing = []
    created = []

    # Loop through the source events
    for dp, run_num, event_num in source_events_list:
        print("Copying event:", dp, run_num, event_num)
        found = False
        present = False

        # Open the source directory if not already opened
        if cur_dir is None or cur_dir.dir_name!=dp:
            if cur_dir is not None:
                cur_dir.close()
            cur_dir = DataDirectory(dp)

        # Loop through all the DataFiles in the current directory (one DataFile can chain multiple ROOT files)
        # for df in cur_dir.file_handle_list:
        for df in cur_dir.file_attrs:
            # Loop through all the trees in the current file (should be 1 in the current scheme, but...)
            source_tree_name = df[1:]
            source_tree = getattr(cur_dir, source_tree_name)
            # for source_tree in df.tree_instances:
            for a in [1]:
                ret = 0
                if "Run" in source_tree.type:
                    ret = source_tree.get_run(run_num)
                # For event trees
                else:
                    ret = source_tree.get_event(event_num, run_num)

                # If the run/event was found
                if ret!=0:
                    # If the tree does not exist in the target directory
                    if not getattr(target_dir, source_tree_name):
                        # Create the tree and its file
                        create_file_tree(target_dir, source_tree_name, source_tree, args.comment)
                        created.append(getattr(target_dir, source_tree_name))

                    # Get the target tree from the target directory
                    target_tree = getattr(target_dir, source_tree_name, source_tree)

                    # If run already exists in the ttree, don't add it
                    # ToDo: Should be modified to change the start/end event/date with new events coming
                    if "Run" in source_tree.type:
                        if target_tree.has_run(run_num) or run_num in list_of_runs[target_tree.tree_name]:
                            continue
                        else:
                            list_of_runs[target_tree.tree_name].add(run_num)

                    # Copy the contents of the source tree current run/event into the target tree
                    target_tree.copy_contents(source_tree)
                    try:
                        target_tree.fill()
                    except NotUniqueEvent:
                        # Already in the target (from another source folder, or
                        # a previous run): skipped, rather than aborting the job
                        # with the target half written (#244)
                        print("Already in the target, skipped:", source_tree_name, run_num, event_num)
                        present = present or "Run" not in source_tree.type
                        continue
                    # Keyed by tree and level: the levels' trees share one name
                    # ("tefield"), so only one of them was written and the other
                    # file was left without a tree (#244)
                    dict_of_trees[source_tree_name] = target_tree
                    if "Run" not in source_tree.type:
                        found = True
                    print("Found!", source_tree_name, run_num, event_num, target_tree.get_entries())
        if found:
            copied_event_num += 1
        elif present:
            already_num += 1
        else:
            print("Event not found:", dp, run_num, event_num)
            missing.append((dp, run_num, event_num))

    print("Events requested:", len(source_events_list), " copied:", copied_event_num,
          " already in the target:", already_num, " not found:", len(missing))

    written_event_num = 0

    # Write all the target trees
    # Loop through all the DataFiles in the target directory
    # Loop through all the trees in the current file
    for key,target_tree in dict_of_trees.items():
        # Build the tree index
        if "Run" in target_tree.type:
            target_tree.build_index("run_number")
        else:
            target_tree.build_index("run_number", "event_number")
        # Write the tree (this also closes the file, and in 1 tree per file scheme it is OK)
        # ToDo: this should be just target_tree.write(), but then I get an error "corrupted double-linked list" at exit
        target_tree._tree.GetCurrentFile().Write()
        written_event_num += 1
        # target_tree.write()

    # target_dir.close()

    # Files created for trees that received nothing would be left without a
    # tree, which DataDirectory cannot read (#244)
    written = {id(tree) for tree in dict_of_trees.values()}
    for tree in created:
        if id(tree) not in written:
            name = tree._file_name
            tree.stop_using()
            Path(name).unlink(missing_ok=True)

    print("Trees written:", written_event_num)

    if missing:
        print("Not found:", ", ".join("%s run %d event %d" % m for m in missing))
        sys.exit(1)
    print("Done")


# Create the tree and its file
def create_file_tree(target_dir, tree_name, source_tree, comment=None):

    # Check if the time string was already generated
    if not hasattr(target_dir, "cur_time_string"):
        # Generate the time string and store it
        from datetime import datetime
        setattr(target_dir, "cur_time_string", datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Generate the file name

    # If run file
    if tree_name[:4]=="trun":
        parts = tree_name.split("_")
        # Replace the run number
        file_name = f"{parts[0][1:]}_00000_{parts[1].upper()}_0000.root"
    else:
        parts = tree_name.split("_")
        # Replace the date and event numbers
        file_name = f"{parts[0][1:]}_{target_dir.cur_time_string}_0-0_{parts[1].upper()}_0000.root"

    # Get the tree class for this tree type
    tree_class = getattr(grand.dataio, source_tree.type)

    # Create the tree instance
    tree_instance = tree_class(_tree_name=source_tree.tree_name, _file_name=target_dir.dir_name+"/"+file_name)

    # Copy/create some metadata
    tree_instance.analysis_level = source_tree.analysis_level
    tree_instance.modification_software = "extract_events.py"
    if comment:
        tree_instance.comment = comment

    # Attach the tree instance to the DataDirectory
    setattr(target_dir, tree_name, tree_instance)

if __name__ == '__main__':
    main()