#!/usr/bin/python
# Created by Lech Wiktor Piotrowski at 16/05/2025
# Extracts events from provided directories and stores in a target directory

import argparse
from pathlib import Path
import shutil
from collections import defaultdict
import grand.dataio
from grand.dataio import DataDirectory


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Extract events from provided directories and store in a target directory.')

    # Add the command-line options
    parser.add_argument('source_events_list_file', metavar='<source_events_list_file>', type=str, help='A file with a list of source events to extract. The format is dir_path,run_num,event_num')
    parser.add_argument('target_dirname', metavar='<dirname>', type=str, help='The target directory to store the extracted events in')
    parser.add_argument("-c", "--comment", help="Comment to add to the target directory name", default=None)
    parser.add_argument("-ow", "--overwrite", action='store_true', help="Overwrite the target directory", default=False)

    # Parse the arguments
    args = parser.parse_args()

    # Read the list of source events
    with open(args.source_events_list_file, 'r') as f:
        source_events_list = f.readlines()
        if len(source_events_list) == 0:
            print('No events in the source file.')
            exit(1)

    target_dir_path = Path(args.target_dirname)

    # Delete the directory if overwrite requested
    if target_dir_path.is_dir() and args.overwrite:
        shutil.rmtree(target_dir_path)

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

    # Loop through the source events
    for source_event in source_events_list:
        # Extract the dir name, run_number and event_number
        print("Copying event:", source_event)
        copied_event_num += 1
        dp, run_num, event_num = source_event.split(',')
        # Transform the directory path to absolute
        dp = str(Path(dp).resolve())
        run_num = int(run_num)
        event_num = int(event_num)

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
                        create_file_tree(target_dir, source_tree_name, source_tree)

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
                    target_tree.fill()
                    dict_of_trees[target_tree._tree.GetName()] = target_tree
                    print("Found!", source_tree_name, run_num, event_num, target_tree.get_entries())
                else:
                    print("Event/run not found", run_num, event_num)

    print("Copied events count:", copied_event_num)

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

    print("Written events count:", written_event_num)

    print("Done")


# Create the tree and its file
def create_file_tree(target_dir, tree_name, source_tree):

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

    # Attach the tree instance to the DataDirectory
    setattr(target_dir, tree_name, tree_instance)

if __name__ == '__main__':
    main()