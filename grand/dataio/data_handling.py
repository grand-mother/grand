# Created by Lech Wiktor Piotrowski at 14/03/2025
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import ROOT

from grand.dataio import logger, DataTree, MotherEventTree
import grand.dataio

# ToDo: Ignore the warning about branches (and all the other ROOT errors :( ) for TChain until an answer in the ROOT forum
ROOT.gErrorIgnoreLevel = ROOT.kFatal

## Class holding the information about GRAND data in a directory
class DataDirectory:
    """Class holding the information about GRAND data in a directory"""

    def __init__(self, dir_name: str, recursive: bool = False, analysis_level: int = -1, sim2root_structure: bool = True):
        """
        @param dir_name: the name of the directory to be scanned
        @param recursive: if to scan the directory recursively
        @param analysis_level: which analysis level files to read. -1 means max
        """

        self.analysis_level = analysis_level

        # Make the path absolute
        self.dir_name = os.path.abspath(dir_name)

        # Get the file list
        self.file_list = self.get_list_of_files(recursive)

        # Get the file handle list
        self.file_handle_list = self.get_list_of_files_handles()

        self.tree_file_types = ["ftruns", "ftrunrawvoltages", "ftrunshowersims", "ftrunefieldsims", "ftefields", "ftshowers", "ftshowersims", "ftvoltages", "ftadcs", "ftrawvoltages", "ftrunnoises"]

        self.init_structure()

        # # Set the structure type depending on the dir name
        # exp_structure = False
        # if os.path.basename(dir_name)[:4]=="sim_":
        #     sim2root_structure = True
        # elif os.path.basename(dir_name)[:4]=="exp_":
        #     sim2root_structure = False
        #     exp_structure = True
        #
        # if sim2root_structure:
        #     self.init_sim2root_structure()
        # elif exp_structure:
        #     self.init_exp_structure()
        # else:
        #     logger.warning("Sorry, non exp or sim2root directories are not supported yet")

    def __getattr__(self, name):
        """For non-existing tree files or tree parameters, return None instead of rising an exception"""
        trees_to_check = ["trun", "trunvoltage", "trunrawvoltage", "trawvoltage", "tadc", "tvoltage", "tefield", "tshower", "trunefieldsim", "trunshowersim", "tshowersim", "trunnoise"]
        if any(s in name for s in trees_to_check):
            return None
        else:
            raise AttributeError(f"'DataDirectory' object has no attribute '{name}'")

    def get_list_of_files(self, recursive: bool = False):
        """Gets list of files in the directory"""
        return sorted(glob.glob(os.path.join(self.dir_name, "*.root"), recursive=recursive))

    def get_list_of_files_handles(self):
        """Go through the list of files in the directory and open all of them"""
        file_handle_list = []

        # Function returning the tree type and analysis level from the filename. That's how we want to group files.
        def split_filenames(x):
            el = Path(x).name.split("_")
            if len(el)==4:
                return el[0], el[2]
            else:
                return el[0], el[4]

        # for filename in self.file_list:
        from itertools import groupby
        for key, filenames in groupby(sorted(self.file_list, key=split_filenames), split_filenames):
            filenames = list(filenames)
            file_handle_list.append(DataFile(filenames))

        return file_handle_list

    # Init the instance with sim2root structure files
    def init_structure(self):
        self.file_attrs = []
        # Loop through groups of files with tree types expected in the directory
        for flistname in self.tree_file_types:
            # Assign the list of files with specific tree type to the class instance
            # setattr(self, flistname, {int(Path(el).name.split("_")[-2][1:]): el for el in self.file_handle_list if Path(el.filename).name.startswith(flistname[2:-1]+"_")})
            setattr(self, flistname, {int(Path(el.filename).name.split("_")[-2][1:]): el for el in self.file_handle_list if Path(el.filename).name.startswith(flistname[2:-1]+"_")})

            max_level = -1
            for (l, f) in getattr(self, flistname).items():
                # Assign the file with the tree with the specific analysis level to the class instance
                setattr(self, f"{flistname[:-1]}_l{l}", f)
                self.file_attrs.append(f"{flistname[:-1]}_l{l}")
                # Assign the tree with the specific analysis level to the class instance
                setattr(self, f"{flistname[1:-1]}_l{l}", getattr(f, f"{flistname[1:-1]}_l{l}"))
                if (l>max_level and self.analysis_level==-1) or l==self.analysis_level:
                    max_level = l
                    # Assign the file with the highest or requested analysis level as default to the class instance
                    # ToDo: This may assign all files until it goes to the max level. Probably could be avoided
                    setattr(self, f"{flistname[:-1]}", f)
                    # Assign the tree with the highest or requested analysis level as default to the class instance
                    setattr(self, f"{flistname[1:-1]}", getattr(f, f"{flistname[1:-1]}"))

    # Init the instance with sim2root structure files
    def init_sim2root_structure(self):
        self.file_attrs = []
        # Loop through groups of files with tree types expected in the directory
        for flistname in ["ftruns", "ftrunshowersims", "ftrunefieldsims", "ftefields", "ftshowers", "ftshowersims", "ftvoltages", "ftadcs", "ftrawvoltages", "ftrunnoises"]:
            # Assign the list of files with specific tree type to the class instance
            # setattr(self, flistname, {int(Path(el.filename).name.split("_")[2][1:]): el for el in self.file_handle_list if Path(el.filename).name.startswith(flistname[2:-1]+"_")})
            setattr(self, flistname, {int(Path(el.filename).name.split("_")[-2][1:]): el for el in self.file_handle_list if Path(el.filename).name.startswith(flistname[2:-1]+"_")})
            max_level = -1
            for (l, f) in getattr(self, flistname).items():
                # Assign the file with the tree with the specific analysis level to the class instance
                setattr(self, f"{flistname[:-1]}_l{l}", f)
                self.file_attrs.append(f"{flistname[:-1]}_l{l}")
                # Assign the tree with the specific analysis level to the class instance
                setattr(self, f"{flistname[1:-1]}_l{l}", getattr(f, f"{flistname[1:-1]}_l{l}"))
                if (l>max_level and self.analysis_level==-1) or l==self.analysis_level:
                    max_level = l
                    # Assign the file with the highest or requested analysis level as default to the class instance
                    # ToDo: This may assign all files until it goes to the max level. Probably could be avoided
                    setattr(self, f"{flistname[:-1]}", f)
                    # Assign the tree with the highest or requested analysis level as default to the class instance
                    setattr(self, f"{flistname[1:-1]}", getattr(f, f"{flistname[1:-1]}"))

    # Init the instance with exp (gtot) structure files
    # ToDo: It should be the same as sim2root, but at the moment sim2root has different naming convention
    def init_exp_structure(self):
        self.file_attrs = []
        # Loop through groups of files with tree types expected in the directory
        for flistname in ["ftruns", "ftrunrawvoltages", "ftadcs", "ftrawvoltages"]:
        # for flistname in ["ftruns", "ftrunshowersims", "ftrunefieldsims", "ftefields", "ftshowers", "ftshowersims", "ftvoltages", "ftadcs", "ftrawvoltages", "ftrunnoises"]:
            # Assign the list of files with specific tree type to the class instance
            setattr(self, flistname, {int(Path(el.filename).name.split("_")[-2][1:]): el for el in self.file_handle_list if Path(el.filename).name.startswith(flistname[2:-1]+"_")})
            max_level = -1
            for (l, f) in getattr(self, flistname).items():
                # Assign the file with the tree with the specific analysis level to the class instance
                setattr(self, f"{flistname[:-1]}_l{l}", f)
                self.file_attrs.append(f"{flistname[:-1]}_l{l}")
                # Assign the tree with the specific analysis level to the class instance
                setattr(self, f"{flistname[1:-1]}_l{l}", getattr(f, f"{flistname[1:-1]}_l{l}"))
                if (l>max_level and self.analysis_level==-1) or l==self.analysis_level:
                    max_level = l
                    # Assign the file with the highest or requested analysis level as default to the class instance
                    # ToDo: This may assign all files until it goes to the max level. Probably could be avoided
                    setattr(self, f"{flistname[:-1]}", f)
                    # Assign the tree with the highest or requested analysis level as default to the class instance
                    setattr(self, f"{flistname[1:-1]}", getattr(f, f"{flistname[1:-1]}"))

    def print(self, verbose=True):
        """Prints all the information about all the data"""
        print(f"This DataDirectory instance has:")
        print(f"  {len(self.file_attrs):<3} file/tree attributes")
        print(f"  {len(self.file_list):<3} files")
        print(f"  {len(self.file_handle_list):<3} file chains")

        if verbose:
            print("\n\033[95;40mProperties of each file attribute:\033[0m")
            for attr in self.file_attrs:
                f = getattr(self, attr)
                print(f"\n\n\033[34;40m{attr}:\033[0m\n")
                f.print()

    def get_list_of_chains(self):
        """Gets list of TTree chains of specific type from the directory"""
        pass

    def print_list_of_chains(self):
        """Prints list of TTree chains of specific type from the directory"""
        pass

    def close(self):
        """Close all the files of the directory"""
        for f in self.file_handle_list:
            f.close()

    def get_max_list_of_events(self):
        """Gets the max list of event,run from all the trees"""

        trees_to_check = ["tadc", "trawvoltage", "tvoltage", "tefield", "tshower", "tshowersim"]
        # Assuming, that the lowest level tadc has the max number, and going up if it doesn't exist
        for level in range(10):
            for tree in trees_to_check:
                if tree_inst := getattr(self, f"{tree}_l{level}"):
                    return tree_inst.get_list_of_events()

## Class holding the information about GRAND TTrees in the specified file
class DataFile:
    """Class holding the information about GRAND TTrees in the specified file"""

    def __init__(self, filename):
        """filename can be either a string or a ROOT.TFile"""

        # Need to init here, so that different instances do not share the same data
        ## Holds all the trees in the file, by tree name
        self.dict_of_trees = {}
        """Holds all the trees in the file, by tree name"""

        ## Holds the list of trees in the file, but just with maximal level
        self.list_of_trees = []
        """Holds the list of trees in the file, but just with maximal level"""

        ## List of tree instances
        self.tree_instances = []
        """List of tree instances"""

        ## Holds dict of tree types, each containing a dict of tree names with tree meta-data as values
        self.tree_types = defaultdict(dict)
        """Holds dict of tree types, each containing a dict of tree names with tree meta-data as values"""

        ## Does this instace hold a chain of files
        self.is_tchain = False
        """Does this instace hold a chain of files"""

        ## File list in case this is a chain
        self.flist = []
        """File list in case this is a chain"""

        # If a string given, open the file
        if type(filename) is str:
            # Check if a chain - filename string resolves to a list longer than 1 (due to wildcards)
            flist = glob.glob(filename)
            self.flist = flist
            if len(flist) > 1:
                f = ROOT.TFile(flist[0])
                self.f = f
                self.filename = flist[0]
                self.is_tchain = True
            # Single file
            else:
                f = ROOT.TFile(filename)
                self.f = f
                self.filename = filename
        # If list of files is given, make a TChain
        elif type(filename) is list:

            if len(filename) > 1:
                f = ROOT.TFile(filename[0])
                self.f = f
                self.filename = filename[0]
                self.is_tchain = True
                self.flist = filename
            # Single file
            else:
                f = ROOT.TFile(filename[0])
                self.f = f
                self.filename = filename[0]
                self.flist = filename
        elif type(filename) is ROOT.TFile:
            self.f = filename
            self.filename = self.f.GetName()
        else:
            raise TypeError(f"Unsupported type {type(filename)} as a filename")

        # Loop through the keys
        for key in self.f.GetListOfKeys():
            t = self.f.Get(key.GetName())
            # Process only TTrees
            if type(t) != ROOT.TTree:
                continue

            # Get the basic information about the tree
            tree_info = self.get_tree_info(t)

            # If we want a TChain
            if self.is_tchain:
                t = ROOT.TChain(t.GetName(), t.GetName())
                # Assign files to the chain
                for el in self.flist:
                    t.Add(el)

                # Build the index for this chain - it is not generated automatically from the Trees indices
                try:
                    # Assuming en Event tree
                    t.BuildIndex("run_number", "event_number")
                # If failed, try as a run tree
                except:
                    try:
                        t.BuildIndex("run_number")
                    except:
                        raise("Unable to build index for the tree")

                # Modify the number of events for this TChain
                tree_info["evt_cnt"] = t.GetEntries()

            # Add the tree to a dict for this tree class
            self.tree_types[tree_info["type"]][tree_info["name"]] = tree_info

            self.dict_of_trees[tree_info["name"]] = t

        # Select the highest analysis level trees for each class and store these trees as main attributes
        # Loop through tree types
        # ToDo: make sure that this is for the instance, not the class
        self.max_tree_instance = None
        for key in self.tree_types:
            max_analysis_level = -1
            # Loop through trees in the current tree type
            for key1 in self.tree_types[key].keys():
                el = self.tree_types[key][key1]
                tree_class = getattr(grand.dataio, el["type"])
                tree_instance = tree_class(_tree_name=self.dict_of_trees[el["name"]])
                tree_instance.file = self.f
                self.tree_instances.append(tree_instance)
                # If there is analysis level info in the tree, attribute each level and max level
                if "analysis_level" in el:
                    if el["analysis_level"] > max_analysis_level or el["analysis_level"] == 0:
                        max_analysis_level = el["analysis_level"]
                        max_anal_tree_name = el["name"]
                        max_anal_tree_type = el["type"]
                        self.max_tree_instance = tree_instance
                    setattr(self, tree_class.get_default_tree_name() + "_l" + str(el["analysis_level"]), tree_instance)
                # In case there is no analysis level info in the tree (old trees), just take the last one
                elif max_analysis_level == -1:
                    max_anal_tree_name = el["name"]
                    max_anal_tree_type = el["type"]
                    self.max_tree_instance = tree_instance

                traces_lenghts = self._get_traces_lengths(tree_instance)
                if traces_lenghts is not None:
                    el["traces_lengths"] = traces_lenghts

                dus = self._get_list_of_all_used_dus(tree_instance)
                if dus is not None:
                    el["dus"] = dus

                el["mem_size"], el["disk_size"] = tree_instance.get_tree_size()

            # tree_class = getattr(thismodule, max_anal_tree_type)
            # tree_instance = tree_class(_tree_name=self.dict_of_trees[max_anal_tree_name])
            # tree_instance.file = self.f
            # setattr(self, tree_class.get_default_tree_name(), tree_instance)
            setattr(self, tree_class.get_default_tree_name(), self.max_tree_instance)
            # setattr(self, tree_class.get_default_tree_name(), getattr(self, tree_class.get_default_tree_name() + "_l" + str(max_analysis_level)))
            self.list_of_trees.append(self.dict_of_trees[max_anal_tree_name])

    def __enter__(self):
        """ enter() for DataFile as context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """On exiting DataFile as context manager call close function"""
        self.close()

    def print(self):
        """Prints the information about the TTrees in the file"""

        # If this file is a chain
        if self.is_tchain:
            print("This DataFile is a chain of the following files:")
            print(self.flist)
            print(f"The first file size: {self.f.GetSize():40}")
            print("Most of the information below are based on the tree in the first file")
        else:
            print("This DataFile refers to the following file:")
            print(self.flist)
            print(f"File size: {self.f.GetSize():40}")

        print(f"Tree classes found in the file: {str([el for el in self.tree_types.keys()]):40}")

        for key in self.tree_types:
            print(f"Trees of type {key:<40}: {str([el for el in self.tree_types[key].keys()]):<40}")

        for key in self.tree_types:
            for key1 in self.tree_types[key].keys():
                # print(f"{key1}: {self.tree_types[key][key1]}")
                print(key1)
                for key2 in self.tree_types[key][key1]:
                    print(f"  {key2:<20}: {self.tree_types[key][key1][key2]}")

    def get_tree_info(self, tree):
        """Gets the information about the tree"""

        # If tree is a string, turn it into a TTree
        if type(tree) == str:
            tree = getattr(self, tree)
        # If tree is a GRAND tree class
        if issubclass(tree.__class__, DataTree):
            tree = tree._tree

        tree_info = {"evt_cnt": tree.GetEntries(), "name": tree.GetName()}

        meta_dict = DataTree.get_metadata_as_dict(tree)

        # Deduce the tree type if not in the metadata
        if "type" not in meta_dict.keys():
            tree_info["type"] = self._guess_tree_type(tree)

        tree_info.update(meta_dict)

        return tree_info

    def _get_traces_lengths(self, tree):
        """Adds traces info to event trees"""
        # If tree is not of Event class (contains traces), do nothing
        if not issubclass(tree.__class__, MotherEventTree) or "sim" in tree.tree_name or "zhaires" in tree.tree_name:
            return None
        else:
            traces_lengths = tree.get_traces_lengths()
            if traces_lengths is None:
                return None

            # Check if traces have constant length
            if np.unique(np.array(traces_lengths).ravel()).size != 1:
                logger.warning(f"Traces lengths vary through events or axes for {tree.tree_name}! {traces_lengths}")
                return traces_lengths
            else:
                return traces_lengths[0][0]

    def _get_list_of_all_used_dus(self, tree):
        """Get list of all data units used in the tree"""
        if issubclass(tree.__class__, MotherEventTree):
            return tree.get_list_of_all_used_dus()
        else:
            return None

    def print_tree_info(self, tree):
        """Prints the information about the tree"""
        pass

    def _guess_tree_type(self, tree):
        """Guesses the tree type from its name. Needed for old trees with missing metadata"""
        name = tree.GetName()

        # Sim trees
        if "sim" in name:
            # if "runvoltage" in name:
            #     return "VoltageRunSimdataTree"
            # elif "eventvoltage" in name:
            #     return "VoltageEventSimdataTree"
            if "runefield" in name:
                return "TRunEfieldSim"
            # elif "eventefield" in name:
            #     return "EfieldEventSimdataTree"
            elif "runshower" in name:
                return "TRunShowerSim"
            elif "shower" in name:
                return "TShowerSim"
        # # Zhaires tree
        # elif "eventshowerzhaires" in name:
        #     return "ShowerEventZHAireSTree"
        # Other trees
        else:
            if "run" in name:
                return "TRun"
            elif "adc" in name:
                return "TADC"
            elif "voltage" in name:
                return "TRawVoltage"
            elif "efield" in name:
                return "TEfield"
            elif "shower" in name:
                return "TShower"

    def _load_trees(self):
        # Loop through the keys
        # Process only TTrees
        # Get the basic information about the tree
        # Add the tree to a dict for this tree class
        # Select the highest analysis level trees for each class and store these trees as main attributes
        pass

    def close(self):
        """Close the file and the belonging trees"""
        for t in self.tree_instances:
            t.stop_using()
        self.f.Close()

    def get_max_list_of_events(self):
        """Gets the max list of event,run from all the trees"""

        trees_to_check = ["tadc", "trawvoltage", "tvoltage", "tefield", "tshower", "tshowersim"]
        # Assuming, that the lowest level tadc has the max number, and going up if it doesn't exist
        for level in range(10):
            for tree in trees_to_check:
                try:
                    if tree_inst := getattr(self, f"{tree}_l{level}"):
                            return tree_inst.get_list_of_events()
                except:
                    pass
