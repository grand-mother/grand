"""!
Classes to handle exchange of information between user and ROOT TTrees holding Zhaires simulation data

@section ROOT for GRAND basic information

This is the interface for accessing GRAND ROOT TTrees that 
(in its final future state) will not require the user 
(reader/writer of the TTrees) to have any knowledge of ROOT. It
 also will hide the internals from the data generator, so that 
 the changes in the format are not concerning the user.

The TTree interface classes are defined in the GRANDROOTTrees.py file.

"""
import ROOT
import numpy as np
import sys
# This import changes in Python 3.10
if sys.version_info.major>=3 and sys.version_info.minor<10:
    from collections import MutableSequence
else:
    from collections.abc import MutableSequence
from dataclasses import dataclass, field
from typing import List, Union

## A list of generated Trees
grand_tree_list = []

## A python list interface to ROOT's std::vector
class StdVectorList(MutableSequence):
    vec_type = ""

    def __init__(self, vec_type, value=[]):
        self._vector = ROOT.vector(vec_type)(value)
        self.vec_type = vec_type

    def __len__(self):
        return self._vector.size()

    def __delitem__(self, index):
        self._vector.erase(index)

    def insert(self, index, value):
        self._vector.insert(index, value)

    def __setitem__(self, index, value):
        self._vector[index] = value

    def __getitem__(self, index):
        # If this is a vector of vectors, convert a subvector to list for the return
        if len(self._vector) > 0:
            if "std.vector" in str(type(self._vector[index])):
                return list(self._vector[index])
            else:
                return self._vector[index]
        else:
            return None

    def append(self, value):
        # exit()
        # std::vector does not want numpy types for push_back, need to use .item()
        if isinstance(value, np.generic):
            self._vector.push_back(value.item())
        else:
            self._vector.push_back(value)

    def clear(self):
        self._vector.clear()

    def __repr__(self):
        if len(self._vector) > 0:
            if "std.vector" in str(type(self._vector[0])):
                return str([list(el) for el in self._vector])

        return str(list(self._vector))

## A python string interface to ROOT's std::string
class StdString():
    def __init__(self, value):
        self.string = ROOT.string(value)

    def __len__(self):
        return len(str(self.string))

    def __repr__(self):
        return str(self.string)


## Mother class for GRAND Tree data classes
@dataclass
class DataTree():
    ## File handle
    _file: ROOT.TFile = None
    ## File name
    _file_name: str = None
    ## Tree object
    _tree: ROOT.TTree = None
    ## Tree name
    _tree_name: str = ""
    ## A list of run_numbers or (run_number, event_number) pairs in the Tree
    _entry_list: list = field(default_factory=list)

    @property
    def tree(self):
        return self._tree

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, val: ROOT.TFile) -> None:
        self._set_file(val)

    @classmethod
    def _type(cls):
        return cls

    def __post_init__(self):
        # Append the instance to the list of generated trees - needed later for adding friends
        grand_tree_list.append(self)
        # Init _file from TFile object
        if self._file is not None:
            self._set_file(self._file)
        # or init _file from a name string
        elif self._file_name is not None and self._file_name!="":
            self._set_file(self._file_name)

        # Init tree from the name string
        if self._tree is None and self._tree_name is not None:
            self._set_tree(self._tree_name)
        # or create the tree
        else:
            self._create_tree()

    ## Set the tree's file
    def _set_file(self, f):
        # If the ROOT TFile is given, just use it
        if isinstance(f, ROOT.TFile):
            self._file = f
            self._file_name = self._file.GetName()
        # If the filename string is given, open/create the ROOT file with this name
        else:
            self._file_name = f
            # print(self._file_name)
            # If the file with that filename is already opened, use it (do not reopen)
            if (f := ROOT.gROOT.GetListOfFiles().FindObject(self._file_name)):
                self._file = f
            # If not opened, open
            else:
                # Update mode both for adding entries and reading
                self._file = ROOT.TFile(self._file_name, "update")

    ## Init/readout the tree from a file
    def _set_tree(self, t):
        # If the ROOT TTree is given, just use it
        if isinstance(t, ROOT.TTree):
            self._tree = t
            self._tree_name = t.GetName()
        # If the tree name string is given, open/create the ROOT TTree with this name
        else:
            self._tree_name = t

            # Try to init with the TTree from file
            if self._file is not None:
                self._tree = self._file.Get(self._tree_name)
                # There was no such tree in the file, so create one
                if self._tree==None:
                    print(f"No valid {self._tree_name} TTree in the file {self._file.GetName()}. Creating a new one.")
                    self._create_tree()
            else:
                print(f"creating tree {self._tree_name} {self._file}")
                self._create_tree()

        # Fill the runs/events numbers from the tree (important if it already existed)
        self.fill_entry_list()

    ## Create the tree
    def _create_tree(self, tree_name=""):
        if tree_name!="": self._tree_name = tree_name
        self._tree = ROOT.TTree(self._tree_name, self._tree_name)

    def fill(self):
        pass

    def write(self, *args, close_file=True, **kwargs):
        # Add the tree friends to this tree
        self.add_proper_friends()

        # If string is ending with ".root" given as a first argument, create the TFile
        # ToDo: Handle TFile if added as the argument
        creating_file = False
        if len(args)>0 and ".root" in args[0][-5:]:
            self._file_name = args[0]
            if (f := ROOT.gROOT.GetListOfFiles().FindObject(self._file_name)):
                self._file = f
            # Overwrite requested
            # ToDo: this does not really seem to work now
            elif kwargs["overwrite"]:
                print("overwriting")
                self._file = ROOT.TFile(args[0], "recreate")
            else:
                # By default append
                self._file = ROOT.TFile(args[0], "update")
            # Make the tree save itself in this file
            self._tree.SetDirectory(self._file)
            # args passed to the TTree::Write() should be the following
            args = args[1:]
            creating_file = True

        # ToDo: For now, I don't know how to do that: Check if the entries in possible tree in the file do not already contain entries from the current tree

        # If the writing options are not explicitly specified, add kWriteDelete option, that deletes the old cycle after writing the new cycle in the TFile
        if len(args)<2:
            args = ["", ROOT.TObject.kWriteDelete]
        self._tree.Write(*args)

        # If TFile was created here, close it
        if creating_file and close_file:
            # Need to set 0 directory so that closing of the file does not delete the internal TTree
            self._tree.SetDirectory(ROOT.nullptr)
            self._file.Close()

    ## Fills the entry list from the tree
    def fill_entry_list(self):
        pass

    ## Check if specified run_number/event_number already exist in the tree
    def is_unique_event(self):
        pass

    ## Add the proper friend trees to this tree (reimplemented in daughter classes)
    def add_proper_friends(self):
        pass

    def scan(self, *args):
        self._tree.Scan(*args)

    def get_entry(self, ev_no):
        res = self._tree.GetEntry(ev_no)
        self.assign_branches()
        return res

    ## All three methods below return the number of entries
    def get_entries(self):
        return self._tree.GetEntries()

    def get_number_of_entries(self):
        return self.get_entries()

    def get_number_of_events(self):
        return self.get_number_of_entries()

    def add_friend(self, value):
        self._tree.AddFriend(value)

    def remove_friend(self, value):
        self._tree.RemoveFriend(value)

    def set_tree_index(self, value):
        self._tree.SetTreeIndex(value)

    ## Create branches of the TTree based on the class fields
    def create_branches(self, set_if_exists=True):
        # Reset all branch addresses just in case
        self._tree.ResetBranchAddresses()

        # If branches already exist, set their address instead of creating, if requested
        set_branches = False
        if set_if_exists and len(self._tree.GetListOfBranches())>0:
            set_branches = True

        # Loop through the class fields
        for field in self.__dataclass_fields__:
            # Skip "tree" and "file" fields, as they are not the part of the stored data
            if field == "_tree" or field == "_file" or field == "_file_name" or field == "_tree_name" or field == "_cur_du_id" or field == "_entry_list": continue
            # Create a branch for the field
            # print(self.__dataclass_fields__[field])
            self.create_branch_from_field(self.__dataclass_fields__[field], set_branches)

    ## Create a specific branch of a TTree computing its type from the corresponding class field
    def create_branch_from_field(self, value, set_branches=False):
        # Handle numpy arrays
        if isinstance(value.default, np.ndarray):
            # Generate ROOT TTree data type string

            # If the value is a (1D) numpy array with more than 1 value, make it an (1D) array in ROOT
            if value.default.size > 1:
                val_type = f"[{value.default.size}]"
            else:
                val_type = ""

            # Data type
            if value.default.dtype == np.int8:
                val_type += "/B"
            elif value.default.dtype == np.uint8:
                val_type += "/b"
            elif value.default.dtype == np.int16:
                val_type += "/S"
            elif value.default.dtype == np.uint16:
                val_type += "/s"
            elif value.default.dtype == np.int32:
                val_type += "/I"
            elif value.default.dtype == np.uint32:
                val_type += "/i"
            elif value.default.dtype == np.int64:
                val_type += "/L"
            elif value.default.dtype == np.uint64:
                val_type += "/l"
            elif value.default.dtype == np.float32:
                val_type += "/F"
            elif value.default.dtype == np.float64:
                val_type += "/D"
            elif value.default.dtype == np.bool_:
                val_type += "/O"

            # Create the branch
            if not set_branches:
                self._tree.Branch(value.name[1:], getattr(self, value.name), value.name[1:] + val_type)
            # Or set its address
            else:
                self._tree.SetBranchAddress(value.name[1:], getattr(self, value.name))
        # ROOT vectors as StdVectorList
        # elif "vector" in str(type(value.default)):
        elif isinstance(value.type, StdVectorList):
            # Create the branch
            if not set_branches:
                self._tree.Branch(value.name[1:], getattr(self, value.name)._vector)
            # Or set its address
            else:
                self._tree.SetBranchAddress(value.name[1:], getattr(self, value.name)._vector)
        # For some reason that I don't get, the isinstance does not work here
        # elif isinstance(value.type, str):
        elif id(value.type)==id(StdString):
            # Create the branch
            if not set_branches:
                self._tree.Branch(value.name[1:], getattr(self, value.name).string)
            # Or set its address
            else:
                self._tree.SetBranchAddress(value.name[1:], getattr(self, value.name).string)
        else:
            print(f"Unsupported type {value.type}. Can't create a branch.")
            exit()

    ## Assign branches to the instance - without calling it, the instance does not show the values read to the TTree
    def assign_branches(self):
        # Assign the TTree branches to the class fields
        for field in self.__dataclass_fields__:
            # Skip "tree" and "file" fields, as they are not the part of the stored data
            if field == "_tree" or field == "_file" or field == "_file_name" or field == "_tree_name" or field == "_cur_du_id" or field == "_entry_list": continue
            # print(field, self.__dataclass_fields__[field])
            # Read the TTree branch
            u = getattr(self._tree, field[1:])
            # print(field[1:], self.__dataclass_fields__[field].name, u, type(u))
            # Assign the TTree branch value to the class field
            setattr(self, field[1:], u)

    ## Get entry with indices
    def get_entry_with_index(self, run_no=0, evt_no=0):
        res = self._tree.GetEntryWithIndex(run_no, evt_no)
        if res==0 or res==-1:
            print(f"No event with event number {evt_no} and run number {run_no} in the tree. Please provide proper numbers.")
            return 0

        self.assign_branches()
        return res

    ## All three methods below return the number of entries
    def print(self):
        return self._tree.Print()

## A mother class for classes with Run values
@dataclass
class MotherRunTree(DataTree):
    _run_number: np.ndarray = np.zeros(1, np.uint32)

    @property
    def run_number(self):
        return self._run_number[0]

    @run_number.setter
    def run_number(self, val: np.uint32) -> None:
        self._run_number[0] = val

    def fill(self):
        # If the current run_number and event_number already exist, raise an exception
        if not self.is_unique_event():
            raise NotUniqueEvent(f"A run with run_number={self.run_number} already exists in the TTree.")

        # Fill the tree
        self._tree.Fill()

        # Add the current run_number and event_number to the entry_list
        self._entry_list.append(self.run_number)

    def add_proper_friends(self):
        # Create the indices
        self._tree.BuildIndex("run_number")

    ## List runs in the tree
    def print_list_of_runs(self):
        count = self._tree.Draw("run_number", "", "goff")
        runs = self._tree.GetV1()
        print("List of runs in the tree:")
        for i in range(count):
            print(int(runs[i]))

    ## Gets list of runs in the tree together
    def get_list_of_runs(self):
        count = self._tree.Draw("run_number", "", "goff")
        runs = self._tree.GetV1()
        return [int(runs[i]) for i in range(count)]

    # Readout the TTree entry corresponding to the run
    def get_run(self, run_no):
        # Try to get the run from the tree
        res = self._tree.GetEntryWithIndex(run_no)
        # If no such entry, return
        if res==0 or res==-1:
            print(f"No run with run number {run_no}. Please provide a proper number.")
            return 0

        self.assign_branches()

        return res

    def build_index(self, run_id):
        self._tree.BuildIndex(run_id)

    ## Fills the entry list from the tree
    def fill_entry_list(self):
        # Fill the entry list if there are some entries in the tree
        if (count := self._tree.Draw("run_number", "", "goff")) > 0:
            v1 = np.array(np.frombuffer(self._tree.GetV1(), dtype=np.float64, count=count))
            self._entry_list = [int(el) for el in v1]

    ## Check if specified run_number/event_number already exist in the tree
    def is_unique_event(self):
        # If the entry list does not exist, the event is unique
        if self._entry_list and self.run_number in self._entry_list:
            return False

        return True


## A mother class for classes with Event values
@dataclass
class MotherEventTree(DataTree):
    _run_number: np.ndarray = np.zeros(1, np.uint32)
    # ToDo: it seems instances propagate this number among them without setting (but not the run number!). I should find why...
    _event_number: np.ndarray = np.zeros(1, np.uint32)

    @property
    def run_number(self):
        return self._run_number[0]

    @run_number.setter
    def run_number(self, val: np.uint32) -> None:
        self._run_number[0] = val

    @property
    def event_number(self):
        return self._event_number[0]

    @event_number.setter
    def event_number(self, val: np.uint32) -> None:
        self._event_number[0] = val

    def fill(self):
        # If the current run_number and event_number already exist, raise an exception
        if not self.is_unique_event():
            raise NotUniqueEvent(f"An event with (run_number,event_number)=({self.run_number},{self.event_number}) already exists in the TTree {self._tree.GetName()}.")

        # Fill the tree
        self._tree.Fill()

        # Add the current run_number and event_number to the entry_list
        self._entry_list.append((self.run_number, self.event_number))


    def add_proper_friends(self):
        # Create the indices
        self.build_index("run_number", "event_number")

        # Add the Run tree as a friend if exists already
        loc_vars = dict(locals())
        run_trees = []
        for inst in grand_tree_list:
            if type(inst) is RunTree: run_trees.append(inst)
        # If any Run tree was found
        if len(run_trees)>0:
            # Warning if there is more than 1 RunTree in memory
            if len(run_trees) > 1:
                print(f"More than 1 RunTree detected in memory. Adding the last one {run_trees[-1]} as a friend")
            # Add the last one RunTree as a friend
            run_tree = run_trees[-1]

            # Add the Run TTree as a friend
            self.add_friend(run_tree.tree)

        # Do not add ADCEventTree as a friend to itself
        if not isinstance(self, ADCEventTree):
            # Add the ADC tree as a friend if exists already
            adc_trees = []
            for inst in grand_tree_list:
                if type(inst) is ADCEventTree: adc_trees.append(inst)
            # If any ADC tree was found
            if len(adc_trees)>0:
                # Warning if there is more than 1 ADCEventTree in memory
                if len(adc_trees) > 1:
                    print(f"More than 1 ADCEventTree detected in memory. Adding the last one {adc_trees[-1]} as a friend")
                # Add the last one ADCEventTree as a friend
                adc_tree = adc_trees[-1]

                # Add the ADC TTree as a friend
                self.add_friend(adc_tree.tree)

        # Do not add VoltageEventTree as a friend to itself
        if not isinstance(self, VoltageEventTree):
            # Add the Voltage tree as a friend if exists already
            voltage_trees = []
            for inst in grand_tree_list:
                if type(inst) is VoltageEventTree: voltage_trees.append(inst)
            # If any ADC tree was found
            if len(voltage_trees) > 0:
                # Warning if there is more than 1 VoltageEventTree in memory
                if len(voltage_trees) > 1:
                    print(f"More than 1 VoltageEventTree detected in memory. Adding the last one {voltage_trees[-1]} as a friend")
                # Add the last one VoltageEventTree as a friend
                voltage_tree = voltage_trees[-1]

                # Add the Voltage TTree as a friend
                self.add_friend(voltage_tree.tree)

        # Do not add EfieldEventTree as a friend to itself
        if not isinstance(self, EfieldEventTree):
            # Add the Efield tree as a friend if exists already
            efield_trees = []
            for inst in grand_tree_list:
                if type(inst) is EfieldEventTree: efield_trees.append(inst)
            # If any ADC tree was found
            if len(efield_trees) > 0:
                # Warning if there is more than 1 EfieldEventTree in memory
                if len(efield_trees) > 1:
                    print(f"More than 1 EfieldEventTree detected in memory. Adding the last one {efield_trees[-1]} as a friend")
                # Add the last one EfieldEventTree as a friend
                efield_tree = efield_trees[-1]

                # Add the Efield TTree as a friend
                self.add_friend(efield_tree.tree)

        # Do not add ShowerEventTree as a friend to itself
        if not isinstance(self, ShowerEventTree):
            # Add the Shower tree as a friend if exists already
            shower_trees = []
            for inst in grand_tree_list:
                if type(inst) is ShowerEventTree: shower_trees.append(inst)
            # If any ADC tree was found
            if len(shower_trees) > 0:
                # Warning if there is more than 1 ShowerEventTree in memory
                if len(shower_trees) > 1:
                    print(f"More than 1 ShowerEventTree detected in memory. Adding the last one {shower_trees[-1]} as a friend")
                # Add the last one ShowerEventTree as a friend
                shower_tree = shower_trees[-1]

                # Add the Shower TTree as a friend
                self.add_friend(shower_tree.tree)


    ## List events in the tree together with runs
    def print_list_of_events(self):
        count = self._tree.Draw("event_number:run_number", "", "goff")
        events = self._tree.GetV1()
        runs = self._tree.GetV2()
        print("List of events in the tree:")
        print("event_number run_number")
        for i in range(count):
            print(int(events[i]), int(runs[i]))

    ## Gets list of events in the tree together with runs
    def get_list_of_events(self):
        count = self._tree.Draw("event_number:run_number", "", "goff")
        events = self._tree.GetV1()
        runs = self._tree.GetV2()
        return [(int(events[i]), int(runs[i])) for i in range(count)]

    ## Readout the TTree entry corresponding to the event and run
    def get_event(self, ev_no, run_no=0):
        # Try to get the requested entry
        res = self._tree.GetEntryWithIndex(run_no, ev_no)
        # If no such entry, return
        if res==0 or res==-1:
            print(f"No event with event number {ev_no} and run number {run_no} in the tree. Please provide proper numbers.")
            return 0

        self.assign_branches()

        return res

    ## Builds index based on run_id and evt_id for the TTree
    def build_index(self, run_id, evt_id):
        self._tree.BuildIndex(run_id, evt_id)

    ## Fills the entry list from the tree
    def fill_entry_list(self, tree=None):
        if tree is None:
            tree = self._tree
        # Fill the entry list if there are some entries in the tree
        if (count := tree.Draw("run_number:event_number", "", "goff")) > 0:
            v1 = np.array(np.frombuffer(tree.GetV1(), dtype=np.float64, count=count))
            v2 = np.array(np.frombuffer(tree.GetV2(), dtype=np.float64, count=count))
            self._entry_list = [(int(el[0]),int(el[1])) for el in zip(v1, v2)]

    ## Check if specified run_number/event_number already exist in the tree
    def is_unique_event(self):
        # If the entry list does not exist, the event is unique
        if self._entry_list and (self.run_number, self.event_number) in self._entry_list:
            return False

        return True

## A class wrapping around a TTree holding values commong for the whole run
@dataclass
class RunTree(MotherRunTree):
    _tree_name: str = "trun"

    ## Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment
    _run_mode: np.ndarray = np.zeros(1, np.uint32)
    ## Run's first event
    _first_event: np.ndarray = np.zeros(1, np.uint32)
    ## First event time
    _first_event_time: np.ndarray = np.zeros(1, np.uint32)
    ## Run's last event
    _last_event: np.ndarray = np.zeros(1, np.uint32)
    ## Last event time
    _last_event_time: np.ndarray = np.zeros(1, np.uint32)

    # These are not from the hardware
    ## Data source: detector, simulation, other
    _data_source: StdString = StdString("detector")
    ## Data generator: gtot (in this case)
    _data_generator: StdString = StdString("GRANDlib")
    ## Generator version: gtot version (in this case)
    _data_generator_version: StdString = StdString("0.1.0")
    ## Site name
    # _site: StdVectorList("string") = StdVectorList("string")
    _site: StdString = StdString("")
    ## Site longitude
    _site_long: np.ndarray = np.zeros(1, np.float32)
    ## Site latitude
    _site_lat: np.ndarray = np.zeros(1, np.float32)
    ## Origin of the coordinate system used for the array
    _origin_geoid: np.ndarray = np.zeros(3, np.float32)

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def run_mode(self):
        return self._run_mode[0]

    @run_mode.setter
    def run_mode(self, value: np.uint32) -> None:
        self._run_mode[0] = value

    @property
    def first_event(self):
        return self._first_event[0]

    @first_event.setter
    def first_event(self, value: np.uint32) -> None:
        self._first_event[0] = value

    @property
    def first_event_time(self):
        return self._first_event_time[0]

    @first_event_time.setter
    def first_event_time(self, value: np.uint32) -> None:
        self._first_event_time[0] = value

    @property
    def last_event(self):
        return self._last_event[0]

    @last_event.setter
    def last_event(self, value: np.uint32) -> None:
        self._last_event[0] = value

    @property
    def last_event_time(self):
        return self._last_event_time[0]

    @last_event_time.setter
    def last_event_time(self, value: np.uint32) -> None:
        self._last_event_time[0] = value

    @property
    def data_source(self):
        return str(self._data_source)

    @data_source.setter
    def data_source(self, value) -> None:
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._data_source.string.assign(value)

    @property
    def data_generator(self):
        return str(self._data_generator)

    @data_generator.setter
    def data_generator(self, value) -> None:
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._data_generator.string.assign(value)

    @property
    def data_generator_version(self):
        return str(self._data_generator_version)

    @data_generator_version.setter
    def data_generator_version(self, value) -> None:
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._data_generator_version.string.assign(value)

    @property
    def site(self):
        return str(self._site)

    @site.setter
    def site(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._site.string.assign(value)


    @property
    def site_long(self):
        return np.array(self._site_long)

    @site_long.setter
    def site_long(self, value):
        self._site_long = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("site_long", self._site_long)

    @property
    def site_lat(self):
        return np.array(self._site_lat)

    @site_lat.setter
    def site_lat(self, value):
        self._site_lat = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("site_lat", self._site_lat)

    @property
    def origin_geoid(self):
        return np.array(self._origin_geoid)

    @origin_geoid.setter
    def origin_geoid(self, value):
        self._origin_geoid = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("origin_geoid", self._origin_geoid)


@dataclass
class ShowerRunSimdataTree(MotherRunTree):
    _tree_name: str = "trunsimdata"

    _shower_sim: StdString = StdString("") # simulation program (and version) used to simulate the shower
    _rel_thin: np.ndarray = np.zeros(1, np.float32)  # relative thinning energy
    _weight_factor: np.ndarray = np.zeros(1, np.float32)  # weight factor
    _lowe_cut_e: np.ndarray = np.zeros(1, np.float32)  # low energy cut for electrons(GeV)
    _lowe_cut_gamma: np.ndarray = np.zeros(1, np.float32)  # low energy cut for gammas(GeV)
    _lowe_cut_mu: np.ndarray = np.zeros(1, np.float32)  # low energy cut for muons(GeV)
    _lowe_cut_meson: np.ndarray = np.zeros(1, np.float32)  # low energy cut for mesons(GeV)
    _lowe_cut_nucleon: np.ndarray = np.zeros(1, np.float32)  # low energy cut for nuceleons(GeV)

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def shower_sim(self):
        return str(self._shower_sim)

    @shower_sim.setter
    def shower_sim(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._shower_sim.string.assign(value)

    @property
    def rel_thin(self):
        return self._rel_thin[0]

    @rel_thin.setter
    def rel_thin(self, value):
        self._rel_thin[0] = value

    @property
    def weight_factor(self):
        return self._weight_factor[0]

    @weight_factor.setter
    def weight_factor(self, value):
        self._weight_factor[0] = value

    @property
    def lowe_cut_e(self):
        return self._lowe_cut_e[0]

    @lowe_cut_e.setter
    def lowe_cut_e(self, value):
        self._lowe_cut_e[0] = value

    @property
    def lowe_cut_gamma(self):
        return self._lowe_cut_gamma[0]

    @lowe_cut_gamma.setter
    def lowe_cut_gamma(self, value):
        self._lowe_cut_gamma[0] = value

    @property
    def lowe_cut_mu(self):
        return self._lowe_cut_mu[0]

    @lowe_cut_mu.setter
    def lowe_cut_mu(self, value):
        self._lowe_cut_mu[0] = value

    @property
    def lowe_cut_meson(self):
        return self._lowe_cut_meson[0]

    @lowe_cut_meson.setter
    def lowe_cut_meson(self, value):
        self._lowe_cut_meson[0] = value

    @property
    def lowe_cut_nucleon(self):
        return self._lowe_cut_nucleon[0]

    @lowe_cut_nucleon.setter
    def lowe_cut_nucleon(self, value):
        self._lowe_cut_nucleon[0] = value

@dataclass
class ShowerEventTree(MotherEventTree):
    _tree_name: str = "teventshower"

    _shower_type: StdString = StdString("")  # shower primary type: If single particle, particle type. If not...tau decay,etc. TODO: Standarize
    _shower_energy: np.ndarray = np.zeros(1, np.float32)  # shower energy (GeV)  Check unit conventions.
    _shower_azimuth: np.ndarray = np.zeros(1, np.float32)  # shower azimuth TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth. Also, geoid vs sphere problem
    _shower_zenith: np.ndarray = np.zeros(1, np.float32)  # shower zenith  TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth
    _shower_core_pos: np.ndarray = np.zeros(4, np.float32)  # shower core position TODO: Coordinates in geoid?. Undefined for neutrinos.
    _atmos_model: StdString = StdString("")  # Atmospheric model name TODO:standarize
    _atmos_model_param: np.ndarray = np.zeros(3, np.float32)  # Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters
    _magnetic_field: np.ndarray = np.zeros(3, np.float32)  # Magnetic field parameters: Inclination, Declination, modulus. TODO: Standarize. Check units. Think about coordinates. Shower coordinates make sense.
    _date: StdString = StdString("")  # Event Date and time. TODO:standarize (date format, time format)
    _ground_alt: np.ndarray = np.zeros(1, np.float32)  # Ground Altitude (m)
    _xmax_grams: np.ndarray = np.zeros(1, np.float32)  # shower Xmax depth  (g/cm2 along the shower axis)
    _xmax_pos_shc: np.ndarray = np.zeros(3, np.float64)  # shower Xmax position in shower coordinates
    _xmax_alt: np.ndarray = np.zeros(1, np.float64)  # altitude of Xmax  (m, in the shower simulation earth. Its important for the index of refraction )
    _gh_fit_param: np.ndarray = np.zeros(3, np.float32)  # X0,Xmax,Lambda (g/cm2) (3 parameter GH function fit to the longitudinal development of all particles)
    _core_time: np.ndarray = np.zeros(1, np.float64)  # ToDo: Check; time when the shower was at the core position - defined in Charles, but not in Zhaires/Coreas?

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def shower_type(self):
        return str(self._shower_type)

    @shower_type.setter
    def shower_type(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._shower_type.string.assign(value)

    @property
    def shower_energy(self):
        return self._shower_energy[0]

    @shower_energy.setter
    def shower_energy(self, value):
        self._shower_energy[0] = value

    @property
    def shower_azimuth(self):
        return self._shower_azimuth[0]

    @shower_azimuth.setter
    def shower_azimuth(self, value):
        self._shower_azimuth[0] = value

    @property
    def shower_zenith(self):
        return self._shower_zenith[0]

    @shower_zenith.setter
    def shower_zenith(self, value):
        self._shower_zenith[0] = value

    @property
    def shower_core_pos(self):
        return np.array(self._shower_core_pos)

    @shower_core_pos.setter
    def shower_core_pos(self, value):
        self._shower_core_pos = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("shower_core_pos", self._shower_core_pos)

    @property
    def atmos_model(self):
        return str(self._atmos_model)

    @atmos_model.setter
    def atmos_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._atmos_model.string.assign(value)

    @property
    def atmos_model_param(self):
        return np.array(self._atmos_model_param)

    @atmos_model_param.setter
    def atmos_model_param(self, value):
        self._atmos_model_param = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("atmos_model_param", self._atmos_model_param)

    @property
    def magnetic_field(self):
        return np.array(self._magnetic_field)

    @magnetic_field.setter
    def magnetic_field(self, value):
        self._magnetic_field = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("magnetic_field", self._magnetic_field)

    @property
    def date(self):
        return str(self._date)

    @date.setter
    def date(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._date.string.assign(value)

    @property
    def ground_alt(self):
        return self._ground_alt[0]

    @ground_alt.setter
    def ground_alt(self, value):
        self._ground_alt[0] = value

    @property
    def xmax_grams(self):
        return self._xmax_grams[0]

    @xmax_grams.setter
    def xmax_grams(self, value):
        self._xmax_grams[0] = value

    @property
    def xmax_pos_shc(self):
        return np.array(self._xmax_pos_shc)

    @xmax_pos_shc.setter
    def xmax_pos_shc(self, value):
        self._xmax_pos_shc = np.array(value).astype(np.float64)
        self._tree.SetBranchAddress("xmax_pos_shc", self._xmax_pos_shc)

    @property
    def xmax_alt(self):
        return self._xmax_alt[0]

    @xmax_alt.setter
    def xmax_alt(self, value):
        self._xmax_alt[0] = value

    @property
    def gh_fit_param(self):
        return np.array(self._gh_fit_param)

    @gh_fit_param.setter
    def gh_fit_param(self, value):
        self._gh_fit_param = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("gh_fit_param", self._gh_fit_param)

    @property
    def core_time(self):
        return self._core_time[0]

    @core_time.setter
    def core_time(self, value):
        self._core_time[0] = value


@dataclass
class ShowerEventSimdataTree(MotherEventTree):
    _tree_name: str = "teventshowersimdata"

    ## Event name
    _event_name: StdString = StdString("")

    ## Event Date and time. TODO:standarize (date format, time format)
    _date: StdString = StdString("")
    ## Random seed
    _rnd_seed: np.ndarray = np.zeros(1, np.float64)
    ## Energy in neutrinos generated in the shower (GeV). Usefull for invisible energy
    _energy_in_neutrinos: np.ndarray = np.zeros(1, np.float32)
    # _prim_energy: np.ndarray = np.zeros(1, np.float32)  # primary energy (GeV) TODO: Support multiple primaries. Check unit conventions. # LWP: Multiple primaries? I guess, variable count. Thus variable size array or a std::vector
    ## Primary energy (GeV) TODO: Check unit conventions. # LWP: Multiple primaries? I guess, variable count. Thus variable size array or a std::vector
    _prim_energy: StdVectorList("float") = StdVectorList("float")
    ## Shower azimuth TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth. Also, geoid vs sphere problem
    _shower_azimuth: np.ndarray = np.zeros(1, np.float32)
    ## Shower zenith  TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth
    _shower_zenith: np.ndarray = np.zeros(1, np.float32)
    # _prim_type: StdVectorList("string") = StdVectorList("string")  # primary particle type TODO: Support multiple primaries. standarize (PDG?)
    ## Primary particle type TODO: standarize (PDG?)
    _prim_type: StdVectorList("string") = StdVectorList("string")
    # _prim_injpoint_shc: np.ndarray = np.zeros(4, np.float32)  # primary injection point in Shower coordinates TODO: Support multiple primaries
    ## Primary injection point in Shower coordinates
    _prim_injpoint_shc: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    # _prim_inj_alt_shc: np.ndarray = np.zeros(1, np.float32)  # primary injection altitude in Shower Coordinates TODO: Support multiple primaries
    ## Primary injection altitude in Shower Coordinates
    _prim_inj_alt_shc: StdVectorList("float") = StdVectorList("float")
    # _prim_inj_dir_shc: np.ndarray = np.zeros(3, np.float32)  # primary injection direction in Shower Coordinates  TODO: Support multiple primaries
    ## primary injection direction in Shower Coordinates
    _prim_inj_dir_shc: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## Atmospheric model name TODO:standarize
    _atmos_model: StdString = StdString("")
    # Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters
    _atmos_model_param: np.ndarray = np.zeros(3, np.float32)
    # Magnetic field parameters: Inclination, Declination, modulus. TODO: Standarize. Check units. Think about coordinates. Shower coordinates make sense.
    _magnetic_field: np.ndarray = np.zeros(3, np.float32)
    ## Shower Xmax depth  (g/cm2 along the shower axis)
    _xmax_grams: np.ndarray = np.zeros(1, np.float32)
    ## Shower Xmax position in shower coordinates
    _xmax_pos_shc: np.ndarray = np.zeros(3, np.float64)
    ## Distance of Xmax  [m]
    _xmax_distance: np.ndarray = np.zeros(1, np.float64)
    ## Altitude of Xmax  (m, in the shower simulation earth. Its important for the index of refraction )
    _xmax_alt: np.ndarray = np.zeros(1, np.float64)
    _hadronic_model: StdString = StdString("")  # high energy hadronic model (and version) used TODO: standarize
    _low_energy_model: StdString = StdString("")  # high energy model (and version) used TODO: standarize
    _cpu_time: np.ndarray = np.zeros(3, np.float32)  # Time it took for the simulation. In the case shower and radio are simulated together, use TotalTime/(nant-1) as an approximation

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def event_name(self):
        return str(self._event_name)

    @event_name.setter
    def event_name(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for event_name {type(value)}. Either a string or a ROOT.std.string is required.")

        self._event_name.string.assign(value)

    @property
    def date(self):
        return str(self._date)

    @date.setter
    def date(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for date {type(value)}. Either a string or a ROOT.std.string is required.")

        self._date.string.assign(value)

    @property
    def rnd_seed(self):
        return self._rnd_seed[0]

    @rnd_seed.setter
    def rnd_seed(self, value):
        self._rnd_seed[0] = value

    @property
    def energy_in_neutrinos(self):
        return self._energy_in_neutrinos[0]

    @energy_in_neutrinos.setter
    def energy_in_neutrinos(self, value):
        self._energy_in_neutrinos[0] = value

    @property
    def prim_energy(self):
        return self._prim_energy

    @prim_energy.setter
    def prim_energy(self, value):
        # A list of strings was given
        if isinstance(value, list):
            # Clear the vector before setting
            self._prim_energy.clear()
            self._prim_energy += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._prim_energy._vector = value
        else:
            exit(f"Incorrect type for prim_energy {type(value)}. Either a list or a ROOT.vector of floats required.")

    @property
    def shower_azimuth(self):
        return self._shower_azimuth[0]

    @shower_azimuth.setter
    def shower_azimuth(self, value):
        self._shower_azimuth[0] = value

    @property
    def shower_zenith(self):
        return self._shower_zenith[0]

    @shower_zenith.setter
    def shower_zenith(self, value):
        self._shower_zenith[0] = value

    @property
    def prim_type(self):
        return self._prim_type

    @prim_type.setter
    def prim_type(self, value):
        # A list of strings was given
        if isinstance(value, list):
            # Clear the vector before setting
            self._prim_type.clear()
            self._prim_type += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<string>")):
            self._prim_type._vector = value
        else:
            exit(f"Incorrect type for prim_type {type(value)}. Either a list or a ROOT.vector of strings required.")

    @property
    def prim_injpoint_shc(self):
        return np.array(self._prim_injpoint_shc)

    @prim_injpoint_shc.setter
    def prim_injpoint_shc(self, value):
        self._prim_injpoint_shc = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("prim_injpoint_shc", self._prim_injpoint_shc)

    @property
    def prim_inj_alt_shc(self):
        return self._prim_inj_alt_shc

    @prim_inj_alt_shc.setter
    def prim_inj_alt_shc(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._prim_inj_alt_shc.clear()
            self._prim_inj_alt_shc += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._prim_inj_alt_shc._vector = value
        else:
            exit(f"Incorrect type for prim_inj_alt_shc {type(value)}. Either a list, an array or a ROOT.vector of floats required.")


    @property
    def prim_inj_dir_shc(self):
        return np.array(self._prim_inj_dir_shc)

    @prim_inj_dir_shc.setter
    def prim_inj_dir_shc(self, value):
        self._prim_inj_dir_shc = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("prim_inj_dir_shc", self._prim_inj_dir_shc)

    @property
    def atmos_model(self):
        return str(self._atmos_model)

    @atmos_model.setter
    def atmos_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._atmos_model.string.assign(value)

    @property
    def atmos_model_param(self):
        return np.array(self._atmos_model_param)

    @atmos_model_param.setter
    def atmos_model_param(self, value):
        self._atmos_model_param = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("atmos_model_param", self._atmos_model_param)

    @property
    def magnetic_field(self):
        return np.array(self._magnetic_field)

    @magnetic_field.setter
    def magnetic_field(self, value):
        self._magnetic_field = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("magnetic_field", self._magnetic_field)

    @property
    def xmax_grams(self):
        return self._xmax_grams[0]

    @xmax_grams.setter
    def xmax_grams(self, value):
        self._xmax_grams[0] = value

    @property
    def xmax_pos_shc(self):
        return np.array(self._xmax_pos_shc)

    @xmax_pos_shc.setter
    def xmax_pos_shc(self, value):
        self._xmax_pos_shc = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("xmax_pos_shc", self._xmax_pos_shc)

    @property
    def xmax_distance(self):
        return self._xmax_distance[0]

    @xmax_distance.setter
    def xmax_distance(self, value):
        self._xmax_distance[0] = value

    @property
    def xmax_alt(self):
        return self._xmax_alt[0]

    @xmax_alt.setter
    def xmax_alt(self, value):
        self._xmax_alt[0] = value

    @property
    def hadronic_model(self):
        return str(self._hadronic_model)

    @hadronic_model.setter
    def hadronic_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._hadronic_model.string.assign(value)

    @property
    def low_energy_model(self):
        return str(self._low_energy_model)

    @low_energy_model.setter
    def low_energy_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._low_energy_model.string.assign(value)

    @property
    def cpu_time(self):
        return np.array(self._cpu_time)

    @cpu_time.setter
    def cpu_time(self, value):
        self._cpu_time = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("cpu_time", self._cpu_time)


@dataclass
class EfieldRunSimdataTree(MotherRunTree):
    _tree_name: str = "trunefieldsimdata"

    ## Name and model of the electric field simulator
    # _field_sim: StdString = StdString("")
    ## Name of the atmospheric index of refraction model
    _refractivity_model: StdString = StdString("")
    _refractivity_model_parameters: StdVectorList("double") = StdVectorList("double")
    ## The antenna time window is defined arround a t0 that changes with the antenna, starts on t0+t_pre (thus t_pre is usually negative) and ends on t0+post
    _t_pre: np.ndarray = np.zeros(1, np.float32)
    _t_post: np.ndarray = np.zeros(1, np.float32)
    _t_bin_size: np.ndarray = np.zeros(1, np.float32)

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    # @property
    # def field_sim(self):
    #     return str(self._field_sim)
    #
    # @field_sim.setter
    # def field_sim(self, value):
    #     # Not a string was given
    #     if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
    #         exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")
    #
    #     self._field_sim.string.assign(value)

    @property
    def refractivity_model(self):
        return str(self._refractivity_model)

    @refractivity_model.setter
    def refractivity_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._refractivity_model.string.assign(value)

    @property
    def refractivity_model_parameters(self):
        return self._refractivity_model_parameters

    @refractivity_model_parameters.setter
    def refractivity_model_parameters(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._refractivity_model_parameters.clear()
            self._refractivity_model_parameters += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._refractivity_model_parameters._vector = value
        else:
            exit(f"Incorrect type for refractivity_model_parameters {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def t_pre(self):
        return self._t_pre[0]

    @t_pre.setter
    def t_pre(self, value):
        self._t_pre[0] = value

    @property
    def t_post(self):
        return self._t_post[0]

    @t_post.setter
    def t_post(self, value):
        self._t_post[0] = value

    @property
    def t_bin_size(self):
        return self._t_bin_size[0]

    @t_bin_size.setter
    def t_bin_size(self, value):
        self._t_bin_size[0] = value


@dataclass
class EfieldEventSimdataTree(MotherEventTree):
    _tree_name: str = "teventefieldsimdata"

    _du_id: StdVectorList("int") = StdVectorList("int")  # Detector ID
    _t_0: StdVectorList("float") = StdVectorList("float")  # Time window t0
    _p2p: StdVectorList("float") = StdVectorList("float")  # peak 2 peak amplitudes (x,y,z,modulus)

    # _event_size: np.ndarray = np.zeros(1, np.uint32)
    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")


    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def du_id(self):
        return self._du_id

    @du_id.setter
    def du_id(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("int")):
            self._du_id._vector = value
        else:
            exit(f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def t_0(self):
        return self._t_0

    @t_0.setter
    def t_0(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._t_0.clear()
            self._t_0 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._t_0._vector = value
        else:
            exit(f"Incorrect type for t_0 {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def p2p(self):
        return self._p2p

    @p2p.setter
    def p2p(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._p2p.clear()
            self._p2p += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._p2p._vector = value
        else:
            exit(f"Incorrect type for p2p {type(value)}. Either a list, an array or a ROOT.vector of float required.")


@dataclass
class VoltageRunSimdataTree(MotherRunTree):
    _tree_name: str = "trunvoltagesimdata"

    _signal_sim: StdString = StdString("")  # name and model of the signal simulator

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def signal_sim(self):
        return str(self._signal_sim)

    @signal_sim.setter
    def signal_sim(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._signal_sim.string.assign(value)


@dataclass
class VoltageEventSimdataTree(MotherEventTree):
    _tree_name: str = "teventvoltagesimdata"

    _du_id: StdVectorList("int") = StdVectorList("int")  # Detector ID
    _t_0: StdVectorList("float") = StdVectorList("float")  # Time window t0
    _p2p: StdVectorList("float") = StdVectorList("float")  # peak 2 peak amplitudes (x,y,z,modulus)

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def du_id(self):
        return self._du_id

    @du_id.setter
    def du_id(self, value):

        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("int")):
            self._du_id._vector = value
        else:
            exit(f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def t_0(self):
        return self._t_0

    @t_0.setter
    def t_0(self, value):

        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._t_0.clear()
            self._t_0 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._t_0._vector = value
        else:
            exit(f"Incorrect type for t_0 {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def p2p(self):
        return self._p2p

    @p2p.setter
    def p2p(self, value):

        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._p2p.clear()
            self._p2p += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._p2p._vector = value
        else:
            exit(f"Incorrect type for p2p {type(value)}. Either a list, an array or a ROOT.vector of float required.")


@dataclass
class ADCEventTree(MotherEventTree):
    _tree_name: str = "teventadc"

    ## Common for the whole event
    ## Event size
    _event_size: np.ndarray = np.zeros(1, np.uint32)
    ## Event in the run number
    _t3_number: np.ndarray = np.zeros(1, np.uint32)
    ## First detector unit that triggered in the event
    _first_du: np.ndarray = np.zeros(1, np.uint32)
    ## Unix time corresponding to the GPS seconds of the first triggered station
    _time_seconds: np.ndarray = np.zeros(1, np.uint32)
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    _time_nanoseconds: np.ndarray = np.zeros(1, np.uint32)
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    _event_type: np.ndarray = np.zeros(1, np.uint32)
    ## Event format version of the DAQ
    _event_version: np.ndarray = np.zeros(1, np.uint32)
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = np.zeros(1, np.uint32)


    ## Specific for each Detector Unit
    ## The T3 trigger number
    _event_id: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Detector unit (antenna) ID
    _du_id: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Unix time of the trigger for this DU
    _du_seconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Nanoseconds of the trigger for this DU
    _du_nanoseconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    _trigger_position: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Same as event_type, but event_type could consist of different triggered DUs
    _trigger_flag: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Atmospheric temperature (read via I2C)
    _atm_temperature: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Atmospheric pressure
    _atm_pressure: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Atmospheric humidity
    _atm_humidity: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Acceleration of the antenna in X
    _acceleration_x: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Acceleration of the antenna in Y
    _acceleration_y: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Acceleration of the antenna in Z
    _acceleration_z: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Battery voltage
    _battery_level: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Firmware version
    _firmware_version: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC sampling frequency in MHz
    _adc_sampling_frequency: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC sampling resolution in bits
    _adc_sampling_resolution: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?
    _adc_input_channels: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?
    _adc_enabled_channels: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in all channels
    _adc_samples_count_total: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel 0
    _adc_samples_count_channel0: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel 1
    _adc_samples_count_channel1: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel 2
    _adc_samples_count_channel2: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel 3
    _adc_samples_count_channel3: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    _trigger_pattern: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    _trigger_rate: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    _clock_tick: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Clock ticks per second
    _clock_ticks_per_second: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    _gps_offset: StdVectorList("float") = StdVectorList("float")
    ## GPS leap second
    _gps_leap_second: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS status
    _gps_status: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS alarms
    _gps_alarms: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS warnings
    _gps_warnings: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS time
    _gps_time: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Longitude
    _gps_long: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Latitude
    _gps_lat: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Altitude
    _gps_alt: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS temperature
    _gps_temp: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual). ToDo: Decode?
    _digi_ctrl: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    _digi_prepost_trig_windows: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel 0 properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties0: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel 1 properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties1: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel 2 properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties2: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel 3 properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties3: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel 0 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings0: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel 1 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings1: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel 2 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings2: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel 3 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings3: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## ?? What is it? Some kind of the adc trace offset?
    _ioff: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")
    ## ADC trace 0
    _trace_0: StdVectorList("vector<short>") = StdVectorList("vector<short>")
    ## ADC trace 1
    _trace_1: StdVectorList("vector<short>") = StdVectorList("vector<short>")
    ## ADC trace 2
    _trace_2: StdVectorList("vector<short>") = StdVectorList("vector<short>")
    ## ADC trace 3
    _trace_3: StdVectorList("vector<short>") = StdVectorList("vector<short>")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName()=="":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def event_size(self):
        return self._event_size[0]

    @event_size.setter
    def event_size(self, value: np.uint32) -> None:
        self._event_size[0] = value

    @property
    def t3_number(self):
        return self._t3_number[0]

    @t3_number.setter
    def t3_number(self, value: np.uint32) -> None:
        self._t3_number[0] = value

    @property
    def first_du(self):
        return self._first_du[0]

    @first_du.setter
    def first_du(self, value: np.uint32) -> None:
        self._first_du[0] = value

    @property
    def time_seconds(self):
        return self._time_seconds[0]

    @time_seconds.setter
    def time_seconds(self, value: np.uint32) -> None:
        self._time_seconds[0] = value

    @property
    def time_nanoseconds(self):
        return self._time_nanoseconds[0]

    @time_nanoseconds.setter
    def time_nanoseconds(self, value: np.uint32) -> None:
        self._time_nanoseconds[0] = value

    @property
    def event_type(self):
        return self._event_type[0]

    @event_type.setter
    def event_type(self, value: np.uint32) -> None:
        self._event_type[0] = value

    @property
    def event_version(self):
        return self._event_version[0]

    @event_version.setter
    def event_version(self, value: np.uint32) -> None:
        self._event_version[0] = value

    @property
    def du_count(self):
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

    @property
    def event_id(self):
        return self._event_id

    @event_id.setter
    def event_id(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._event_id.clear()
            self._event_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._event_id._vector = value
        else:
            exit(f"Incorrect type for event_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def du_id(self):
        return self._du_id

    @du_id.setter
    def du_id(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._du_id._vector = value
        else:
            exit(f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def du_seconds(self):
        return self._du_seconds

    @du_seconds.setter
    def du_seconds(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_seconds.clear()
            self._du_seconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_seconds._vector = value
        else:
            exit(f"Incorrect type for du_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    @property
    def du_nanoseconds(self):
        return self._du_nanoseconds

    @du_nanoseconds.setter
    def du_nanoseconds(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_nanoseconds.clear()
            self._du_nanoseconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_nanoseconds._vector = value
        else:
            exit(f"Incorrect type for du_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    @property
    def trigger_position(self):
        return self._trigger_position

    @trigger_position.setter
    def trigger_position(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_position.clear()
            self._trigger_position += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_position._vector = value
        else:
            exit(f"Incorrect type for trigger_position {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def trigger_flag(self):
        return self._trigger_flag

    @trigger_flag.setter
    def trigger_flag(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_flag.clear()
            self._trigger_flag += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_flag._vector = value
        else:
            exit(f"Incorrect type for trigger_flag {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def atm_temperature(self):
        return self._atm_temperature

    @atm_temperature.setter
    def atm_temperature(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_temperature.clear()
            self._atm_temperature += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._atm_temperature._vector = value
        else:
            exit(f"Incorrect type for atm_temperature {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def atm_pressure(self):
        return self._atm_pressure

    @atm_pressure.setter
    def atm_pressure(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_pressure.clear()
            self._atm_pressure += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._atm_pressure._vector = value
        else:
            exit(f"Incorrect type for atm_pressure {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def atm_humidity(self):
        return self._atm_humidity

    @atm_humidity.setter
    def atm_humidity(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_humidity.clear()
            self._atm_humidity += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._atm_humidity._vector = value
        else:
            exit(f"Incorrect type for atm_humidity {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def acceleration_x(self):
        return self._acceleration_x

    @acceleration_x.setter
    def acceleration_x(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._acceleration_x.clear()
            self._acceleration_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._acceleration_x._vector = value
        else:
            exit(f"Incorrect type for acceleration_x {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def acceleration_y(self):
        return self._acceleration_y

    @acceleration_y.setter
    def acceleration_y(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._acceleration_y.clear()
            self._acceleration_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._acceleration_y._vector = value
        else:
            exit(f"Incorrect type for acceleration_y {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def acceleration_z(self):
        return self._acceleration_z

    @acceleration_z.setter
    def acceleration_z(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._acceleration_z.clear()
            self._acceleration_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._acceleration_z._vector = value
        else:
            exit(f"Incorrect type for acceleration_z {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def battery_level(self):
        return self._battery_level

    @battery_level.setter
    def battery_level(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._battery_level.clear()
            self._battery_level += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._battery_level._vector = value
        else:
            exit(f"Incorrect type for battery_level {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def firmware_version(self):
        return self._firmware_version

    @firmware_version.setter
    def firmware_version(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._firmware_version.clear()
            self._firmware_version += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._firmware_version._vector = value
        else:
            exit(f"Incorrect type for firmware_version {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")


    @property
    def adc_sampling_frequency(self):
        return self._adc_sampling_frequency

    @adc_sampling_frequency.setter
    def adc_sampling_frequency(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_sampling_frequency.clear()
            self._adc_sampling_frequency += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_frequency._vector = value
        else:
            exit(f"Incorrect type for adc_sampling_frequency {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_sampling_resolution(self):
        return self._adc_sampling_resolution

    @adc_sampling_resolution.setter
    def adc_sampling_resolution(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_sampling_resolution.clear()
            self._adc_sampling_resolution += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_resolution._vector = value
        else:
            exit(f"Incorrect type for adc_sampling_resolution {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_input_channels(self):
        return self._adc_input_channels

    @adc_input_channels.setter
    def adc_input_channels(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_input_channels.clear()
            self._adc_input_channels += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_input_channels._vector = value
        else:
            exit(f"Incorrect type for adc_input_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_enabled_channels(self):
        return self._adc_enabled_channels

    @adc_enabled_channels.setter
    def adc_enabled_channels(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_enabled_channels.clear()
            self._adc_enabled_channels += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_enabled_channels._vector = value
        else:
            exit(f"Incorrect type for adc_enabled_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_samples_count_total(self):
        return self._adc_samples_count_total

    @adc_samples_count_total.setter
    def adc_samples_count_total(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_total.clear()
            self._adc_samples_count_total += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_total._vector = value
        else:
            exit(f"Incorrect type for adc_samples_count_total {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_samples_count_channel0(self):
        return self._adc_samples_count_channel0

    @adc_samples_count_channel0.setter
    def adc_samples_count_channel0(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel0.clear()
            self._adc_samples_count_channel0 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel0._vector = value
        else:
            exit(f"Incorrect type for adc_samples_count_channel0 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_samples_count_channel1(self):
        return self._adc_samples_count_channel1

    @adc_samples_count_channel1.setter
    def adc_samples_count_channel1(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel1.clear()
            self._adc_samples_count_channel1 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel1._vector = value
        else:
            exit(f"Incorrect type for adc_samples_count_channel1 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_samples_count_channel2(self):
        return self._adc_samples_count_channel2

    @adc_samples_count_channel2.setter
    def adc_samples_count_channel2(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel2.clear()
            self._adc_samples_count_channel2 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel2._vector = value
        else:
            exit(f"Incorrect type for adc_samples_count_channel2 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_samples_count_channel3(self):
        return self._adc_samples_count_channel3

    @adc_samples_count_channel3.setter
    def adc_samples_count_channel3(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel3.clear()
            self._adc_samples_count_channel3 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel3._vector = value
        else:
            exit(f"Incorrect type for adc_samples_count_channel3 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def trigger_pattern(self):
        return self._trigger_pattern

    @trigger_pattern.setter
    def trigger_pattern(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_pattern.clear()
            self._trigger_pattern += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_pattern._vector = value
        else:
            exit(f"Incorrect type for trigger_pattern {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def trigger_rate(self):
        return self._trigger_rate

    @trigger_rate.setter
    def trigger_rate(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_rate.clear()
            self._trigger_rate += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_rate._vector = value
        else:
            exit(f"Incorrect type for trigger_rate {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def clock_tick(self):
        return self._clock_tick

    @clock_tick.setter
    def clock_tick(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._clock_tick.clear()
            self._clock_tick += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._clock_tick._vector = value
        else:
            exit(f"Incorrect type for clock_tick {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def clock_ticks_per_second(self):
        return self._clock_ticks_per_second

    @clock_ticks_per_second.setter
    def clock_ticks_per_second(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._clock_ticks_per_second.clear()
            self._clock_ticks_per_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._clock_ticks_per_second._vector = value
        else:
            exit(f"Incorrect type for clock_ticks_per_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_offset(self):
        return self._gps_offset

    @gps_offset.setter
    def gps_offset(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_offset.clear()
            self._gps_offset += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_offset._vector = value
        else:
            exit(f"Incorrect type for gps_offset {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def gps_leap_second(self):
        return self._gps_leap_second

    @gps_leap_second.setter
    def gps_leap_second(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_leap_second.clear()
            self._gps_leap_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_leap_second._vector = value
        else:
            exit(f"Incorrect type for gps_leap_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_status(self):
        return self._gps_status

    @gps_status.setter
    def gps_status(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_status.clear()
            self._gps_status += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_status._vector = value
        else:
            exit(f"Incorrect type for gps_status {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_alarms(self):
        return self._gps_alarms

    @gps_alarms.setter
    def gps_alarms(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_alarms.clear()
            self._gps_alarms += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_alarms._vector = value
        else:
            exit(f"Incorrect type for gps_alarms {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_warnings(self):
        return self._gps_warnings

    @gps_warnings.setter
    def gps_warnings(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_warnings.clear()
            self._gps_warnings += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_warnings._vector = value
        else:
            exit(f"Incorrect type for gps_warnings {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_time(self):
        return self._gps_time

    @gps_time.setter
    def gps_time(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_time.clear()
            self._gps_time += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._gps_time._vector = value
        else:
            exit(f"Incorrect type for gps_time {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    @property
    def gps_long(self):
        return self._gps_long

    @gps_long.setter
    def gps_long(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_long.clear()
            self._gps_long += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_long._vector = value
        else:
            exit(f"Incorrect type for gps_long {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_lat(self):
        return self._gps_lat

    @gps_lat.setter
    def gps_lat(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_lat.clear()
            self._gps_lat += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_lat._vector = value
        else:
            exit(f"Incorrect type for gps_lat {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_alt(self):
        return self._gps_alt

    @gps_alt.setter
    def gps_alt(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_alt.clear()
            self._gps_alt += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_alt._vector = value
        else:
            exit(f"Incorrect type for gps_alt {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_temp(self):
        return self._gps_temp

    @gps_temp.setter
    def gps_temp(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_temp.clear()
            self._gps_temp += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_temp._vector = value
        else:
            exit(f"Incorrect type for gps_temp {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def digi_ctrl(self):
        return self._digi_ctrl

    @digi_ctrl.setter
    def digi_ctrl(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._digi_ctrl.clear()
            self._digi_ctrl += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._digi_ctrl._vector = value
        else:
            exit(f"Incorrect type for digi_ctrl {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def digi_prepost_trig_windows(self):
        return self._digi_prepost_trig_windows

    @digi_prepost_trig_windows.setter
    def digi_prepost_trig_windows(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._digi_prepost_trig_windows.clear()
            self._digi_prepost_trig_windows += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._digi_prepost_trig_windows._vector = value
        else:
            exit(f"Incorrect type for digi_prepost_trig_windows {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_properties0(self):
        return self._channel_properties0

    @channel_properties0.setter
    def channel_properties0(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties0.clear()
            self._channel_properties0 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties0._vector = value
        else:
            exit(f"Incorrect type for channel_properties0 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_properties1(self):
        return self._channel_properties1

    @channel_properties1.setter
    def channel_properties1(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties1.clear()
            self._channel_properties1 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties1._vector = value
        else:
            exit(f"Incorrect type for channel_properties1 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_properties2(self):
        return self._channel_properties2

    @channel_properties2.setter
    def channel_properties2(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties2.clear()
            self._channel_properties2 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties2._vector = value
        else:
            exit(f"Incorrect type for channel_properties2 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_properties3(self):
        return self._channel_properties3

    @channel_properties3.setter
    def channel_properties3(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties3.clear()
            self._channel_properties3 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties3._vector = value
        else:
            exit(f"Incorrect type for channel_properties3 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_trig_settings0(self):
        return self._channel_trig_settings0

    @channel_trig_settings0.setter
    def channel_trig_settings0(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings0.clear()
            self._channel_trig_settings0 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings0._vector = value
        else:
            exit(f"Incorrect type for channel_trig_settings0 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_trig_settings1(self):
        return self._channel_trig_settings1

    @channel_trig_settings1.setter
    def channel_trig_settings1(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings1.clear()
            self._channel_trig_settings1 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings1._vector = value
        else:
            exit(f"Incorrect type for channel_trig_settings1 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_trig_settings2(self):
        return self._channel_trig_settings2

    @channel_trig_settings2.setter
    def channel_trig_settings2(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings2.clear()
            self._channel_trig_settings2 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings2._vector = value
        else:
            exit(f"Incorrect type for channel_trig_settings2 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_trig_settings3(self):
        return self._channel_trig_settings3

    @channel_trig_settings3.setter
    def channel_trig_settings3(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings3.clear()
            self._channel_trig_settings3 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings3._vector = value
        else:
            exit(f"Incorrect type for channel_trig_settings3 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def ioff(self):
        return self._ioff

    @ioff.setter
    def ioff(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._ioff.clear()
            self._ioff += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._ioff._vector._vector = value
        else:
            exit(f"Incorrect type for ioff {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def trace_0(self):
        return self._trace_0

    @trace_0.setter
    def trace_0(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_0.clear()
            self._trace_0 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<short>")):
            # With vectors, I think the address is assigned, so in principle the below is needed only on the first setting of the branch
            self._trace_0._vector = value
        else:
            exit(f"Incorrect type for trace_0 {type(value)}. Either a list, an array or a ROOT.vector of vector<short> required.")

    @property
    def trace_1(self):
        return self._trace_1

    @trace_1.setter
    def trace_1(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_1.clear()
            self._trace_1 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<short>")):
            self._trace_1._vector = value
        else:
            exit(f"Incorrect type for trace_1 {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required.")

    @property
    def trace_2(self):
        return self._trace_2

    @trace_2.setter
    def trace_2(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_2.clear()
            self._trace_2 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<short>")):
            self._trace_2._vector = value
        else:
            exit(f"Incorrect type for trace_2 {type(value)}. Either a list, an array or a ROOT.vector of vector<short> required.")

    @property
    def trace_3(self):
        return self._trace_3

    @trace_3.setter
    def trace_3(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_3.clear()
            self._trace_3 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<short>")):
            self._trace_3._vector = value
        else:
            exit(f"Incorrect type for trace_3 {type(value)}. Either a list, an array or a ROOT.vector of vector<short> required.")


@dataclass
class VoltageEventTree(MotherEventTree):
    _tree_name: str = "teventvoltage"

    # _du_id: StdVectorList("int") = StdVectorList("int")
    # _event_size: np.ndarray = np.zeros(1, np.uint32)
    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")

    ## Common for the whole event
    ## Event size
    _event_size: np.ndarray = np.zeros(1, np.uint32)
    ## Event in the run number
    _t3_number: np.ndarray = np.zeros(1, np.uint32)
    ## First detector unit that triggered in the event
    _first_du: np.ndarray = np.zeros(1, np.uint32)
    ## Unix time corresponding to the GPS seconds of the trigger
    _time_seconds: np.ndarray = np.zeros(1, np.uint32)
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    _time_nanoseconds: np.ndarray = np.zeros(1, np.uint32)
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    _event_type: np.ndarray = np.zeros(1, np.uint32)
    ## Event format version of the DAQ
    _event_version: np.ndarray = np.zeros(1, np.uint32)
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = np.zeros(1, np.uint32)


    ## Specific for each Detector Unit
    ## The T3 trigger number
    _event_id: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Detector unit (antenna) ID
    _du_id: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Unix time of the trigger for this DU
    _du_seconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Nanoseconds of the trigger for this DU
    _du_nanoseconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Unix time of the start of the trace for this DU
    # _du_t0_seconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Nanoseconds of the start of the trace for this DU
    # _du_t0_nanoseconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    _trigger_position: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Same as event_type, but event_type could consist of different triggered DUs
    _trigger_flag: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Atmospheric temperature (read via I2C)
    _atm_temperature: StdVectorList("float") = StdVectorList("float")
    ## Atmospheric pressure
    _atm_pressure: StdVectorList("float") = StdVectorList("float")
    ## Atmospheric humidity
    _atm_humidity: StdVectorList("float") = StdVectorList("float")
    ## Acceleration of the antenna in X
    _acceleration_x: StdVectorList("float") = StdVectorList("float")
    ## Acceleration of the antenna in Y
    _acceleration_y: StdVectorList("float") = StdVectorList("float")
    ## Acceleration of the antenna in Z
    _acceleration_z: StdVectorList("float") = StdVectorList("float")
    ## Battery voltage
    _battery_level: StdVectorList("float") = StdVectorList("float")
    ## Firmware version
    _firmware_version: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC sampling frequency in MHz
    _adc_sampling_frequency: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC sampling resolution in bits
    _adc_sampling_resolution: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?
    _adc_input_channels: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?
    _adc_enabled_channels: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in all channels
    _adc_samples_count_total: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel x
    _adc_samples_count_channel_x: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel y
    _adc_samples_count_channel_y: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel z
    _adc_samples_count_channel_z: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    _trigger_pattern: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    _trigger_rate: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    _clock_tick: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Clock ticks per second
    _clock_ticks_per_second: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    _gps_offset: StdVectorList("float") = StdVectorList("float")
    ## GPS leap second
    _gps_leap_second: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS status
    _gps_status: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS alarms
    _gps_alarms: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS warnings
    _gps_warnings: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS time
    _gps_time: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Longitude
    _gps_long: StdVectorList("float") = StdVectorList("float")
    ## Latitude
    _gps_lat: StdVectorList("float") = StdVectorList("float")
    ## Altitude
    _gps_alt: StdVectorList("float") = StdVectorList("float")
    ## X position in site's referential
    _pos_x: StdVectorList("float") = StdVectorList("float")
    ## Y position in site's referential
    _pos_y: StdVectorList("float") = StdVectorList("float")
    ## Z position in site's referential
    _pos_z: StdVectorList("float") = StdVectorList("float")
    ## GPS temperature
    _gps_temp: StdVectorList("float") = StdVectorList("float")
    ## Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual). ToDo: Decode?
    _digi_ctrl: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    _digi_prepost_trig_windows: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel x properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties_x: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel y properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties_y: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel z properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties_z: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel x trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings_x: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel y trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings_y: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Channel z trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings_z: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## ?? What is it? Some kind of the adc trace offset?
    _ioff: StdVectorList("unsigned short") = StdVectorList("unsigned short")

    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")

    ## Voltage trace in X direction
    _trace_x: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## Voltage trace in Y direction
    _trace_y: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## Voltage trace in Z direction
    _trace_z: StdVectorList("vector<float>") = StdVectorList("vector<float>")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName()=="":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def event_size(self):
        return self._event_size[0]

    @event_size.setter
    def event_size(self, value: np.uint32) -> None:
        self._event_size[0] = value

    @property
    def t3_number(self):
        return self._t3_number[0]

    @t3_number.setter
    def t3_number(self, value: np.uint32) -> None:
        self._t3_number[0] = value

    @property
    def first_du(self):
        return self._first_du[0]

    @first_du.setter
    def first_du(self, value: np.uint32) -> None:
        self._first_du[0] = value

    @property
    def time_seconds(self):
        return self._time_seconds[0]

    @time_seconds.setter
    def time_seconds(self, value: np.uint32) -> None:
        self._time_seconds[0] = value

    @property
    def time_nanoseconds(self):
        return self._time_nanoseconds[0]

    @time_nanoseconds.setter
    def time_nanoseconds(self, value: np.uint32) -> None:
        self._time_nanoseconds[0] = value

    @property
    def event_type(self):
        return self._event_type[0]

    @event_type.setter
    def event_type(self, value: np.uint32) -> None:
        self._event_type[0] = value

    @property
    def event_version(self):
        return self._event_version[0]

    @event_version.setter
    def event_version(self, value: np.uint32) -> None:
        self._event_version[0] = value

    @property
    def du_count(self):
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

    @property
    def event_id(self):
        return self._event_id

    @event_id.setter
    def event_id(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._event_id.clear()
            self._event_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._event_id._vector = value
        else:
            exit(f"Incorrect type for event_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def du_id(self):
        return self._du_id

    @du_id.setter
    def du_id(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._du_id._vector = value
        else:
            exit(f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def du_seconds(self):
        return self._du_seconds

    @du_seconds.setter
    def du_seconds(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_seconds.clear()
            self._du_seconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_seconds._vector = value
        else:
            exit(f"Incorrect type for du_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    @property
    def du_nanoseconds(self):
        return self._du_nanoseconds

    @du_nanoseconds.setter
    def du_nanoseconds(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_nanoseconds.clear()
            self._du_nanoseconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_nanoseconds._vector = value
        else:
            exit(f"Incorrect type for du_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    # @property
    # def du_t0_seconds(self):
    #     return self._du_t0_seconds
    #
    # @du_t0_seconds.setter
    # def du_t0_seconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray):
    #         # Clear the vector before setting
    #         self._du_t0_seconds.clear()
    #         self._du_t0_seconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_seconds._vector = value
    #     else:
    #         exit(f"Incorrect type for du_t0_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")
    #
    # @property
    # def du_t0_nanoseconds(self):
    #     return self._du_t0_nanoseconds
    #
    # @du_t0_nanoseconds.setter
    # def du_t0_nanoseconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray):
    #         # Clear the vector before setting
    #         self._du_t0_nanoseconds.clear()
    #         self._du_t0_nanoseconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_nanoseconds._vector = value
    #     else:
    #         exit(f"Incorrect type for du_t0_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    @property
    def trigger_position(self):
        return self._trigger_position

    @trigger_position.setter
    def trigger_position(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_position.clear()
            self._trigger_position += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_position._vector = value
        else:
            exit(f"Incorrect type for trigger_position {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def trigger_flag(self):
        return self._trigger_flag

    @trigger_flag.setter
    def trigger_flag(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_flag.clear()
            self._trigger_flag += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_flag._vector = value
        else:
            exit(f"Incorrect type for trigger_flag {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def atm_temperature(self):
        return self._atm_temperature

    @atm_temperature.setter
    def atm_temperature(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_temperature.clear()
            self._atm_temperature += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._atm_temperature._vector = value
        else:
            exit(f"Incorrect type for atm_temperature {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def atm_pressure(self):
        return self._atm_pressure

    @atm_pressure.setter
    def atm_pressure(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_pressure.clear()
            self._atm_pressure += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._atm_pressure._vector = value
        else:
            exit(f"Incorrect type for atm_pressure {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def atm_humidity(self):
        return self._atm_humidity

    @atm_humidity.setter
    def atm_humidity(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_humidity.clear()
            self._atm_humidity += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._atm_humidity._vector = value
        else:
            exit(f"Incorrect type for atm_humidity {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def acceleration_x(self):
        return self._acceleration_x

    @acceleration_x.setter
    def acceleration_x(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._acceleration_x.clear()
            self._acceleration_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._acceleration_x._vector = value
        else:
            exit(f"Incorrect type for acceleration_x {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def acceleration_y(self):
        return self._acceleration_y

    @acceleration_y.setter
    def acceleration_y(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._acceleration_y.clear()
            self._acceleration_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._acceleration_y._vector = value
        else:
            exit(f"Incorrect type for acceleration_y {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def acceleration_z(self):
        return self._acceleration_z

    @acceleration_z.setter
    def acceleration_z(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._acceleration_z.clear()
            self._acceleration_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._acceleration_z._vector = value
        else:
            exit(f"Incorrect type for acceleration_z {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def battery_level(self):
        return self._battery_level

    @battery_level.setter
    def battery_level(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._battery_level.clear()
            self._battery_level += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._battery_level._vector = value
        else:
            exit(f"Incorrect type for battery_level {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def firmware_version(self):
        return self._firmware_version

    @firmware_version.setter
    def firmware_version(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._firmware_version.clear()
            self._firmware_version += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._firmware_version._vector = value
        else:
            exit(f"Incorrect type for firmware_version {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")


    @property
    def adc_sampling_frequency(self):
        return self._adc_sampling_frequency

    @adc_sampling_frequency.setter
    def adc_sampling_frequency(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_sampling_frequency.clear()
            self._adc_sampling_frequency += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_frequency._vector = value
        else:
            exit(f"Incorrect type for adc_sampling_frequency {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_sampling_resolution(self):
        return self._adc_sampling_resolution

    @adc_sampling_resolution.setter
    def adc_sampling_resolution(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_sampling_resolution.clear()
            self._adc_sampling_resolution += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_resolution._vector = value
        else:
            exit(f"Incorrect type for adc_sampling_resolution {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_input_channels(self):
        return self._adc_input_channels

    @adc_input_channels.setter
    def adc_input_channels(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_input_channels.clear()
            self._adc_input_channels += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_input_channels._vector = value
        else:
            exit(f"Incorrect type for adc_input_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_enabled_channels(self):
        return self._adc_enabled_channels

    @adc_enabled_channels.setter
    def adc_enabled_channels(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_enabled_channels.clear()
            self._adc_enabled_channels += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_enabled_channels._vector = value
        else:
            exit(f"Incorrect type for adc_enabled_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_samples_count_total(self):
        return self._adc_samples_count_total

    @adc_samples_count_total.setter
    def adc_samples_count_total(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_total.clear()
            self._adc_samples_count_total += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_total._vector = value
        else:
            exit(f"Incorrect type for adc_samples_count_total {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_samples_count_channel_x(self):
        return self._adc_samples_count_channel_x

    @adc_samples_count_channel_x.setter
    def adc_samples_count_channel_x(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel_x.clear()
            self._adc_samples_count_channel_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel_x._vector = value
        else:
            exit(f"Incorrect type for adc_samples_count_channel_x {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_samples_count_channel_y(self):
        return self._adc_samples_count_channel_y

    @adc_samples_count_channel_y.setter
    def adc_samples_count_channel_y(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel_y.clear()
            self._adc_samples_count_channel_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel_y._vector = value
        else:
            exit(f"Incorrect type for adc_samples_count_channel_y {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def adc_samples_count_channel_z(self):
        return self._adc_samples_count_channel_z

    @adc_samples_count_channel_z.setter
    def adc_samples_count_channel_z(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel_z.clear()
            self._adc_samples_count_channel_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel_z._vector = value
        else:
            exit(f"Incorrect type for adc_samples_count_channel_z {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def trigger_pattern(self):
        return self._trigger_pattern

    @trigger_pattern.setter
    def trigger_pattern(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_pattern.clear()
            self._trigger_pattern += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_pattern._vector = value
        else:
            exit(f"Incorrect type for trigger_pattern {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def trigger_rate(self):
        return self._trigger_rate

    @trigger_rate.setter
    def trigger_rate(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_rate.clear()
            self._trigger_rate += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_rate._vector = value
        else:
            exit(f"Incorrect type for trigger_rate {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def clock_tick(self):
        return self._clock_tick

    @clock_tick.setter
    def clock_tick(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._clock_tick.clear()
            self._clock_tick += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._clock_tick._vector = value
        else:
            exit(f"Incorrect type for clock_tick {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def clock_ticks_per_second(self):
        return self._clock_ticks_per_second

    @clock_ticks_per_second.setter
    def clock_ticks_per_second(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._clock_ticks_per_second.clear()
            self._clock_ticks_per_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._clock_ticks_per_second._vector = value
        else:
            exit(f"Incorrect type for clock_ticks_per_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_offset(self):
        return self._gps_offset

    @gps_offset.setter
    def gps_offset(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_offset.clear()
            self._gps_offset += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_offset._vector = value
        else:
            exit(f"Incorrect type for gps_offset {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def gps_leap_second(self):
        return self._gps_leap_second

    @gps_leap_second.setter
    def gps_leap_second(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_leap_second.clear()
            self._gps_leap_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_leap_second._vector = value
        else:
            exit(f"Incorrect type for gps_leap_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_status(self):
        return self._gps_status

    @gps_status.setter
    def gps_status(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_status.clear()
            self._gps_status += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_status._vector = value
        else:
            exit(f"Incorrect type for gps_status {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_alarms(self):
        return self._gps_alarms

    @gps_alarms.setter
    def gps_alarms(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_alarms.clear()
            self._gps_alarms += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_alarms._vector = value
        else:
            exit(f"Incorrect type for gps_alarms {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_warnings(self):
        return self._gps_warnings

    @gps_warnings.setter
    def gps_warnings(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_warnings.clear()
            self._gps_warnings += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_warnings._vector = value
        else:
            exit(f"Incorrect type for gps_warnings {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_time(self):
        return self._gps_time

    @gps_time.setter
    def gps_time(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_time.clear()
            self._gps_time += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._gps_time._vector = value
        else:
            exit(f"Incorrect type for gps_time {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    @property
    def gps_long(self):
        return self._gps_long

    @gps_long.setter
    def gps_long(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_long.clear()
            self._gps_long += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_long._vector = value
        else:
            exit(f"Incorrect type for gps_long {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def gps_lat(self):
        return self._gps_lat

    @gps_lat.setter
    def gps_lat(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_lat.clear()
            self._gps_lat += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_lat._vector = value
        else:
            exit(f"Incorrect type for gps_lat {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def gps_alt(self):
        return self._gps_alt

    @gps_alt.setter
    def gps_alt(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_alt.clear()
            self._gps_alt += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_alt._vector = value
        else:
            exit(f"Incorrect type for gps_alt {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def gps_temp(self):
        return self._gps_temp

    @gps_temp.setter
    def gps_temp(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_temp.clear()
            self._gps_temp += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_temp._vector = value
        else:
            exit(f"Incorrect type for gps_temp {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def pos_x(self):
        return self._pos_x

    @pos_x.setter
    def pos_x(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._pos_x.clear()
            self._pos_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._pos_x._vector = value
        else:
            exit(f"Incorrect type for pos_x {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def pos_y(self):
        return self._pos_y

    @pos_y.setter
    def pos_y(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._pos_y.clear()
            self._pos_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._pos_y._vector = value
        else:
            exit(f"Incorrect type for pos_y {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def pos_z(self):
        return self._pos_z

    @pos_z.setter
    def pos_z(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._pos_z.clear()
            self._pos_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._pos_z._vector = value
        else:
            exit(f"Incorrect type for pos_z {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def digi_ctrl(self):
        return self._digi_ctrl

    @digi_ctrl.setter
    def digi_ctrl(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._digi_ctrl.clear()
            self._digi_ctrl += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._digi_ctrl._vector = value
        else:
            exit(f"Incorrect type for digi_ctrl {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def digi_prepost_trig_windows(self):
        return self._digi_prepost_trig_windows

    @digi_prepost_trig_windows.setter
    def digi_prepost_trig_windows(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._digi_prepost_trig_windows.clear()
            self._digi_prepost_trig_windows += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._digi_prepost_trig_windows._vector = value
        else:
            exit(f"Incorrect type for digi_prepost_trig_windows {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_properties_x(self):
        return self._channel_properties_x

    @channel_properties_x.setter
    def channel_properties_x(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties_x.clear()
            self._channel_properties_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties_x._vector = value
        else:
            exit(f"Incorrect type for channel_properties_x {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_properties_y(self):
        return self._channel_properties_y

    @channel_properties_y.setter
    def channel_properties_y(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties_y.clear()
            self._channel_properties_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties_y._vector = value
        else:
            exit(f"Incorrect type for channel_properties_y {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_properties_z(self):
        return self._channel_properties_z

    @channel_properties_z.setter
    def channel_properties_z(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties_z.clear()
            self._channel_properties_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties_z._vector = value
        else:
            exit(f"Incorrect type for channel_properties_z {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_trig_settings_x(self):
        return self._channel_trig_settings_x

    @channel_trig_settings_x.setter
    def channel_trig_settings_x(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings_x.clear()
            self._channel_trig_settings_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings_x._vector = value
        else:
            exit(f"Incorrect type for channel_trig_settings_x {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_trig_settings_y(self):
        return self._channel_trig_settings_y

    @channel_trig_settings_y.setter
    def channel_trig_settings_y(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings_y.clear()
            self._channel_trig_settings_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings_y._vector = value
        else:
            exit(f"Incorrect type for channel_trig_settings_y {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def channel_trig_settings_z(self):
        return self._channel_trig_settings_z

    @channel_trig_settings_z.setter
    def channel_trig_settings_z(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings_z.clear()
            self._channel_trig_settings_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings_z._vector = value
        else:
            exit(f"Incorrect type for channel_trig_settings_z {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def ioff(self):
        return self._ioff

    @ioff.setter
    def ioff(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._ioff.clear()
            self._ioff += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._ioff._vector = value
        else:
            exit(f"Incorrect type for ioff {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")


    @property
    def trace_x(self):
        return self._trace_x

    @trace_x.setter
    def trace_x(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_x.clear()
            self._trace_x += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_x._vector = value
        else:
            exit(f"Incorrect type for trace_x {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required.")

    @property
    def trace_y(self):
        return self._trace_y

    @trace_y.setter
    def trace_y(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_y.clear()
            self._trace_y += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_y._vector = value
        else:
            exit(f"Incorrect type for trace_y {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def trace_z(self):
        return self._trace_z

    @trace_z.setter
    def trace_z(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_z.clear()
            self._trace_z += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_z._vector = value
        else:
            exit(f"Incorrect type for trace_z {type(value)}. Either a list, an array or a ROOT.vector of float required.")

@dataclass
class EfieldEventTree(MotherEventTree):
    _tree_name: str = "teventefield"

    # _du_id: StdVectorList("int") = StdVectorList("int")
    # _event_size: np.ndarray = np.zeros(1, np.uint32)
    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")

    ## Common for the whole event
    ## Unix time corresponding to the GPS seconds of the trigger
    _time_seconds: np.ndarray = np.zeros(1, np.uint32)
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    _time_nanoseconds: np.ndarray = np.zeros(1, np.uint32)
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    _event_type: np.ndarray = np.zeros(1, np.uint32)
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = np.zeros(1, np.uint32)


    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    _du_id: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Unix time of the trigger for this DU
    _du_seconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Nanoseconds of the trigger for this DU
    _du_nanoseconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Unix time of the start of the trace for this DU
    # _du_t0_seconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Nanoseconds of the start of the trace for this DU
    # _du_t0_nanoseconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    _trigger_position: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Same as event_type, but event_type could consist of different triggered DUs
    _trigger_flag: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Atmospheric temperature (read via I2C)
    _atm_temperature: StdVectorList("float") = StdVectorList("float")
    ## Atmospheric pressure
    _atm_pressure: StdVectorList("float") = StdVectorList("float")
    ## Atmospheric humidity
    _atm_humidity: StdVectorList("float") = StdVectorList("float")
    ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    _trigger_pattern: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    _trigger_rate: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Longitude
    _gps_long: StdVectorList("float") = StdVectorList("float")
    ## Latitude
    _gps_lat: StdVectorList("float") = StdVectorList("float")
    ## Altitude
    _gps_alt: StdVectorList("float") = StdVectorList("float")
    ## X position in site's referential
    _pos_x: StdVectorList("float") = StdVectorList("float")
    ## Y position in site's referential
    _pos_y: StdVectorList("float") = StdVectorList("float")
    ## Z position in site's referential
    _pos_z: StdVectorList("float") = StdVectorList("float")
    ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    _digi_prepost_trig_windows: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")

    ## Efield trace in X direction
    _trace_x: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## Efield trace in Y direction
    _trace_y: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## Efield trace in Z direction
    _trace_z: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT magnitude in X direction
    _fft_mag_x: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT magnitude in Y direction
    _fft_mag_y: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT magnitude in Z direction
    _fft_mag_z: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT phase in X direction
    _fft_phase_x: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT phase in Y direction
    _fft_phase_y: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT phase in Z direction
    _fft_phase_z: StdVectorList("vector<float>") = StdVectorList("vector<float>")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName()=="":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def time_seconds(self):
        return self._time_seconds[0]

    @time_seconds.setter
    def time_seconds(self, value: np.uint32) -> None:
        self._time_seconds[0] = value

    @property
    def time_nanoseconds(self):
        return self._time_nanoseconds[0]

    @time_nanoseconds.setter
    def time_nanoseconds(self, value: np.uint32) -> None:
        self._time_nanoseconds[0] = value

    @property
    def event_type(self):
        return self._event_type[0]

    @event_type.setter
    def event_type(self, value: np.uint32) -> None:
        self._event_type[0] = value

    @property
    def du_count(self):
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

    @property
    def du_id(self):
        return self._du_id

    @du_id.setter
    def du_id(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._du_id._vector = value
        else:
            exit(f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def du_seconds(self):
        return self._du_seconds

    @du_seconds.setter
    def du_seconds(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_seconds.clear()
            self._du_seconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_seconds._vector = value
        else:
            exit(f"Incorrect type for du_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    @property
    def du_nanoseconds(self):
        return self._du_nanoseconds

    @du_nanoseconds.setter
    def du_nanoseconds(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_nanoseconds.clear()
            self._du_nanoseconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_nanoseconds._vector = value
        else:
            exit(f"Incorrect type for du_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    # @property
    # def du_t0_seconds(self):
    #     return self._du_t0_seconds
    #
    # @du_t0_seconds.setter
    # def du_t0_seconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray):
    #         # Clear the vector before setting
    #         self._du_t0_seconds.clear()
    #         self._du_t0_seconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_seconds._vector = value
    #     else:
    #         exit(f"Incorrect type for du_t0_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")
    #
    # @property
    # def du_t0_nanoseconds(self):
    #     return self._du_t0_nanoseconds
    #
    # @du_t0_nanoseconds.setter
    # def du_t0_nanoseconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray):
    #         # Clear the vector before setting
    #         self._du_t0_nanoseconds.clear()
    #         self._du_t0_nanoseconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_nanoseconds._vector = value
    #     else:
    #         exit(f"Incorrect type for du_t0_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    @property
    def trigger_position(self):
        return self._trigger_position

    @trigger_position.setter
    def trigger_position(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_position.clear()
            self._trigger_position += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_position._vector = value
        else:
            exit(f"Incorrect type for trigger_position {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def trigger_flag(self):
        return self._trigger_flag

    @trigger_flag.setter
    def trigger_flag(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_flag.clear()
            self._trigger_flag += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_flag._vector = value
        else:
            exit(f"Incorrect type for trigger_flag {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def atm_temperature(self):
        return self._atm_temperature

    @atm_temperature.setter
    def atm_temperature(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_temperature.clear()
            self._atm_temperature += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._atm_temperature._vector = value
        else:
            exit(f"Incorrect type for atm_temperature {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def atm_pressure(self):
        return self._atm_pressure

    @atm_pressure.setter
    def atm_pressure(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_pressure.clear()
            self._atm_pressure += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._atm_pressure._vector = value
        else:
            exit(f"Incorrect type for atm_pressure {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def atm_humidity(self):
        return self._atm_humidity

    @atm_humidity.setter
    def atm_humidity(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_humidity.clear()
            self._atm_humidity += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._atm_humidity._vector = value
        else:
            exit(f"Incorrect type for atm_humidity {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def trigger_pattern(self):
        return self._trigger_pattern

    @trigger_pattern.setter
    def trigger_pattern(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_pattern.clear()
            self._trigger_pattern += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_pattern._vector = value
        else:
            exit(f"Incorrect type for trigger_pattern {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def trigger_rate(self):
        return self._trigger_rate

    @trigger_rate.setter
    def trigger_rate(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_rate.clear()
            self._trigger_rate += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_rate._vector = value
        else:
            exit(f"Incorrect type for trigger_rate {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def gps_long(self):
        return self._gps_long

    @gps_long.setter
    def gps_long(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_long.clear()
            self._gps_long += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_long._vector = value
        else:
            exit(f"Incorrect type for gps_long {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def gps_lat(self):
        return self._gps_lat

    @gps_lat.setter
    def gps_lat(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_lat.clear()
            self._gps_lat += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_lat._vector = value
        else:
            exit(f"Incorrect type for gps_lat {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def gps_alt(self):
        return self._gps_alt

    @gps_alt.setter
    def gps_alt(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_alt.clear()
            self._gps_alt += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_alt._vector = value
        else:
            exit(f"Incorrect type for gps_alt {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def pos_x(self):
        return self._pos_x

    @pos_x.setter
    def pos_x(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._pos_x.clear()
            self._pos_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._pos_x._vector = value
        else:
            exit(f"Incorrect type for pos_x {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def pos_y(self):
        return self._pos_y

    @pos_y.setter
    def pos_y(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._pos_y.clear()
            self._pos_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._pos_y._vector = value
        else:
            exit(f"Incorrect type for pos_y {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def pos_z(self):
        return self._pos_z

    @pos_z.setter
    def pos_z(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._pos_z.clear()
            self._pos_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._pos_z._vector = value
        else:
            exit(f"Incorrect type for pos_z {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def digi_prepost_trig_windows(self):
        return self._digi_prepost_trig_windows

    @digi_prepost_trig_windows.setter
    def digi_prepost_trig_windows(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._digi_prepost_trig_windows.clear()
            self._digi_prepost_trig_windows += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._digi_prepost_trig_windows._vector = value
        else:
            exit(f"Incorrect type for digi_prepost_trig_windows {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def trace_x(self):
        return self._trace_x

    @trace_x.setter
    def trace_x(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_x.clear()
            self._trace_x += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_x._vector = value
        else:
            exit(f"Incorrect type for trace_x {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required.")

    @property
    def trace_y(self):
        return self._trace_y

    @trace_y.setter
    def trace_y(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_y.clear()
            self._trace_y += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_y._vector = value
        else:
            exit(f"Incorrect type for trace_y {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def trace_z(self):
        return self._trace_z

    @trace_z.setter
    def trace_z(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_z.clear()
            self._trace_z += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_z._vector = value
        else:
            exit(f"Incorrect type for trace_z {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_mag_x(self):
        return self._fft_mag_x

    @fft_mag_x.setter
    def fft_mag_x(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_mag_x.clear()
            self._fft_mag_x += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_mag_x._vector = value
        else:
            exit(f"Incorrect type for fft_mag_x {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_mag_y(self):
        return self._fft_mag_y

    @fft_mag_y.setter
    def fft_mag_y(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_mag_y.clear()
            self._fft_mag_y += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_mag_y._vector = value
        else:
            exit(f"Incorrect type for fft_mag_y {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_mag_z(self):
        return self._fft_mag_z

    @fft_mag_z.setter
    def fft_mag_z(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_mag_z.clear()
            self._fft_mag_z += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_mag_z._vector = value
        else:
            exit(f"Incorrect type for fft_mag_z {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_phase_x(self):
        return self._fft_phase_x

    @fft_phase_x.setter
    def fft_phase_x(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_phase_x.clear()
            self._fft_phase_x += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_phase_x._vector = value
        else:
            exit(f"Incorrect type for fft_phase_x {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_phase_y(self):
        return self._fft_phase_y

    @fft_phase_y.setter
    def fft_phase_y(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_phase_y.clear()
            self._fft_phase_y += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_phase_y._vector = value
        else:
            exit(f"Incorrect type for fft_phase_y {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_phase_z(self):
        return self._fft_phase_z

    @fft_phase_z.setter
    def fft_phase_z(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_phase_z.clear()
            self._fft_phase_z += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_phase_z._vector = value
        else:
            exit(f"Incorrect type for fft_phase_z {type(value)}. Either a list, an array or a ROOT.vector of float required.")


## A class wrapping around TTree describing the detector information, like position, type, etc. It works as an array for readout in principle
@dataclass
class DetectorInfo(DataTree):
    _tree_name: str = "tdetectorinfo"

    ## Detector Unit id
    _du_id: np.ndarray = np.zeros(1, np.float32)
    ## Currently read out unit. Not publicly visible
    _cur_du_id: int = -1
    ## Detector longitude
    _long: np.ndarray = np.zeros(1, np.float32)
    ## Detector latitude
    _lat: np.ndarray = np.zeros(1, np.float32)
    ## Detector altitude
    _alt: np.ndarray = np.zeros(1, np.float32)
    ## Detector type
    _type: StdString = StdString("antenna")
    ## Detector description
    _description: StdString = StdString("")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName()=="":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    def __len__(self):
        return self._tree.GetEntries()

    # def __delitem__(self, index):
    #     self.vector.erase(index)
    #
    # def insert(self, index, value):
    #     self.vector.insert(index, value)
    #
    # def __setitem__(self, index, value):
    #     self.vector[index] = value

    def __getitem__(self, index):
        # Read the unit if not read already
        if self._cur_du_id != index:
            self._tree.GetEntryWithIndex(index)
            self._cur_du_id = index
        return self

    ## Don't really add friends, just generates an index
    def add_proper_friends(self):

        # Create the index
        self._tree.BuildIndex("du_id")

    ## Fill the tree
    def fill(self):
        self._tree.Fill()

    @property
    def du_id(self):
        return self._du_id[0]

    @du_id.setter
    def du_id(self, value: int) -> None:
        self._du_id[0] = value

    @property
    def long(self):
        return self._long[0]

    @long.setter
    def long(self, value: np.float32) -> None:
        self._long[0] = value

    @property
    def lat(self):
        return self._lat[0]

    @lat.setter
    def lat(self, value: np.float32) -> None:
        self._lat[0] = value

    @property
    def alt(self):
        return self._alt[0]

    @alt.setter
    def alt(self, value: np.float32) -> None:
        self._alt[0] = value

    @property
    def type(self):
        return str(self._type)

    @type.setter
    def type(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for type {type(value)}. Either a string or a ROOT.std.string is required.")

        self._type.string.assign(value)

    @property
    def description(self):
        return str(self._description)

    @description.setter
    def description(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for description {type(value)}. Either a string or a ROOT.std.string is required.")

        self._description.string.assign(value)

@dataclass
class ShowerEventZHAireSTree(MotherEventTree):
    _tree_name: str = "teventshowerzhaires"

    # ToDo: we need explanations of these parameters

    _relative_thining: StdString = StdString("")
    _weight_factor: np.ndarray = np.zeros(1, np.float64)
    _gamma_energy_cut: StdString = StdString("")
    _electron_energy_cut: StdString = StdString("")
    _muon_energy_cut: StdString = StdString("")
    _meson_energy_cut: StdString = StdString("")
    _nucleon_energy_cut: StdString = StdString("")
    _other_parameters: StdString = StdString("")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def relative_thining(self):
        return str(self._relative_thining)

    @relative_thining.setter
    def relative_thining(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for relative_thining {type(value)}. Either a string or a ROOT.std.string is required.")

        self._relative_thining.string.assign(value)

    @property
    def weight_factor(self):
        return self._weight_factor[0]

    @weight_factor.setter
    def weight_factor(self, value: np.float64) -> None:
        self._weight_factor[0] = value

    @property
    def gamma_energy_cut(self):
        return str(self._gamma_energy_cut)

    @gamma_energy_cut.setter
    def gamma_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for gamma_energy_cut {type(value)}. Either a string or a ROOT.std.string is required.")

        self._gamma_energy_cut.string.assign(value)

    @property
    def electron_energy_cut(self):
        return str(self._electron_energy_cut)

    @electron_energy_cut.setter
    def electron_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for electron_energy_cut {type(value)}. Either a string or a ROOT.std.string is required.")

        self._electron_energy_cut.string.assign(value)

    @property
    def muon_energy_cut(self):
        return str(self._muon_energy_cut)

    @muon_energy_cut.setter
    def muon_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for muon_energy_cut {type(value)}. Either a string or a ROOT.std.string is required.")

        self._muon_energy_cut.string.assign(value)

    @property
    def meson_energy_cut(self):
        return str(self._meson_energy_cut)

    @meson_energy_cut.setter
    def meson_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for meson_energy_cut {type(value)}. Either a string or a ROOT.std.string is required.")

        self._meson_energy_cut.string.assign(value)

    @property
    def nucleon_energy_cut(self):
        return str(self._nucleon_energy_cut)

    @nucleon_energy_cut.setter
    def nucleon_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for nucleon_energy_cut {type(value)}. Either a string or a ROOT.std.string is required.")

        self._nucleon_energy_cut.string.assign(value)

    @property
    def other_parameters(self):
        return str(self._other_parameters)

    @other_parameters.setter
    def other_parameters(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for other_parameters {type(value)}. Either a string or a ROOT.std.string is required.")

        self._other_parameters.string.assign(value)


## Exception raised when an already existing event/run is added to a tree
class NotUniqueEvent(Exception):
    pass