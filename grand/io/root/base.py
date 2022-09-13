"""
Read/Write python interface to GRAND data (real and simulated) stored in Cern ROOT TTrees.

This is the interface for accessing GRAND ROOT TTrees that do not require the user (reader/writer of the TTrees) to have any knowledge of ROOT. It also hides the internals from the data generator, so that the changes in the format are not concerning the user.
"""
import ROOT
import numpy as np
import sys

# This import changes in Python 3.10
if sys.version_info.major >= 3 and sys.version_info.minor < 10:
    from collections import MutableSequence
else:
    from collections.abc import MutableSequence
from dataclasses import dataclass, field

# from typing import List, Union

## A list of generated Trees
grand_tree_list = []
"""Internal list of generated Trees"""


class StdVectorList(MutableSequence):
    """A python list interface to ROOT's std::vector"""

    # vec_type = ""

    def __init__(self, vec_type, value=[]):
        """
        Args:
            vec_type: C++ type for the std::vector (eg. "float", "string", etc.)
            value: list with which to init the vector
        """
        self._vector = ROOT.vector(vec_type)(value)
        #: C++ type for the std::vector (eg. "float", "string", etc.)
        self.vec_type = vec_type

    def __len__(self):
        return self._vector.size()

    def __delitem__(self, index):
        self._vector.erase(index)

    def insert(self, index, value):
        """Insert the value to the vector at index"""
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
        """Append the value to the list"""
        # std::vector does not want numpy types for push_back, need to use .item()
        if isinstance(value, np.generic):
            self._vector.push_back(value.item())
        else:
            self._vector.push_back(value)

    def clear(self):
        """Remove all the values from the vector"""
        self._vector.clear()

    def __repr__(self):
        if len(self._vector) > 0:
            if "std.vector" in str(type(self._vector[0])):
                return str([list(el) for el in self._vector])

        return str(list(self._vector))


class StdString:
    """A python string interface to ROOT's std::string"""

    def __init__(self, value):
        self.string = ROOT.string(value)

    def __len__(self):
        return len(str(self.string))

    def __repr__(self):
        return str(self.string)


@dataclass
class DataTree:
    """
    Mother class for GRAND Tree data classes
    """

    # File handle
    _file: ROOT.TFile = None
    # File name
    _file_name: str = None  #: hjehe
    # Tree object
    _tree: ROOT.TTree = None
    # Tree name
    _tree_name: str = ""
    # A list of run_numbers or (run_number, event_number) pairs in the Tree
    _entry_list: list = field(default_factory=list)

    @property
    def tree(self):
        """The ROOT TTree in which the variables' values are stored"""
        return self._tree

    @property
    def file(self):
        """The ROOT TFile in which the tree is stored"""
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
        elif self._file_name is not None and self._file_name != "":
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
            if f := ROOT.gROOT.GetListOfFiles().FindObject(self._file_name):
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
                if self._tree == None:
                    print(
                        f"No valid {self._tree_name} TTree in the file {self._file.GetName()}. Creating a new one."
                    )
                    self._create_tree()
            else:
                print(f"creating tree {self._tree_name} {self._file}")
                self._create_tree()

        # Fill the runs/events numbers from the tree (important if it already existed)
        self.fill_entry_list()

    ## Create the tree
    def _create_tree(self, tree_name=""):
        if tree_name != "":
            self._tree_name = tree_name
        self._tree = ROOT.TTree(self._tree_name, self._tree_name)

    def fill(self):
        """Adds the current variable values as a new event to the tree"""
        pass

    def write(self, *args, close_file=True, **kwargs):
        """Write the tree to the file"""
        # Add the tree friends to this tree
        self.add_proper_friends()

        # If string is ending with ".root" given as a first argument, create the TFile
        # ToDo: Handle TFile if added as the argument
        creating_file = False
        if len(args) > 0 and ".root" in args[0][-5:]:
            self._file_name = args[0]
            if f := ROOT.gROOT.GetListOfFiles().FindObject(self._file_name):
                self._file = f
            # Overwrite requested
            # ToDo: this does not really seem to work now
            elif "overwrite" in kwargs:
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
        if len(args) < 2:
            args = ["", ROOT.TObject.kWriteDelete]
        self._tree.Write(*args)

        # If TFile was created here, close it
        if creating_file and close_file:
            # Need to set 0 directory so that closing of the file does not delete the internal TTree
            self._tree.SetDirectory(ROOT.nullptr)
            self._file.Close()

    ## Fills the entry list from the tree
    def fill_entry_list(self):
        """Fills the entry list from the tree"""
        pass

    ## Check if specified run_number/event_number already exist in the tree
    def is_unique_event(self):
        """Check if specified run_number/event_number already exist in the tree"""
        pass

    ## Add the proper friend trees to this tree (reimplemented in daughter classes)
    def add_proper_friends(self):
        """Add the proper friend trees to this tree (reimplemented in daughter classes)"""
        pass

    def scan(self, *args):
        """Print out the values of the specified members of the tree (TTree::Scan() interface)"""
        self._tree.Scan(*args)

    def get_entry(self, ev_no):
        """Read into memory the ev_no entry of the tree"""
        res = self._tree.GetEntry(ev_no)
        self.assign_branches()
        return res

    ## All three methods below return the number of entries
    def get_entries(self):
        """Return the number of events in the tree"""
        return self._tree.GetEntries()

    def get_number_of_entries(self):
        """Return the number of events in the tree"""
        return self.get_entries()

    def get_number_of_events(self):
        """Return the number of events in the tree"""
        return self.get_number_of_entries()

    def add_friend(self, value):
        """Add a friend to the tree"""
        self._tree.AddFriend(value)

    def remove_friend(self, value):
        """Remove a friend from the tree"""
        self._tree.RemoveFriend(value)

    def set_tree_index(self, value):
        """Set the tree index (necessary for working with friends)"""
        self._tree.SetTreeIndex(value)

    ## Create branches of the TTree based on the class fields
    def create_branches(self, set_if_exists=True):
        """Create branches of the TTree based on the class fields"""
        # Reset all branch addresses just in case
        self._tree.ResetBranchAddresses()

        # If branches already exist, set their address instead of creating, if requested
        set_branches = False
        if set_if_exists and len(self._tree.GetListOfBranches()) > 0:
            set_branches = True

        # Loop through the class fields
        for field in self.__dataclass_fields__:
            # Skip "tree" and "file" fields, as they are not the part of the stored data
            if (
                field == "_tree"
                or field == "_file"
                or field == "_file_name"
                or field == "_tree_name"
                or field == "_cur_du_id"
                or field == "_entry_list"
            ):
                continue
            # Create a branch for the field
            # print(self.__dataclass_fields__[field])
            self.create_branch_from_field(self.__dataclass_fields__[field], set_branches)

    ## Create a specific branch of a TTree computing its type from the corresponding class field
    def create_branch_from_field(self, value, set_branches=False):
        """Create a specific branch of a TTree computing its type from the corresponding class field"""
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
                self._tree.Branch(
                    value.name[1:], getattr(self, value.name), value.name[1:] + val_type
                )
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
        elif id(value.type) == id(StdString):
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
        """Assign branches to the instance - without calling it, the instance does not show the values read to the TTree"""
        # Assign the TTree branches to the class fields
        for field in self.__dataclass_fields__:
            # Skip "tree" and "file" fields, as they are not the part of the stored data
            if (
                field == "_tree"
                or field == "_file"
                or field == "_file_name"
                or field == "_tree_name"
                or field == "_cur_du_id"
                or field == "_entry_list"
            ):
                continue
            # print(field, self.__dataclass_fields__[field])
            # Read the TTree branch
            u = getattr(self._tree, field[1:])
            # print(field[1:], self.__dataclass_fields__[field].name, u, type(u))
            # Assign the TTree branch value to the class field
            setattr(self, field[1:], u)

    ## Get entry with indices
    def get_entry_with_index(self, run_no=0, evt_no=0):
        """Get the event with run_no and evt_no"""
        res = self._tree.GetEntryWithIndex(run_no, evt_no)
        if res == 0 or res == -1:
            print(
                f"No event with event number {evt_no} and run number {run_no} in the tree. Please provide proper numbers."
            )
            return 0

        self.assign_branches()
        return res

    ## Print out the tree scheme
    def print(self):
        """Print out the tree scheme"""
        return self._tree.Print()


## A class wrapping around TTree describing the detector information, like position, type, etc. It works as an array for readout in principle
@dataclass
class DetectorInfo(DataTree):
    """A class wrapping around TTree describing the detector information, like position, type, etc. It works as an array for readout in principle"""

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

        if self._tree.GetName() == "":
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
        """Don't really add friends, just generates an index"""
        # Create the index
        self._tree.BuildIndex("du_id")

    ## Fill the tree
    def fill(self):
        """Fill the tree"""
        self._tree.Fill()

    @property
    def du_id(self):
        """Detector Unit id"""
        return self._du_id[0]

    @du_id.setter
    def du_id(self, value: int) -> None:
        self._du_id[0] = value

    @property
    def long(self):
        """Detector longitude"""
        return self._long[0]

    @long.setter
    def long(self, value: np.float32) -> None:
        self._long[0] = value

    @property
    def lat(self):
        """Detector latitude"""
        return self._lat[0]

    @lat.setter
    def lat(self, value: np.float32) -> None:
        self._lat[0] = value

    @property
    def alt(self):
        """Detector altitude"""
        return self._alt[0]

    @alt.setter
    def alt(self, value: np.float32) -> None:
        self._alt[0] = value

    @property
    def type(self):
        """Detector type"""
        return str(self._type)

    @type.setter
    def type(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for type {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._type.string.assign(value)

    @property
    def description(self):
        """Detector description"""
        return str(self._description)

    @description.setter
    def description(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for description {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._description.string.assign(value)


## Exception raised when an already existing event/run is added to a tree
class NotUniqueEvent(Exception):
    """Exception raised when an already existing event/run is added to a tree"""

    pass
