"""
Read/Write python interface to GRAND data (real and simulated) stored in Cern ROOT TTrees.

This is the interface for accessing GRAND ROOT TTrees that do not require the user (reader/writer of the TTrees) to have any knowledge of ROOT. It also hides the internals from the data generator, so that the changes in the format are not concerning the user.
"""

from logging import getLogger
import sys
import datetime
import os

import ROOT
import numpy as np
import glob
import array

from collections import defaultdict

# Conversion between numpy dtype and array.array typecodes
numpy_to_array_typecodes = {np.dtype('int8'): 'b', np.dtype('int16'): 'h', np.dtype('int32'): 'i', np.dtype('int64'): 'q', np.dtype('uint8'): 'B', np.dtype('uint16'): 'H', np.dtype('uint32'): 'I', np.dtype('uint64'): 'Q', np.dtype('float32'): 'f', np.dtype('float64'): 'd', np.dtype('complex64'): 'F', np.dtype('complex128'): 'D', np.dtype('int16'): 'h'}
# numpy_to_array_typecodes = {np.int8: 'b', np.int16: 'h', np.int32: 'i', np.int64: 'q', np.uint8: 'B', np.uint16: 'H', np.uint32: 'I', np.uint64: 'Q', np.float32: 'f', np.float64: 'd', np.complex64: 'F', np.complex128: 'D', "int8": 'b', "int16": 'h', "int32": 'i', "int64": 'q', "uint8": 'B', "uint16": 'H', "uint32": 'I', "uint64": 'Q', "float32": 'f', "float64": 'd', "complex64": 'F', "complex128": 'D'}

# Conversion between C++ type and array.array typecodes
cpp_to_array_typecodes = {'char': 'b', 'short': 'h', 'int': 'i', 'long long': 'q', 'unsigned char': 'B', 'unsigned short': 'H', 'unsigned int': 'I', 'unsigned long long': 'Q', 'float': 'f', 'double': 'd', 'string': 'u'}

# This import changes in Python 3.10
if sys.version_info.major >= 3 and sys.version_info.minor < 10:
    from collections import MutableSequence
else:
    from collections.abc import MutableSequence
from dataclasses import dataclass, field

thismodule = sys.modules[__name__]

logger = getLogger(__name__)

## A list of generated Trees
grand_tree_list = []
"""Internal list of generated Trees"""


# _ups: StdVectorList["float"]() = field(default_factory=lambda: StdVectorList("float"))


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
        # Basic type of the vector - different than vec_type in case of vector of vectors
        if "<" in self.vec_type:
            self.basic_vec_type = self.vec_type.split("<")[-1].split(">")[0]
        else:
            self.basic_vec_type = self.vec_type
        # The number of dimensions of this vector
        self.ndim = vec_type.count("vector") + 1

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
                if self.ndim == 2:
                    return list(self._vector[index])
                elif self.ndim == 3:
                    return [list(el) for el in self._vector[index]]
                elif self.ndim == 4:
                    return [list(el) for el1 in self._vector[index] for el in el1]
                else:
                    return self._vector[index]
            else:
                return self._vector[index]
        else:
            raise IndexError("list index out of range")

    def __eq__(self, other):
        # Make comparisons to lists with same contents true
        if self.ndim == 1:
            return list(self._vector) == other
        elif self.ndim == 2:
            return [list(el) for el in self._vector] == other
        elif self.ndim == 3:
            return [list(el) for el1 in self._vector for el in el1] == other

    def append(self, value):
        """Append the value to the list"""
        # std::vector does not want numpy types for push_back, need to use .item()
        if isinstance(value, np.generic):
            self._vector.push_back(value.item())
        else:
            if (isinstance(value, list) and self.basic_vec_type.split()[-1] == "float") or isinstance(value, np.ndarray):
                if self.ndim == 2: value = array.array(cpp_to_array_typecodes[self.basic_vec_type], value)
                if self.ndim == 3: value = [array.array(cpp_to_array_typecodes[self.basic_vec_type], el) for el in value]
                if self.ndim == 4: value = [[array.array(cpp_to_array_typecodes[self.basic_vec_type], el1) for el1 in el] for el in value]
            self._vector.push_back(value)

    def clear(self):
        """Remove all the values from the vector"""
        self._vector.clear()

    def __repr__(self):
        if len(self._vector) > 0:
            if "std.vector" in str(type(self._vector[0])):
                return str([list(el) for el in self._vector])

        return str(list(self._vector))

    # The standard way of adding stuff to a ROOT.vector is +=. However, for ndim>2 it wants only list, so let's always give it a list
    def __iadd__(self, value):
        # Python float is really a double, so for vector of floats it sometimes is not accepted (but why not always?)
        if (isinstance(value, list) and self.basic_vec_type.split()[-1] == "float") or isinstance(value, np.ndarray):
            if self.ndim == 1: value = array.array(cpp_to_array_typecodes[self.basic_vec_type], value)
            if self.ndim == 2: value = [array.array(cpp_to_array_typecodes[self.basic_vec_type], el) for el in value]
            if self.ndim == 3: value = [[array.array(cpp_to_array_typecodes[self.basic_vec_type], el1) for el1 in el] for el in value]

        # elif isinstance(value, np.ndarray):
        #     # Fastest to convert this way to array.array, that is accepted properly by ROOT.vector()
        #     value = array.array(numpy_to_array_typecodes[value.dtype], value.tobytes())
        else:
            value = list(value)

        # The list needs to have simple Python types - ROOT.vector does not accept numpy types
        try:
            self._vector += value
        except TypeError:
            # Slow conversion to simple types. No better idea for now
            if self.basic_vec_type.split()[-1] in ["int", "long", "short", "char", "float"]:
                if self.ndim == 1: value = array.array(cpp_to_array_typecodes[self.basic_vec_type], value)
                if self.ndim == 2: value = [array.array(cpp_to_array_typecodes[self.basic_vec_type], el) for el in value]
                if self.ndim == 3: value = [[array.array(cpp_to_array_typecodes[self.basic_vec_type], el1) for el1 in el] for el in value]

            self._vector += value

        return self

class StdVectorListDesc:
    """A descriptor for StdVectorList - makes use of it possible in dataclasses without setting property and setter"""
    def __init__(self, vec_type):
        self.factory = lambda: StdVectorList(vec_type)

    def __set_name__(self, type, name):
        self.name = name
        self.attrname = f"_{name}"

    def create_default(self, obj):
        setattr(obj, self.attrname, self.factory())

    def __get__(self, obj, obj_type):
        if not hasattr(obj, self.attrname):
            self.create_default(obj)
        return getattr(obj, self.attrname)

    def __set__(self, obj, value):
        if not hasattr(obj, self.attrname):
            self.create_default(obj)
        # This is needed for default init as a field of an upper class
        if isinstance(value, StdVectorListDesc):
            value = getattr(obj, self.attrname)
        inst = getattr(obj, self.attrname)
        vector = inst._vector
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray) or isinstance(value, StdVectorList):
            # Clear the vector before setting
            vector.clear()
            inst += value
        # A vector of vectors was given
        elif isinstance(value, ROOT.vector(inst.vec_type)):
            vector = value
        else:
            if "vector" in inst.vec_type:
                raise ValueError(
                    f"Incorrect type for {self.name} {type(value)}. Either a list of lists, a list of arrays or a ROOT.vector of vectors required."
                )
            else:
                raise ValueError(
                    f"Incorrect type for {self.name} {type(value)}. Either a list, an array or a ROOT.vector required."
                )

class TTreeScalarDesc:
    """A descriptor for scalars assigned to TTrees as numpy arrays of size 1 - makes use of it possible in dataclasses without setting property and setter"""
    def __init__(self, dtype):
        self.factory = lambda: np.zeros(1, dtype)

    def __set_name__(self, type, name):
        self.name = name
        self.attrname = f"_{name}"

    def create_default(self, obj):
        setattr(obj, self.attrname, self.factory())

    def __get__(self, obj, obj_type):
        if not hasattr(obj, self.attrname):
            self.create_default(obj)
        return getattr(obj, self.attrname)[0]

    def __set__(self, obj, value):
        if not hasattr(obj, self.attrname):
            self.create_default(obj)
        # This is needed for default init as a field of an upper class
        if isinstance(value, TTreeScalarDesc):
            value = getattr(obj, self.attrname)
        inst = getattr(obj, self.attrname)
        print("inst", inst, inst[0], value)

        inst[0] = value

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
    _file_name: str = None
    # Tree object
    _tree: ROOT.TTree = None
    # Tree name
    _tree_name: str = ""
    # Tree type
    _type: str = ""
    # A list of run_numbers or (run_number, event_number) pairs in the Tree
    _entry_list: list = field(default_factory=list)
    # Comment - if needed, added by user
    _comment: str = ""
    # TTree creation date/time in UTC - a naive time, without timezone set
    _creation_datetime: datetime.datetime = None
    # Modification history - JSON
    _modification_history: str = ""

    # Fields that are not branches
    _nonbranch_fields = [
        "_nonbranch_fields",
        "_type",
        "_tree",
        "_file",
        "_file_name",
        "_tree_name",
        "_cur_du_id",
        "_entry_list",
        "_attributes_and_properties",
        "_comment",
        "_creation_datetime",
        "_modification_software",
        "_modification_software_version",
        "_source_datetime",
        "_analysis_level",
        "_modification_history",
        "__setattr__"
    ]

    # def __setattr__(self, key, value):
    def mod_setattr(self, key, value):
        # Create a list of attributes and properties for the class if it doesn't exist
        if not hasattr(self, "_attributes_and_properties"):
            super().__setattr__("_attributes_and_properties", set([el1 for el in type(self).__mro__[:-1] for el1 in list(el.__dict__.keys()) + list(el.__annotations__.keys())]))
        # If the attribute not in the list of class's attributes and properties, don't add it
        if key not in self._attributes_and_properties:
            raise AttributeError(f"{key} attribute for class {type(self)} doesn't exist.")
        super().__setattr__(key, value)

    @property
    def tree(self):
        """The ROOT TTree in which the variables' values are stored"""
        return self._tree

    @property
    def type(self):
        """The type of the tree"""
        return self._type

    @type.setter
    def type(self, val: str) -> None:
        # The meta field does not exist, add it
        if (el:=self._tree.GetUserInfo().FindObject("type")) == None:
            self._tree.GetUserInfo().Add(ROOT.TNamed("type", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)
        # Update the property
        self._type = val

    @property
    def file(self):
        """The ROOT TFile in which the tree is stored"""
        return self._file

    @file.setter
    def file(self, val: ROOT.TFile) -> None:
        self._set_file(val)

    @property
    def tree_name(self):
        """The name of the TTree"""
        return self._tree_name

    @tree_name.setter
    def tree_name(self, val):
        """Set the tree name"""
        # ToDo: enforce the name to start with the type!
        self._tree_name = val
        self._tree.SetName(val)
        self._tree.SetTitle(val)

    @property
    def file_name(self):
        """The file in which the TTree is stored"""
        return self._file_name

    @property
    def comment(self):
        """Comment - if needed, added by user"""
        return self._comment

    @comment.setter
    def comment(self, val: str) -> None:
        # The meta field does not exist, add it
        if (el:=self._tree.GetUserInfo().FindObject("comment")) == None:
            self._tree.GetUserInfo().Add(ROOT.TNamed("comment", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)

        # Update the property
        self._comment = val

    @property
    def creation_datetime(self):
        return self._creation_datetime

    @creation_datetime.setter
    def creation_datetime(self, val: datetime.datetime) -> None:
        # If datetime was given, convert it to int
        if type(val) == datetime.datetime:
            val = int(val.timestamp())
            val_dt = val
        elif type(val) == int:
            val_dt = datetime.datetime.fromtimestamp(val)
        else:
            raise ValueError(f"Unsupported type {type(val)} for creation_datetime!")

        # The meta field does not exist, add it
        if (el := self._tree.GetUserInfo().FindObject("creation_datetime")) == None:
            self._tree.GetUserInfo().Add(ROOT.TParameter(int)("creation_datetime", val))
        # The meta field exists, change the value
        else:
            el.SetVal(val)

        self._creation_datetime = val_dt

    @property
    def modification_history(self):
        """Modification_history - if needed, added by user"""
        return self._modification_history

    @modification_history.setter
    def modification_history(self, val: str) -> None:
        # The meta field does not exist, add it
        if (el:=self._tree.GetUserInfo().FindObject("modification_history")) == None:
            self._tree.GetUserInfo().Add(ROOT.TNamed("modification_history", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)

        # Update the property
        self._modification_history = val

    @classmethod
    def get_default_tree_name(cls):
        """Gets the default name of the tree of the class"""
        return cls._tree_name

    def __post_init__(self):
        self._type = type(self).__name__

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
        elif self._tree is not None:
            self._set_tree(self._tree)
        # or create the tree
        else:
            self._create_tree()

        for field in self.__dict__:
            if field[0] == "_" and hasattr(self, field[1:]) == False and isinstance(self.__dict__[field], StdVectorList):
                print("not set for", field)
                
        self.__setattr__ = self.mod_setattr

    ## Return the iterable over self
    def __iter__(self):
        # Always start the iteration with the first entry
        current_entry = 0

        while current_entry < self._tree.GetEntries():
            self.get_entry(current_entry)
            yield self
            current_entry += 1

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
                # If file exists, initially open in the read-only mode (changed during write())
                if os.path.isfile(self._file_name):
                    self._file = ROOT.TFile(self._file_name, "read")
                # If the file does not exist, create it
                else:
                    self._file = ROOT.TFile(self._file_name, "create")

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
                    logger.warning(
                        f"No valid {self._tree_name} TTree in the file {self._file.GetName()}. Creating a new one."
                    )
                    self._create_tree()
            else:
                logger.info(f"creating tree {self._tree_name} {self._file}")
                self._create_tree()

        self.assign_metadata()

        # Fill the runs/events numbers from the tree (important if it already existed)
        self.fill_entry_list()

    ## Create the tree
    def _create_tree(self, tree_name=""):
        if tree_name != "":
            self._tree_name = tree_name
        self._tree = ROOT.TTree(self._tree_name, self._tree_name)

        self.create_metadata()

    def fill(self):
        """Adds the current variable values as a new event to the tree"""
        pass

    def write(self, *args, close_file=True, overwrite=False, force_close_file=False, **kwargs):
        """Write the tree to the file"""
        # Add the tree friends to this tree
        self.add_proper_friends()

        # If string is ending with ".root" given as a first argument, create the TFile
        # ToDo: Handle TFile if added as the argument
        creating_file = False
        if len(args) > 0 and ".root" in args[0][-5:]:
            self._file_name = args[0]
            # The TFile object is already in memory, just use it
            if f := ROOT.gROOT.GetListOfFiles().FindObject(self._file_name):
                self._file = f
                # File exists, but reopen the file in the update mode in case it was read only
                self._file.ReOpen("update")
            # Create a new TFile object
            else:
                creating_file = True
                # Overwrite requested
                # ToDo: this does not really seem to work now
                if overwrite:
                    self._file = ROOT.TFile(args[0], "recreate")
                else:
                    # By default append
                    self._file = ROOT.TFile(args[0], "update")
            # Make the tree save itself in this file
            self._tree.SetDirectory(self._file)
            # args passed to the TTree::Write() should be the following
            args = args[1:]
        # File exists, so reopen the file in the update mode in case it was read only
        else:
            self._file.ReOpen("update")

        # ToDo: For now, I don't know how to do that: Check if the entries in possible tree in the file do not already contain entries from the current tree

        # If the writing options are not explicitly specified, add kWriteDelete option, that deletes the old cycle after writing the new cycle in the TFile
        if len(args) < 2:
            args = ["", ROOT.TObject.kWriteDelete]
        # ToDo: make sure that the tree has different name than the trees existing in the file!
        self._tree.Write(*args)

        # If TFile was created here, close it
        if (creating_file and close_file) or force_close_file:
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

    def draw(self, varexp, selection, option="", nentries=ROOT.TTree.kMaxEntries, firstentry=0):
        """An interface to TTree::Draw(). Allows for drawing specific TTree columns or getting their values with get_vX()."""
        return self._tree.Draw(varexp, selection, option, nentries, firstentry)

    def get_v1(self):
        '''Get first vector of results from draw()'''
        return self._tree.GetV1()

    def get_v2(self):
        '''Get second vector of results from draw()'''
        return self._tree.GetV2()

    def get_v3(self):
        '''Get third vector of results from draw()'''
        return self._tree.GetV3()

    def get_v4(self):
        '''Get fourth vector of results from draw()'''
        return self._tree.GetV4()

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

    def add_friend(self, value, filename=""):
        # ToDo: Due to a bug discovered during DC1, disable adding of the friends for now
        return 0
        """Add a friend to the tree"""
        self._tree.AddFriend(value, filename)

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
        # for field in self.__dataclass_fields__:
        for field in self.__dict__:
            # Skip fields that are not the part of the stored data
            if field in self._nonbranch_fields:
                continue
            # Create a branch for the field
            # print(self.__dataclass_fields__[field])
            # self.create_branch_from_field(self.__dataclass_fields__[field], set_branches)
            self.create_branch_from_field(self.__dict__[field], set_branches, field)

    ## Create a specific branch of a TTree computing its type from the corresponding class field
    def create_branch_from_field(self, value, set_branches=False, value_name=""):
        """Create a specific branch of a TTree computing its type from the corresponding class field"""
        # Handle numpy arrays
        # for key in dir(value):
        #     print(getattr(value, key))

        # Not all values start with _
        branch_name = value_name
        if value_name[0] == "_":
            branch_name = value_name[1:]

        if isinstance(value, np.ndarray):
            # Generate ROOT TTree data type string

            # If the value is a (1D) numpy array with more than 1 value, make it an (1D) array in ROOT
            if value.size > 1:
                val_type = f"[{value.size}]"
            else:
                val_type = ""

            # Data type
            if value.dtype == np.int8:
                val_type += "/B"
            elif value.dtype == np.uint8:
                val_type += "/b"
            elif value.dtype == np.int16:
                val_type += "/S"
            elif value.dtype == np.uint16:
                val_type += "/s"
            elif value.dtype == np.int32:
                val_type += "/I"
            elif value.dtype == np.uint32:
                val_type += "/i"
            elif value.dtype == np.int64:
                val_type += "/L"
            elif value.dtype == np.uint64:
                val_type += "/l"
            elif value.dtype == np.float32:
                val_type += "/F"
            elif value.dtype == np.float64:
                val_type += "/D"
            elif value.dtype == np.bool_:
                val_type += "/O"

            # Create the branch
            if not set_branches:
                # self._tree.Branch(value_name[1:], getattr(self, value_name), value_name[1:] + val_type)
                self._tree.Branch(branch_name, getattr(self, value_name), branch_name + val_type)
            # Or set its address
            else:
                # self._tree.SetBranchAddress(value_name[1:], getattr(self, value_name))
                self._tree.SetBranchAddress(branch_name, getattr(self, value_name))
        # ROOT vectors as StdVectorList
        # elif "vector" in str(type(value.default)):
        # Not sure why type is not StdVectorList when using factory... thus not isinstance, but id comparison
        # elif id(value.type) == id(StdVectorList):
        elif type(value) == StdVectorList or type(value) == StdVectorListDesc:
            # Create the branch
            if not set_branches:
                self._tree.Branch(branch_name, getattr(self, value_name)._vector)
            # Or set its address
            else:
                # Try to attach the branch from the tree
                try:
                    # self._tree.SetBranchAddress(value_name[1:], getattr(self, value_name)._vector)
                    self._tree.SetBranchAddress(branch_name, getattr(self, value_name)._vector)
                except:
                    # logger.warning(f"Could not find branch {value_name[1:]} in tree {self.tree_name}. This branch will not be filled.")
                    logger.warning(f"Could not find branch {branch_name} in tree {self.tree_name}. This branch will not be filled.")
        # For some reason that I don't get, the isinstance does not work here
        # elif isinstance(value.type, str):
        # elif id(value.type) == id(StdString):
        elif type(value) == StdString:
            # Create the branch
            if not set_branches:
                # self._tree.Branch(value.name[1:], getattr(self, value.name).string)
                self._tree.Branch(branch_name, getattr(self, value_name).string)
            # Or set its address
            else:
                # self._tree.SetBranchAddress(value.name[1:], getattr(self, value.name).string)
                self._tree.SetBranchAddress(branch_name, getattr(self, value_name).string)
        else:
            raise ValueError(f"Unsupported type {type(value)}. Can't create a branch.")

    ## Assign branches to the instance - without calling it, the instance does not show the values read to the TTree
    def assign_branches(self):
        """Assign branches to the instance - without calling it, the instance does not show the values read to the TTree"""
        # Assign the TTree branches to the class fields
        for field in self.__dataclass_fields__:
            # Skip fields that are not the part of the stored data
            if field in self._nonbranch_fields:
                continue
            field_name = field
            if field[0] == "_": field_name=field[1:]
            # print(field, self.__dataclass_fields__[field])
            # Read the TTree branch
            u = getattr(self._tree, field_name)
            # print("*", field[1:], self.__dataclass_fields__[field].name, u, type(u), id(u))
            # Assign the TTree branch value to the class field
            setattr(self, field_name, u)

    ## Create metadata for the tree
    def create_metadata(self):
        """Create metadata for the tree"""
        # ToDo: stupid, because default values are generated here and in the class fields definitions. But definition of the class field does not call the setter, which is needed to attach these fields to the tree.
        self.type = self._type
        self.comment = ""
        self.creation_datetime = datetime.datetime.utcnow()
        self.modification_history = ""

    ## Assign metadata to the instance - without calling it, the instance does not show the metadata stored in the TTree
    def assign_metadata(self):
        """Assign metadata to the instance - without calling it, the instance does not show the metadata stored in the TTree"""
        metadata_count = self._tree.GetUserInfo().GetEntries()
        for i in range(metadata_count):
            el = self._tree.GetUserInfo().At(i)
            # meta as TNamed
            if type(el) == ROOT.TNamed:
                setattr(self, el.GetName(), el.GetTitle())
            # meta as TParameter
            else:
                setattr(self, el.GetName(), el.GetVal())

    ## Get entry with indices
    def get_entry_with_index(self, run_no=0, evt_no=0):
        """Get the event with run_no and evt_no"""
        res = self._tree.GetEntryWithIndex(run_no, evt_no)
        if res == 0 or res == -1:
            logger.error(
                f"No event with event number {evt_no} and run number {run_no} in the {self.tree_name} tree. Please provide proper numbers."
            )
            return 0

        self.assign_branches()
        return res

    ## Print out the tree scheme
    def print(self):
        """Print out the tree scheme"""
        return self._tree.Print()

    ## Print the meta information
    def print_metadata(self):
        """Print the meta information"""
        for el in self._tree.GetUserInfo():
            try:
                val = el.GetVal()
            except:
                val = el.GetTitle()
                # Add "" to the string to show it is a string
                val = f'"{val}"'
            print(f"{el.GetName():40} {val}")

    @staticmethod
    def get_metadata_as_dict(tree):
        """Get the meta information as a dictionary"""

        metadata = {}

        for el in tree.GetUserInfo():
            try:
                val = el.GetVal()
            except:
                val = el.GetTitle()

            # Convert unix time if this is the datetime
            if "datetime" in el.GetName() and val!=0:
                val = datetime.datetime.fromtimestamp(val)

            metadata[el.GetName()] = val

        return metadata

    ## Copy contents of another dataclass instance of the same type to this instance
    def copy_contents(self, source):
        """Copy contents of another dataclass instance of similar type to this instance
        The source has to have some field the same as this tree. For example EventEfieldTree and EventVoltageTree"""
        # ToDo: Shallow copy with assigning branches would be probably faster, but it would be... shallow ;)
        for k in source.__dict__.keys():
            # Skip the nonbranch fields and fields not belonging to this tree type
            if k in self._nonbranch_fields or k not in self.__dict__.keys():
                continue
            try:
                setattr(self, k[1:], getattr(source, k[1:]))
            except TypeError:
                logger.warning(f"The type of {k} in {source.tree_name} and {self._tree_name} differs. Not copying.")

    def get_tree_size(self):
        """Get the tree size in memory and on disk, similar to what comes from the Print()"""
        mem_size = self._tree.GetDirectory().GetKey(self._tree_name).GetKeylen()
        mem_size += self._tree.GetTotBytes()
        b = ROOT.TBufferFile(ROOT.TBuffer.kWrite, 10000)
        self._tree.IsA().WriteBuffer(b, self._tree)
        mem_size += b.Length()

        disk_size = self._tree.GetZipBytes()
        disk_size += self._tree.GetDirectory().GetKey(self._tree_name).GetNbytes()

        return mem_size, disk_size

    def close_file(self):
        """Close the file associated to the tree"""
        self._file.Close()


## A mother class for classes with Run values
@dataclass
class MotherRunTree(DataTree):
    """A mother class for classes with Run values"""

    _run_number: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))

    @property
    def run_number(self):
        """The run number for this tree entry"""
        return int(self._run_number[0])

    @run_number.setter
    def run_number(self, val: np.uint32) -> None:
        self._run_number[0] = val

    def fill(self):
        """Adds the current variable values as a new event to the tree"""
        # If the current run_number and event_number already exist, raise an exception
        if not self.is_unique_event():
            raise NotUniqueEvent(
                f"A run with run_number={self.run_number} already exists in the TTree."
            )

        # Repoen the file in write mode, if it exists
        # Reopening in case of different mode takes here ~0.06 s, in case of the same mode, 0.0005 s, so negligible
        if self._file is not None:
            self._file.ReOpen("update")

        # Fill the tree
        self._tree.Fill()

        # Add the current run_number and event_number to the entry_list
        self._entry_list.append(self.run_number)

    def add_proper_friends(self):
        """Add proper friends to this tree"""
        # Create the indices
        self._tree.BuildIndex("run_number")

    ## List runs in the tree
    def print_list_of_runs(self):
        """List runs in the tree"""
        count = self._tree.Draw("run_number", "", "goff")
        runs = self._tree.GetV1()
        print("List of runs in the tree:")
        for i in range(count):
            print(int(runs[i]))

    ## Gets list of runs in the tree together
    def get_list_of_runs(self):
        """Gets list of runs in the tree together"""
        count = self._tree.Draw("run_number", "", "goff")
        runs = self._tree.GetV1()
        return [int(runs[i]) for i in range(count)]

    # Readout the TTree entry corresponding to the run
    def get_run(self, run_no):
        """Readout the TTree entry corresponding to the run"""
        # Make sure we have an int
        run_no = int(run_no)
        # Try to get the run from the tree
        res = self._tree.GetEntryWithIndex(run_no)
        # If no such entry, return
        if res == 0 or res == -1:
            logger.error(f"No run with run number {run_no}. Please provide a proper number.")
            return 0

        self.assign_branches()

        return res

    def build_index(self, run_id):
        """Build the tree index (necessary for working with friends)"""
        self._tree.BuildIndex(run_id)

    ## Fills the entry list from the tree
    def fill_entry_list(self):
        """Fills the entry list from the tree"""
        # Fill the entry list if there are some entries in the tree
        if (count := self._tree.Draw("run_number", "", "goff")) > 0:
            v1 = np.array(np.frombuffer(self._tree.GetV1(), dtype=np.float64, count=count))
            self._entry_list = [int(el) for el in v1]

    ## Check if specified run_number/event_number already exist in the tree
    def is_unique_event(self):
        """Check if specified run_number/event_number already exist in the tree"""
        # If the entry list does not exist, the event is unique
        if self._entry_list and self.run_number in self._entry_list:
            return False

        return True


## A mother class for classes with Event values
@dataclass
class MotherEventTree(DataTree):
    """A mother class for classes with Event values"""

    _run_number: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    # ToDo: it seems instances propagate this number among them without setting (but not the run number!). I should find why...
    _event_number: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))

    # Unix creation datetime of the source tree; 0 s means no source
    _source_datetime: datetime.datetime = None
    # The tool used to generate this tree's values from another tree
    _modification_software: str = ""
    # The version of the tool used to generate this tree's values from another tree
    _modification_software_version: str = ""
    # The analysis level of this tree
    _analysis_level: int = 0

    @property
    def run_number(self):
        """The run number of the current event"""
        return int(self._run_number[0])

    @run_number.setter
    def run_number(self, val: np.uint32) -> None:
        self._run_number[0] = val

    @property
    def event_number(self):
        """The event number of the current event"""
        return int(self._event_number[0])

    @event_number.setter
    def event_number(self, val: np.uint32) -> None:
        self._event_number[0] = val

    @property
    def source_datetime(self):
        """Unix creation datetime of the source tree; 0 s means no source"""
        # Convert from ROOT's TDatime into Python's datetime object
        # return datetime.datetime.fromtimestamp(self._tree.GetUserInfo().At(3).GetVal())
        return self._source_datetime

    @source_datetime.setter
    def source_datetime(self, val: datetime.datetime) -> None:
        # Remove the existing datetime
        self._tree.GetUserInfo().Remove(self._tree.GetUserInfo().FindObject("source_datetime"))

        # If datetime was given
        if type(val) == datetime.datetime:
            val = int(val.timestamp())
            val_dt = val
        # If timestamp was given - this happens when initialising with self.assign_metadata()
        elif type(val) == int:
            val_dt = datetime.datetime.fromtimestamp(val)
        else:
            raise ValueError(f"Unsupported type {type(val)} for source_datetime!")

        # The meta field does not exist, add it
        if (el:=self._tree.GetUserInfo().FindObject("source_datetime")) == None:
            self._tree.GetUserInfo().Add(ROOT.TParameter(int)("source_datetime", val))
        # The meta field exists, change the value
        else:
            el.SetVal(val)

        self._source_datetime = val_dt


    @property
    def modification_software(self):
        """The tool used to generate this tree's values from another tree"""
        return self._modification_software

    @modification_software.setter
    def modification_software(self, val: str) -> None:
        # The meta field does not exist, add it
        if (el:=self._tree.GetUserInfo().FindObject("modification_software")) == None:
            self._tree.GetUserInfo().Add(ROOT.TNamed("modification_software", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)

        self._modification_software = val

    @property
    def modification_software_version(self):
        """The tool used to generate this tree's values from another tree"""
        return self._modification_software_version

    @modification_software_version.setter
    def modification_software_version(self, val: str) -> None:
        # The meta field does not exist, add it
        if (el:=self._tree.GetUserInfo().FindObject("modification_software_version")) == None:
            self._tree.GetUserInfo().Add(ROOT.TNamed("modification_software_version", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)

        self._modification_software_version = val

    @property
    def analysis_level(self):
        """The analysis level of this tree"""
        return self._analysis_level

    @analysis_level.setter
    def analysis_level(self, val: int) -> None:
        # The meta field does not exist, add it
        if (el:=self._tree.GetUserInfo().FindObject("analysis_level")) == None:
            self._tree.GetUserInfo().Add(ROOT.TParameter(int)("analysis_level", val))
        # The meta field exists, change the value
        else:
            el.SetVal(val)

        self._analysis_level = val

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    ## Create metadata for the tree
    def create_metadata(self):
        """Create metadata for the tree"""
        # First add the medatata of the mother class
        super().create_metadata()
        # ToDo: stupid, because default values are generated here and in the class fields definitions. But definition of the class field does not call the setter, which is needed to attach these fields to the tree.
        self.source_datetime = datetime.datetime.fromtimestamp(0)
        self.modification_software = ""
        self.modification_software_version = ""
        self.analysis_level = 0

    def fill(self):
        """Adds the current variable values as a new event to the tree"""
        # If the current run_number and event_number already exist, raise an exception
        if not self.is_unique_event():
            raise NotUniqueEvent(
                f"An event with (run_number,event_number)=({self.run_number},{self.event_number}) already exists in the TTree {self._tree.GetName()}."
            )

        # Repoen the file in write mode, if it exists
        # Reopening in case of different mode takes here ~0.06 s, in case of the same mode, 0.0005 s, so negligible
        if self._file is not None:
            self._file.ReOpen("update")

        # Fill the tree
        self._tree.Fill()

        # Add the current run_number and event_number to the entry_list
        self._entry_list.append((self.run_number, self.event_number))

    def add_proper_friends(self):
        """Add proper friends to this tree"""
        # Create the indices
        self.build_index("run_number", "event_number")

        # Add the Run tree as a friend if exists already
        loc_vars = dict(locals())
        run_trees = []
        for inst in grand_tree_list:
            if type(inst) is TRun:
                run_trees.append(inst)
        # If any Run tree was found
        if len(run_trees) > 0:
            # Warning if there is more than 1 TRun in memory
            if len(run_trees) > 1:
                logger.warning(
                    f"More than 1 TRun detected in memory. Adding the last one {run_trees[-1]} as a friend"
                )
            # Add the last one TRun as a friend
            run_tree = run_trees[-1]

            # Add the Run TTree as a friend
            self.add_friend(run_tree.tree, run_tree.file)

        # Do not add TADC as a friend to itself
        if not isinstance(self, TADC):
            # Add the ADC tree as a friend if exists already
            adc_trees = []
            for inst in grand_tree_list:
                if type(inst) is TADC:
                    adc_trees.append(inst)
            # If any ADC tree was found
            if len(adc_trees) > 0:
                # Warning if there is more than 1 TADC in memory
                if len(adc_trees) > 1:
                    logger.warning(
                        f"More than 1 TADC detected in memory. Adding the last one {adc_trees[-1]} as a friend"
                    )
                # Add the last one TADC as a friend
                adc_tree = adc_trees[-1]

                # Add the ADC TTree as a friend
                self.add_friend(adc_tree.tree, adc_tree.file)

        # Do not add TRawVoltage as a friend to itself
        if not isinstance(self, TRawVoltage):
            # Add the Voltage tree as a friend if exists already
            voltage_trees = []
            for inst in grand_tree_list:
                if type(inst) is TRawVoltage:
                    voltage_trees.append(inst)
            # If any voltage tree was found
            if len(voltage_trees) > 0:
                # Warning if there is more than 1 TRawVoltage in memory
                if len(voltage_trees) > 1:
                    logger.warning(
                        f"More than 1 TRawVoltage detected in memory. Adding the last one {voltage_trees[-1]} as a friend"
                    )
                # Add the last one TRawVoltage as a friend
                voltage_tree = voltage_trees[-1]

                # Add the Voltage TTree as a friend
                self.add_friend(voltage_tree.tree, voltage_tree.file)

        # Do not add TEfield as a friend to itself
        if not isinstance(self, TEfield):
            # Add the Efield tree as a friend if exists already
            efield_trees = []
            for inst in grand_tree_list:
                if type(inst) is TEfield:
                    efield_trees.append(inst)
            # If any Efield tree was found
            if len(efield_trees) > 0:
                # Warning if there is more than 1 TEfield in memory
                if len(efield_trees) > 1:
                    logger.warning(
                        f"More than 1 TEfield detected in memory. Adding the last one {efield_trees[-1]} as a friend"
                    )
                # Add the last one TEfield as a friend
                efield_tree = efield_trees[-1]

                # Add the Efield TTree as a friend
                self.add_friend(efield_tree.tree, efield_tree.file)

        # Do not add TShower as a friend to itself
        if not isinstance(self, TShower):
            # Add the Shower tree as a friend if exists already
            shower_trees = []
            for inst in grand_tree_list:
                if type(inst) is TShower:
                    shower_trees.append(inst)
            # If any Shower tree was found
            if len(shower_trees) > 0:
                # Warning if there is more than 1 TShower in memory
                if len(shower_trees) > 1:
                    logger.warning(
                        f"More than 1 TShower detected in memory. Adding the last one {shower_trees[-1]} as a friend"
                    )
                # Add the last one TShower as a friend
                shower_tree = shower_trees[-1]

                # Add the Shower TTree as a friend
                self.add_friend(shower_tree.tree, shower_tree.file)

    ## List events in the tree together with runs
    def print_list_of_events(self):
        """List events in the tree together with runs"""
        count = self._tree.Draw("event_number:run_number", "", "goff")
        events = self._tree.GetV1()
        runs = self._tree.GetV2()
        print("List of events in the tree:")
        print("event_number run_number")
        for i in range(count):
            print(int(events[i]), int(runs[i]))

    ## Gets list of events in the tree together with runs
    def get_list_of_events(self):
        """Gets list of events in the tree together with runs"""
        count = self._tree.Draw("event_number:run_number", "", "goff")
        events = self._tree.GetV1()
        runs = self._tree.GetV2()
        return [(int(events[i]), int(runs[i])) for i in range(count)]

    ## Readout the TTree entry corresponding to the event and run
    def get_event(self, ev_no, run_no=0):
        """Readout the TTree entry corresponding to the event and run"""
        # Try to get the requested entry
        res = self._tree.GetEntryWithIndex(run_no, ev_no)
        # If no such entry, return
        if res == 0 or res == -1:
            logger.error(
                f"No event with event number {ev_no} and run number {run_no} in the {self.tree_name} tree. Please provide proper numbers."
            )
            return 0

        self.assign_branches()

        return res

    ## Builds index based on run_id and evt_id for the TTree
    def build_index(self, run_id, evt_id):
        """Builds index based on run_id and evt_id for the TTree"""
        self._tree.BuildIndex(run_id, evt_id)

    ## Fills the entry list from the tree
    def fill_entry_list(self, tree=None):
        """Fills the entry list from the tree"""
        if tree is None:
            tree = self._tree
        # Fill the entry list if there are some entries in the tree
        if (count := tree.Draw("run_number:event_number", "", "goff")) > 0:
            v1 = np.array(np.frombuffer(tree.GetV1(), dtype=np.float64, count=count))
            v2 = np.array(np.frombuffer(tree.GetV2(), dtype=np.float64, count=count))
            self._entry_list = [(int(el[0]), int(el[1])) for el in zip(v1, v2)]

    ## Check if specified run_number/event_number already exist in the tree
    def is_unique_event(self):
        """Check if specified run_number/event_number already exist in the tree"""
        # If the entry list does not exist, the event is unique
        if self._entry_list and (self.run_number, self.event_number) in self._entry_list:
            return False

        return True

    def get_traces_lengths(self):
        """Gets the traces lengths for each event"""

        # If there are no traces in the tree, return None
        if self._tree.GetListOfLeaves().FindObject("trace_x") == None and self._tree.GetListOfLeaves().FindObject("trace_0") == None:
            return None

        traces_lengths = []
        # For ADC traces - 4 traces, different names
        if "ADC" in self.__class__.__name__ or "RawVoltage" in self.__class__.__name__:
            traces_suffixes = [0, 1, 2, 3]
        # Other traces
        else:
            traces_suffixes = ["x", "y", "z"]

        # Get sizes of each traces
        for i in traces_suffixes:
            cnt = self._tree.Draw(f"@trace_{i}.size()", "", "goff")
            traces_lengths.append(np.frombuffer(self._tree.GetV1(), count=cnt, dtype=np.float64).astype(int).tolist())

        return traces_lengths

    def get_list_of_dus(self):
        """Gets the list of all detector units used for each event"""

        # If there are no detector unit ids in the tree, return None
        if self._tree.GetListOfLeaves().FindObject("du_id") == None:
            return None

        # Try to store the currently read entry
        try:
            current_entry = self._tree.GetReadEntry()
        # if failed, store None
        except:
            current_entry = None

        # Get the detector units branch
        du_br = self._tree.GetBranch("du_id")

        # detector_units = []
        # # Loop through all entries
        # for i in range(du_br.GetEntries()):
        #     du_br.GetEntry(i)
        #     detector_units.append(self.du_id)

        count = self.draw("du_id", "", "goff")
        detector_units = np.unique(np.array(np.frombuffer(self.get_v1(), dtype=np.float64, count=count)).astype(int))

        # If there was an entry read before this action, come back to this entry
        if current_entry is not None:
            du_br.GetEntry(current_entry)

        return detector_units

    def get_list_of_all_used_dus(self):
        """Compiles the list of all detector units used in the events of the tree"""
        dus = self.get_list_of_dus()
        if dus is not None:
            return np.unique(np.array(dus).flatten()).tolist()
        else:
            return None


## A class wrapping around a TTree holding values common for the whole run
@dataclass
class TRun(MotherRunTree):
    """A class wrapping around a TTree holding values common for the whole run"""

    _type: str = "run"

    _tree_name: str = "trun"

    ## Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment
    _run_mode: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Run's first event
    _first_event: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## First event time
    _first_event_time: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Run's last event
    _last_event: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Last event time
    _last_event_time: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))

    # These are not from the hardware
    ## Data source: detector, sim, other
    _data_source: StdString = StdString("detector")
    ## Data generator: gtot (in this case)
    _data_generator: StdString = StdString("GRANDlib")
    ## Generator version: gtot version (in this case)
    _data_generator_version: StdString = StdString("0.1.0")
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    _event_type: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Event format version of the DAQ
    _event_version: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Site name
    # _site: StdVectorList("string") = StdVectorList("string")
    _site: StdString = StdString("")
    ## Site layout
    _site_layout: StdString = StdString("")
    # ## Site longitude
    # _site_long: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # ## Site latitude
    # _site_lat: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    ## Origin of the coordinate system used for the array
    _origin_geoid: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))

    ## Detector unit (antenna) ID
    _du_id: StdVectorList = field(default_factory=lambda: StdVectorList("int"))
    ## Detector unit (antenna) (lat,lon,alt) position
    _du_geoid: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## Detector unit (antenna) (x,y,z) position in site's referential
    _du_xyz: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## Detector unit type
    _du_type: StdVectorList = field(default_factory=lambda: StdVectorList("string"))
    ## Detector unit (antenna) angular tilt
    _du_tilt: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## Angular tilt of the ground at the antenna
    _du_ground_tilt: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## Detector unit (antenna) nut ID
    _du_nut: StdVectorList = field(default_factory=lambda: StdVectorList("int"))
    ## Detector unit (antenna) FrontEnd Board ID
    _du_feb: StdVectorList = field(default_factory=lambda: StdVectorList("int"))
    ## Time bin size in ns (for hardware, computed as 1/adc_sampling_frequency)
    _t_bin_size: np.ndarray = field(default_factory=lambda: StdVectorList("float"))

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def run_mode(self):
        """Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment"""
        return self._run_mode[0]

    @run_mode.setter
    def run_mode(self, value: np.uint32) -> None:
        self._run_mode[0] = value

    @property
    def first_event(self):
        """Run's first event"""
        return self._first_event[0]

    @first_event.setter
    def first_event(self, value: np.uint32) -> None:
        self._first_event[0] = value

    @property
    def first_event_time(self):
        """First event time"""
        return self._first_event_time[0]

    @first_event_time.setter
    def first_event_time(self, value: np.uint32) -> None:
        self._first_event_time[0] = value

    @property
    def last_event(self):
        """Run's last event"""
        return self._last_event[0]

    @last_event.setter
    def last_event(self, value: np.uint32) -> None:
        self._last_event[0] = value

    @property
    def last_event_time(self):
        """Last event time"""
        return self._last_event_time[0]

    @last_event_time.setter
    def last_event_time(self, value: np.uint32) -> None:
        self._last_event_time[0] = value

    @property
    def data_source(self):
        """Data source: detector, sim, other"""
        return str(self._data_source)

    @data_source.setter
    def data_source(self, value) -> None:
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._data_source.string.assign(value)

    @property
    def data_generator(self):
        """Data generator: gtot (in this case)"""
        return str(self._data_generator)

    @data_generator.setter
    def data_generator(self, value) -> None:
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._data_generator.string.assign(value)

    @property
    def data_generator_version(self):
        """Generator version: gtot version (in this case)"""
        return str(self._data_generator_version)

    @data_generator_version.setter
    def data_generator_version(self, value) -> None:
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._data_generator_version.string.assign(value)

    @property
    def event_type(self):
        """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
        return self._event_type[0]

    @event_type.setter
    def event_type(self, value: np.uint32) -> None:
        self._event_type[0] = value

    @property
    def event_version(self):
        """Event format version of the DAQ"""
        return self._event_version[0]

    @event_version.setter
    def event_version(self, value: np.uint32) -> None:
        self._event_version[0] = value

    @property
    def site(self):
        """Site name"""
        return str(self._site)

    @site.setter
    def site(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._site.string.assign(value)

    @property
    def site_layout(self):
        """Site layout"""
        return str(self._site_layout)

    @site_layout.setter
    def site_layout(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site_layout {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._site_layout.string.assign(value)

    # @property
    # def site_long(self):
    #     """Site longitude"""
    #     return np.array(self._site_long)
    #
    # @site_long.setter
    # def site_long(self, value):
    #     self._site_long = np.array(value).astype(np.float32)
    #     self._tree.SetBranchAddress("site_long", self._site_long)
    #
    # @property
    # def site_lat(self):
    #     """Site latitude"""
    #     return np.array(self._site_lat)
    #
    # @site_lat.setter
    # def site_lat(self, value):
    #     self._site_lat = np.array(value).astype(np.float32)
    #     self._tree.SetBranchAddress("site_lat", self._site_lat)

    @property
    def origin_geoid(self):
        """Origin of the coordinate system used for the array"""
        return np.array(self._origin_geoid)

    @origin_geoid.setter
    def origin_geoid(self, value):
        self._origin_geoid = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("origin_geoid", self._origin_geoid)

    @property
    def du_id(self):
        """Detector unit (antenna) ID"""
        return self._du_id

    @du_id.setter
    def du_id(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("int")):
            self._du_id._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of ints required."
            )

    @property
    def du_geoid(self):
        """Detector unit (antenna) (lat,lon,alt) position"""
        return self._du_geoid

    @du_geoid.setter
    def du_geoid(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_geoid.clear()
            self._du_geoid += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._du_geoid._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_geoid {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )

    @property
    def du_xyz(self):
        """Detector unit (antenna) (x,y,z) position in site's referential"""
        return self._du_xyz

    @du_xyz.setter
    def du_xyz(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_xyz.clear()
            self._du_xyz += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._du_xyz._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_xyz {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )

    @property
    def du_type(self):
        """Detector unit type"""
        return self._du_type

    @du_type.setter
    def du_type(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_type.clear()
            self._du_type += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("string")):
            self._du_type._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_type {type(value)}. Either a list, an array or a ROOT.vector of string required."
            )

    @property
    def du_tilt(self):
        """Detector unit (antenna) angular tilt"""
        return self._du_tilt

    @du_tilt.setter
    def du_tilt(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_tilt.clear()
            self._du_tilt += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._du_tilt._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_tilt {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )

    @property
    def du_ground_tilt(self):
        """Angular tilt of the ground at the antenna"""
        return self._du_ground_tilt

    @du_ground_tilt.setter
    def du_ground_tilt(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_ground_tilt.clear()
            self._du_ground_tilt += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._du_ground_tilt._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_ground_tilt {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )

    @property
    def du_nut(self):
        """Detector unit (antenna) nut ID"""
        return self._du_nut

    @du_nut.setter
    def du_nut(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_nut.clear()
            self._du_nut += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("int")):
            self._du_nut._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_nut {type(value)}. Either a list, an array or a ROOT.vector of int required."
            )

    @property
    def du_feb(self):
        """Detector unit (antenna) FrontEnd Board ID"""
        return self._du_feb

    @du_feb.setter
    def du_feb(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_feb.clear()
            self._du_feb += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("int")):
            self._du_feb._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_feb {type(value)}. Either a list, an array or a ROOT.vector of int required."
            )

    @property
    def t_bin_size(self):
        """Time bin size in ns (for hardware, computed as 1/adc_sampling_frequency)"""
        return self._t_bin_size

    @t_bin_size.setter
    def t_bin_size(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._t_bin_size.clear()
            self._t_bin_size += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._t_bin_size._vector = value
        else:
            raise ValueError(
                f"Incorrect type for t_bin_size {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )


## General info on the voltage common to all events.
@dataclass
class TRunVoltage(MotherRunTree):
    """General info on the voltage common to all events."""

    _type: str = "runvoltage"

    _tree_name: str = "trunvoltage"

    ## Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual).
    _digi_ctrl: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<unsigned short>")
    )
    ## Firmware version
    _firmware_version: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned short")
    )
    ## Nominal trace length in units of samples
    _trace_length: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<int>")
    )
    ## Nominal trigger position in the trace in unit of samples
    _trigger_position: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<int>")
    )
    ## ADC sampling frequency in MHz
    _adc_sampling_frequency: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned short")
    )
    ## ADC sampling resolution in bits
    _adc_sampling_resolution: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned short")
    )
    ## ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?
    _adc_input_channels: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned short")
    )
    ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?
    _adc_enabled_channels: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned short")
    )
    ## Value of the Variable gain amplification on the board
    _gain: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<int>")
    )
    ## Conversion factor from bits to V for ADC
    _adc_conversion: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<float>")
    )
    ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    _digi_prepost_trig_windows: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<unsigned short>")
    )
    ## Channel x properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties_x: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<unsigned short>")
    )
    ## Channel y properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties_y: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<unsigned short>")
    )
    ## Channel z properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties_z: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<unsigned short>")
    )
    ## Channel x trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings_x: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<unsigned short>")
    )
    ## Channel y trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings_y: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<unsigned short>")
    )
    ## Channel z trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings_z: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<unsigned short>")
    )

    @property
    def digi_ctrl(self):
        """Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual). ToDo: Decode?"""
        return self._digi_ctrl

    @digi_ctrl.setter
    def digi_ctrl(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._digi_ctrl.clear()
            self._digi_ctrl += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._digi_ctrl._vector = value
        else:
            raise ValueError(
                f"Incorrect type for digi_ctrl {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def firmware_version(self):
        """Firmware version"""
        return self._firmware_version[0]

    @firmware_version.setter
    def firmware_version(self, value: np.uint16) -> None:
        self._firmware_version[0] = value

    @property
    def trace_length(self):
        """Nominal trace length in units of samples"""
        return np.array(self._trace_length)

    @trace_length.setter
    def trace_length(self, value):
        set_vector_of_vectors(value, "vector<int>", self._trace_length, "trace_length")

    @property
    def trigger_position(self):
        """Nominal trigger position in the trace in unit of samples"""
        return np.array(self._trigger_position)

    @trigger_position.setter
    def trigger_position(self, value):
        set_vector_of_vectors(value, "vector<int>", self._trigger_position, "trigger_position")

    @property
    def adc_sampling_frequency(self):
        """ADC sampling frequency in MHz"""
        return self._adc_sampling_frequency

    @adc_sampling_frequency.setter
    def adc_sampling_frequency(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_sampling_frequency.clear()
            self._adc_sampling_frequency += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_frequency._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_sampling_frequency {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_sampling_resolution(self):
        """ADC sampling resolution in bits"""
        return self._adc_sampling_resolution

    @adc_sampling_resolution.setter
    def adc_sampling_resolution(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_sampling_resolution.clear()
            self._adc_sampling_resolution += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_resolution._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_sampling_resolution {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_input_channels(self):
        """ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?"""
        return self._adc_input_channels

    @adc_input_channels.setter
    def adc_input_channels(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_input_channels.clear()
            self._adc_input_channels += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_input_channels._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_input_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_enabled_channels(self):
        """ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?"""
        return self._adc_enabled_channels

    @adc_enabled_channels.setter
    def adc_enabled_channels(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_enabled_channels.clear()
            self._adc_enabled_channels += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_enabled_channels._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_enabled_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gain(self):
        """Value of the Variable gain amplification on the board"""
        return np.array(self._gain)

    @gain.setter
    def gain(self, value):
        set_vector_of_vectors(value, "vector<int>", self._gain, "gain")

    @property
    def adc_conversion(self):
        """Conversion factor from bits to V for ADC"""
        return np.array(self._adc_conversion)

    @adc_conversion.setter
    def adc_conversion(self, value):
        set_vector_of_vectors(value, "vector<int>", self._adc_conversion, "adc_conversion")

    @property
    def digi_prepost_trig_windows(self):
        """Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?"""
        return self._digi_prepost_trig_windows

    @digi_prepost_trig_windows.setter
    def digi_prepost_trig_windows(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._digi_prepost_trig_windows.clear()
            self._digi_prepost_trig_windows += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._digi_prepost_trig_windows._vector = value
        else:
            raise ValueError(
                f"Incorrect type for digi_prepost_trig_windows {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_properties_x(self):
        """Channel x properties - described in Channel property parameters in the manual. ToDo: Decode?"""
        return self._channel_properties_x

    @channel_properties_x.setter
    def channel_properties_x(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._channel_properties_x.clear()
            self._channel_properties_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties_x._vector = value
        else:
            raise ValueError(
                f"Incorrect type for channel_properties_x {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_properties_y(self):
        """Channel y properties - described in Channel property parameters in the manual. ToDo: Decode?"""
        return self._channel_properties_y

    @channel_properties_y.setter
    def channel_properties_y(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._channel_properties_y.clear()
            self._channel_properties_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties_y._vector = value
        else:
            raise ValueError(
                f"Incorrect type for channel_properties_y {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_properties_z(self):
        """Channel z properties - described in Channel property parameters in the manual. ToDo: Decode?"""
        return self._channel_properties_z

    @channel_properties_z.setter
    def channel_properties_z(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._channel_properties_z.clear()
            self._channel_properties_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties_z._vector = value
        else:
            raise ValueError(
                f"Incorrect type for channel_properties_z {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_trig_settings_x(self):
        """Channel x trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
        return self._channel_trig_settings_x

    @channel_trig_settings_x.setter
    def channel_trig_settings_x(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._channel_trig_settings_x.clear()
            self._channel_trig_settings_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings_x._vector = value
        else:
            raise ValueError(
                f"Incorrect type for channel_trig_settings_x {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_trig_settings_y(self):
        """Channel y trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
        return self._channel_trig_settings_y

    @channel_trig_settings_y.setter
    def channel_trig_settings_y(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._channel_trig_settings_y.clear()
            self._channel_trig_settings_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings_y._vector = value
        else:
            raise ValueError(
                f"Incorrect type for channel_trig_settings_y {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_trig_settings_z(self):
        """Channel z trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
        return self._channel_trig_settings_z

    @channel_trig_settings_z.setter
    def channel_trig_settings_z(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._channel_trig_settings_z.clear()
            self._channel_trig_settings_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings_z._vector = value
        else:
            raise ValueError(
                f"Incorrect type for channel_trig_settings_z {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )


@dataclass
## The class for storing ADC traces and associated values for each event
class TADC(MotherEventTree):
    """The class for storing ADC traces and associated values for each event"""

    _type: str = "adc"

    _tree_name: str = "tadc"

    ## Common for the whole event
    ## Event size
    _event_size: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Event in the run number
    _t3_number: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## First detector unit that triggered in the event
    _first_du: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Unix time corresponding to the GPS seconds of the first triggered station
    _time_seconds: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    _time_nanoseconds: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    _event_type: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Event format version of the DAQ
    _event_version: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))

    ## Specific for each Detector Unit
    ## The T3 trigger number
    _event_id: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Detector unit (antenna) ID
    _du_id: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Unix time of the trigger for this DU
    _du_seconds: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Nanoseconds of the trigger for this DU
    _du_nanoseconds: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    # ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    _trigger_position: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned short")
    )
    ## Same as event_type, but event_type could consist of different triggered DUs
    _trigger_flag: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Atmospheric temperature (read via I2C)
    _atm_temperature: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Atmospheric pressure
    _atm_pressure: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Atmospheric humidity
    _atm_humidity: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Acceleration of the antenna in X
    _acceleration_x: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Acceleration of the antenna in Y
    _acceleration_y: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Acceleration of the antenna in Z
    _acceleration_z: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Battery voltage
    _battery_level: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Firmware version
    _firmware_version: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned short")
    )
    ## ADC sampling frequency in MHz
    _adc_sampling_frequency: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned short")
    )
    ## ADC sampling resolution in bits
    _adc_sampling_resolution: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned short")
    )

    # ## ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?
    # _adc_input_channels: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )

    adc_input_channels_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """ADC input channels"""

    # ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?
    # _adc_enabled_channels: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )

    adc_enabled_channels_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """ADC enabled channels"""

    # ## ADC samples callected in all channels
    # _adc_samples_count_total: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC samples callected in channel 0
    # _adc_samples_count_channel0: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC samples callected in channel 1
    # _adc_samples_count_channel1: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC samples callected in channel 2
    # _adc_samples_count_channel2: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC samples callected in channel 3
    # _adc_samples_count_channel3: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )

    adc_samples_count_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    # ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    # _trigger_pattern: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))

    trigger_pattern_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    trigger_pattern_ch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_notch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_redch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_ch2_ch3: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_calibration: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_10s: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_external_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))

    ## Trigger rate - the number of triggers recorded in the second preceding the event
    _trigger_rate: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    _clock_tick: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Clock ticks per second
    _clock_ticks_per_second: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned int")
    )
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    _gps_offset: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    ## GPS leap second
    _gps_leap_second: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## GPS status
    _gps_status: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## GPS alarms
    _gps_alarms: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## GPS warnings
    _gps_warnings: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## GPS time
    _gps_time: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Longitude
    _gps_long: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned long long"))
    ## Latitude
    _gps_lat: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned long long"))
    ## Altitude
    _gps_alt: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned long long"))
    ## GPS temperature
    _gps_temp: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    # ## Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual). ToDo: Decode?
    # _digi_ctrl: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )

    # Digital control register
    enable_auto_reset_timeout: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    force_firmware_reset: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_filter_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    enable_1PPS: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_DAQ: StdVectorListDesc = field(default=StdVectorListDesc("bool"))

    # Trigger enable mask register
    enable_trigger_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    enable_trigger_ch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_notch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_redch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_ch2_ch3: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_calibration: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_10s: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_external_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))

    # Test pulse rate divider and channel readout enable
    enable_readout_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    fire_single_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    test_pulse_rate_divider: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))

    # Common coincidence readout time window
    common_coincidence_time: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))

    # Input selector for readout channel
    selector_readout_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))

    # ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    # _digi_prepost_trig_windows: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )

    pre_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    post_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    # ## Channel 0 properties - described in Channel property parameters in the manual. ToDo: Decode?
    # _channel_properties0: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel 1 properties - described in Channel property parameters in the manual. ToDo: Decode?
    # _channel_properties1: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel 2 properties - described in Channel property parameters in the manual. ToDo: Decode?
    # _channel_properties2: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel 3 properties - described in Channel property parameters in the manual. ToDo: Decode?
    # _channel_properties3: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )

    gain_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    integration_time_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    offset_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    base_maximum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    base_minimum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    # ## Channel 0 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    # _channel_trig_settings0: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel 1 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    # _channel_trig_settings1: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel 2 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    # _channel_trig_settings2: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel 3 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    # _channel_trig_settings3: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )

    signal_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    noise_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    tper_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    tprev_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    ncmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    tcmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    qmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    ncmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    qmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))

    ## ?? What is it? Some kind of the adc trace offset?
    _ioff: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")
    # ## ADC trace 0
    # _trace_0: StdVectorList = field(default_factory=lambda: StdVectorList("vector<short>"))
    # ## ADC trace 1
    # _trace_1: StdVectorList = field(default_factory=lambda: StdVectorList("vector<short>"))
    # ## ADC trace 2
    # _trace_2: StdVectorList = field(default_factory=lambda: StdVectorList("vector<short>"))
    # ## ADC trace 3
    # _trace_3: StdVectorList = field(default_factory=lambda: StdVectorList("vector<short>"))
    ## ADC traces for channels (0,1,2,3)
    _trace_ch: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<short>>"))

    # def __post_init__(self):
    #     super().__post_init__()
    #
    #     if self._tree.GetName() == "":
    #         self._tree.SetName(self._tree_name)
    #     if self._tree.GetTitle() == "":
    #         self._tree.SetTitle(self._tree_name)
    #
    #     self.create_branches()

    @property
    def event_size(self):
        """Event size"""
        return self._event_size[0]

    @event_size.setter
    def event_size(self, value: np.uint32) -> None:
        self._event_size[0] = value

    @property
    def t3_number(self):
        """Event in the run number"""
        return self._t3_number[0]

    @t3_number.setter
    def t3_number(self, value: np.uint32) -> None:
        self._t3_number[0] = value

    @property
    def first_du(self):
        """First detector unit that triggered in the event"""
        return self._first_du[0]

    @first_du.setter
    def first_du(self, value: np.uint32) -> None:
        self._first_du[0] = value

    @property
    def time_seconds(self):
        """Unix time corresponding to the GPS seconds of the first triggered station"""
        return self._time_seconds[0]

    @time_seconds.setter
    def time_seconds(self, value: np.uint32) -> None:
        self._time_seconds[0] = value

    @property
    def time_nanoseconds(self):
        """GPS nanoseconds corresponding to the trigger of the first triggered station"""
        return self._time_nanoseconds[0]

    @time_nanoseconds.setter
    def time_nanoseconds(self, value: np.uint32) -> None:
        self._time_nanoseconds[0] = value

    @property
    def event_type(self):
        """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
        return self._event_type[0]

    @event_type.setter
    def event_type(self, value: np.uint32) -> None:
        self._event_type[0] = value

    @property
    def event_version(self):
        """Event format version of the DAQ"""
        return self._event_version[0]

    @event_version.setter
    def event_version(self, value: np.uint32) -> None:
        self._event_version[0] = value

    @property
    def du_count(self):
        """Number of detector units in the event - basically the antennas count"""
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

    @property
    def event_id(self):
        """The T3 trigger number"""
        return self._event_id

    @event_id.setter
    def event_id(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._event_id.clear()
            self._event_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._event_id._vector = value
        else:
            raise ValueError(
                f"Incorrect type for event_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def du_id(self):
        """Detector unit (antenna) ID"""
        return self._du_id

    @du_id.setter
    def du_id(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._du_id._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def du_seconds(self):
        """Unix time of the trigger for this DU"""
        return self._du_seconds

    @du_seconds.setter
    def du_seconds(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_seconds.clear()
            self._du_seconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_seconds._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def du_nanoseconds(self):
        """Nanoseconds of the trigger for this DU"""
        return self._du_nanoseconds

    @du_nanoseconds.setter
    def du_nanoseconds(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_nanoseconds.clear()
            self._du_nanoseconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_nanoseconds._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def trigger_position(self):
        """Trigger position in the trace (trigger start = nanoseconds - 2*sample number)"""
        return self._trigger_position

    @trigger_position.setter
    def trigger_position(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trigger_position.clear()
            self._trigger_position += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_position._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trigger_position {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def trigger_flag(self):
        """Same as event_type, but event_type could consist of different triggered DUs"""
        return self._trigger_flag

    @trigger_flag.setter
    def trigger_flag(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trigger_flag.clear()
            self._trigger_flag += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_flag._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trigger_flag {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def atm_temperature(self):
        """Atmospheric temperature (read via I2C)"""
        return self._atm_temperature

    @atm_temperature.setter
    def atm_temperature(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._atm_temperature.clear()
            self._atm_temperature += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._atm_temperature._vector = value
        else:
            raise ValueError(
                f"Incorrect type for atm_temperature {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def atm_pressure(self):
        """Atmospheric pressure"""
        return self._atm_pressure

    @atm_pressure.setter
    def atm_pressure(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._atm_pressure.clear()
            self._atm_pressure += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._atm_pressure._vector = value
        else:
            raise ValueError(
                f"Incorrect type for atm_pressure {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def atm_humidity(self):
        """Atmospheric humidity"""
        return self._atm_humidity

    @atm_humidity.setter
    def atm_humidity(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._atm_humidity.clear()
            self._atm_humidity += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._atm_humidity._vector = value
        else:
            raise ValueError(
                f"Incorrect type for atm_humidity {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def acceleration_x(self):
        """Acceleration of the antenna in X"""
        return self._acceleration_x

    @acceleration_x.setter
    def acceleration_x(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._acceleration_x.clear()
            self._acceleration_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._acceleration_x._vector = value
        else:
            raise ValueError(
                f"Incorrect type for acceleration_x {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def acceleration_y(self):
        """Acceleration of the antenna in Y"""
        return self._acceleration_y

    @acceleration_y.setter
    def acceleration_y(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._acceleration_y.clear()
            self._acceleration_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._acceleration_y._vector = value
        else:
            raise ValueError(
                f"Incorrect type for acceleration_y {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def acceleration_z(self):
        """Acceleration of the antenna in Z"""
        return self._acceleration_z

    @acceleration_z.setter
    def acceleration_z(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._acceleration_z.clear()
            self._acceleration_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._acceleration_z._vector = value
        else:
            raise ValueError(
                f"Incorrect type for acceleration_z {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def battery_level(self):
        """Battery voltage"""
        return self._battery_level

    @battery_level.setter
    def battery_level(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._battery_level.clear()
            self._battery_level += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._battery_level._vector = value
        else:
            raise ValueError(
                f"Incorrect type for battery_level {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def firmware_version(self):
        """Firmware version"""
        return self._firmware_version

    @firmware_version.setter
    def firmware_version(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._firmware_version.clear()
            self._firmware_version += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._firmware_version._vector = value
        else:
            raise ValueError(
                f"Incorrect type for firmware_version {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_sampling_frequency(self):
        """ADC sampling frequency in MHz"""
        return self._adc_sampling_frequency

    @adc_sampling_frequency.setter
    def adc_sampling_frequency(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_sampling_frequency.clear()
            self._adc_sampling_frequency += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_frequency._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_sampling_frequency {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_sampling_resolution(self):
        """ADC sampling resolution in bits"""
        return self._adc_sampling_resolution

    @adc_sampling_resolution.setter
    def adc_sampling_resolution(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_sampling_resolution.clear()
            self._adc_sampling_resolution += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_resolution._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_sampling_resolution {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    # @property
    # def adc_input_channels(self):
    #     """ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?"""
    #     return self._adc_input_channels
    #
    # @adc_input_channels.setter
    # def adc_input_channels(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_input_channels.clear()
    #         self._adc_input_channels += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_input_channels._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_input_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )

    # @property
    # def adc_enabled_channels(self):
    #     """ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?"""
    #     return self._adc_enabled_channels
    #
    # @adc_enabled_channels.setter
    # def adc_enabled_channels(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_enabled_channels.clear()
    #         self._adc_enabled_channels += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_enabled_channels._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_enabled_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )

    @property
    def adc_samples_count_total(self):
        """ADC samples callected in all channels"""
        return self._adc_samples_count_total

    @adc_samples_count_total.setter
    def adc_samples_count_total(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_samples_count_total.clear()
            self._adc_samples_count_total += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_total._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_samples_count_total {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_samples_count_channel0(self):
        """ADC samples callected in channel 0"""
        return self._adc_samples_count_channel0

    @adc_samples_count_channel0.setter
    def adc_samples_count_channel0(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_samples_count_channel0.clear()
            self._adc_samples_count_channel0 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel0._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_samples_count_channel0 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_samples_count_channel1(self):
        """ADC samples callected in channel 1"""
        return self._adc_samples_count_channel1

    @adc_samples_count_channel1.setter
    def adc_samples_count_channel1(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_samples_count_channel1.clear()
            self._adc_samples_count_channel1 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel1._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_samples_count_channel1 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_samples_count_channel2(self):
        """ADC samples callected in channel 2"""
        return self._adc_samples_count_channel2

    @adc_samples_count_channel2.setter
    def adc_samples_count_channel2(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_samples_count_channel2.clear()
            self._adc_samples_count_channel2 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel2._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_samples_count_channel2 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_samples_count_channel3(self):
        """ADC samples callected in channel 3"""
        return self._adc_samples_count_channel3

    @adc_samples_count_channel3.setter
    def adc_samples_count_channel3(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_samples_count_channel3.clear()
            self._adc_samples_count_channel3 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel3._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_samples_count_channel3 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    # @property
    # def trigger_pattern(self):
    #     """Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?"""
    #     return self._trigger_pattern
    #
    # @trigger_pattern.setter
    # def trigger_pattern(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trigger_pattern.clear()
    #         self._trigger_pattern += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._trigger_pattern._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trigger_pattern {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )

    @property
    def trigger_rate(self):
        """Trigger rate - the number of triggers recorded in the second preceding the event"""
        return self._trigger_rate

    @trigger_rate.setter
    def trigger_rate(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trigger_rate.clear()
            self._trigger_rate += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_rate._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trigger_rate {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def clock_tick(self):
        """Clock tick at which the event was triggered (used to calculate the trigger time)"""
        return self._clock_tick

    @clock_tick.setter
    def clock_tick(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._clock_tick.clear()
            self._clock_tick += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._clock_tick._vector = value
        else:
            raise ValueError(
                f"Incorrect type for clock_tick {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def clock_ticks_per_second(self):
        """Clock ticks per second"""
        return self._clock_ticks_per_second

    @clock_ticks_per_second.setter
    def clock_ticks_per_second(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._clock_ticks_per_second.clear()
            self._clock_ticks_per_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._clock_ticks_per_second._vector = value
        else:
            raise ValueError(
                f"Incorrect type for clock_ticks_per_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def gps_offset(self):
        """GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?"""
        return self._gps_offset

    @gps_offset.setter
    def gps_offset(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_offset.clear()
            self._gps_offset += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_offset._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_offset {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    @property
    def gps_leap_second(self):
        """GPS leap second"""
        return self._gps_leap_second

    @gps_leap_second.setter
    def gps_leap_second(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_leap_second.clear()
            self._gps_leap_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_leap_second._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_leap_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_status(self):
        """GPS status"""
        return self._gps_status

    @gps_status.setter
    def gps_status(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_status.clear()
            self._gps_status += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_status._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_status {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_alarms(self):
        """GPS alarms"""
        return self._gps_alarms

    @gps_alarms.setter
    def gps_alarms(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_alarms.clear()
            self._gps_alarms += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_alarms._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_alarms {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_warnings(self):
        """GPS warnings"""
        return self._gps_warnings

    @gps_warnings.setter
    def gps_warnings(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_warnings.clear()
            self._gps_warnings += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_warnings._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_warnings {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_time(self):
        """GPS time"""
        return self._gps_time

    @gps_time.setter
    def gps_time(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_time.clear()
            self._gps_time += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._gps_time._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_time {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def gps_long(self):
        """Longitude"""
        return self._gps_long

    @gps_long.setter
    def gps_long(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_long.clear()
            self._gps_long += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned long long")):
            self._gps_long._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_long {type(value)}. Either a list, an array or a ROOT.vector of unsigned long longs required."
            )

    @property
    def gps_lat(self):
        """Latitude"""
        return self._gps_lat

    @gps_lat.setter
    def gps_lat(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_lat.clear()
            self._gps_lat += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned long long")):
            self._gps_lat._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_lat {type(value)}. Either a list, an array or a ROOT.vector of unsigned long longs required."
            )

    @property
    def gps_alt(self):
        """Altitude"""
        return self._gps_alt

    @gps_alt.setter
    def gps_alt(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_alt.clear()
            self._gps_alt += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned long long")):
            self._gps_alt._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_alt {type(value)}. Either a list, an array or a ROOT.vector of unsigned long longs required."
            )

    @property
    def gps_temp(self):
        """GPS temperature"""
        return self._gps_temp

    @gps_temp.setter
    def gps_temp(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_temp.clear()
            self._gps_temp += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._gps_temp._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_temp {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    # @property
    # def digi_ctrl(self):
    #     """Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual). ToDo: Decode?"""
    #     return self._digi_ctrl
    #
    # @digi_ctrl.setter
    # def digi_ctrl(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._digi_ctrl.clear()
    #         self._digi_ctrl += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._digi_ctrl._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for digi_ctrl {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )

    # @property
    # def digi_prepost_trig_windows(self):
    #     """Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?"""
    #     return self._digi_prepost_trig_windows
    #
    # @digi_prepost_trig_windows.setter
    # def digi_prepost_trig_windows(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._digi_prepost_trig_windows.clear()
    #         self._digi_prepost_trig_windows += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._digi_prepost_trig_windows._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for digi_prepost_trig_windows {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )

    # @property
    # def channel_properties0(self):
    #     """Channel 0 properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    #     return self._channel_properties0
    #
    # @channel_properties0.setter
    # def channel_properties0(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_properties0.clear()
    #         self._channel_properties0 += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_properties0._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_properties0 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_properties1(self):
    #     """Channel 1 properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    #     return self._channel_properties1
    #
    # @channel_properties1.setter
    # def channel_properties1(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_properties1.clear()
    #         self._channel_properties1 += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_properties1._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_properties1 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_properties2(self):
    #     """Channel 2 properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    #     return self._channel_properties2
    #
    # @channel_properties2.setter
    # def channel_properties2(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_properties2.clear()
    #         self._channel_properties2 += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_properties2._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_properties2 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_properties3(self):
    #     """Channel 3 properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    #     return self._channel_properties3
    #
    # @channel_properties3.setter
    # def channel_properties3(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_properties3.clear()
    #         self._channel_properties3 += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_properties3._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_properties3 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_trig_settings0(self):
    #     """Channel 0 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    #     return self._channel_trig_settings0
    #
    # @channel_trig_settings0.setter
    # def channel_trig_settings0(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_trig_settings0.clear()
    #         self._channel_trig_settings0 += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_trig_settings0._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_trig_settings0 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_trig_settings1(self):
    #     """Channel 1 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    #     return self._channel_trig_settings1
    #
    # @channel_trig_settings1.setter
    # def channel_trig_settings1(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_trig_settings1.clear()
    #         self._channel_trig_settings1 += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_trig_settings1._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_trig_settings1 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_trig_settings2(self):
    #     """Channel 2 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    #     return self._channel_trig_settings2
    #
    # @channel_trig_settings2.setter
    # def channel_trig_settings2(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_trig_settings2.clear()
    #         self._channel_trig_settings2 += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_trig_settings2._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_trig_settings2 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_trig_settings3(self):
    #     """Channel 3 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    #     return self._channel_trig_settings3
    #
    # @channel_trig_settings3.setter
    # def channel_trig_settings3(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_trig_settings3.clear()
    #         self._channel_trig_settings3 += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_trig_settings3._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_trig_settings3 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )

    @property
    def ioff(self):
        """?? What is it? Some kind of the adc trace offset?"""
        return self._ioff

    @ioff.setter
    def ioff(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._ioff.clear()
            self._ioff += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._ioff._vector._vector = value
        else:
            raise ValueError(
                f"Incorrect type for ioff {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    # @property
    # def trace_0(self):
    #     """ADC trace 0"""
    #     return self._trace_0
    #
    # @trace_0.setter
    # def trace_0(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_0.clear()
    #         self._trace_0 += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<short>")):
    #         # With vectors, I think the address is assigned, so in principle the below is needed only on the first setting of the branch
    #         self._trace_0._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_0 {type(value)}. Either a list, an array or a ROOT.vector of vector<short> required."
    #         )
    #
    # @property
    # def trace_1(self):
    #     """ADC trace 1"""
    #     return self._trace_1
    #
    # @trace_1.setter
    # def trace_1(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_1.clear()
    #         self._trace_1 += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<short>")):
    #         self._trace_1._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_1 {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
    #         )
    #
    # @property
    # def trace_2(self):
    #     """ADC trace 2"""
    #     return self._trace_2
    #
    # @trace_2.setter
    # def trace_2(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_2.clear()
    #         self._trace_2 += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<short>")):
    #         self._trace_2._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_2 {type(value)}. Either a list, an array or a ROOT.vector of vector<short> required."
    #         )
    #
    # @property
    # def trace_3(self):
    #     """ADC trace 3"""
    #     return self._trace_3
    #
    # @trace_3.setter
    # def trace_3(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_3.clear()
    #         self._trace_3 += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<short>")):
    #         self._trace_3._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_3 {type(value)}. Either a list, an array or a ROOT.vector of vector<short> required."
    #         )

    @property
    def trace_ch(self):
        """ADC traces for channels (0,1,2,3)"""
        return self._trace_ch

    @trace_ch.setter
    def trace_ch(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trace_ch._vector.clear()
            self._trace_ch += value

        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<vector<short>>")):
            self._trace_ch._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trace_ch {type(value)}. Either a list, an array or a ROOT.vector of vector<vector<short>> required."
            )


@dataclass
## The class for storing voltage traces and associated values for each event
class TRawVoltage(MotherEventTree):
    """The class for storing voltage traces and associated values at ADC input level for each event. Derived from TADC but in human readable format and physics units."""

    _type: str = "rawvoltage"

    _tree_name: str = "trawvoltage"

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
    _event_size: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    # ## Event in the run number
    # _t3_number: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## First detector unit that triggered in the event
    _first_du: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Unix time corresponding to the GPS seconds of the trigger
    _time_seconds: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    _time_nanoseconds: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    # ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    # _event_type: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    # ## Event format version of the DAQ
    # _event_version: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))

    ## Specific for each Detector Unit
    # ## The T3 trigger number
    # _event_id: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Detector unit (antenna) ID
    _du_id: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Unix time of the trigger for this DU
    _du_seconds: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Nanoseconds of the trigger for this DU
    _du_nanoseconds: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Unix time of the start of the trace for this DU
    # _du_t0_seconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Nanoseconds of the start of the trace for this DU
    # _du_t0_nanoseconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    # ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    # _trigger_position: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    ## Same as event_type, but event_type could consist of different triggered DUs
    _trigger_flag: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Atmospheric temperature (read via I2C)
    _atm_temperature: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    ## Atmospheric pressure
    _atm_pressure: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    ## Atmospheric humidity
    _atm_humidity: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    ## Acceleration of the antenna in (x,y,z) in m/s2
    _du_acceleration: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Acceleration of the antenna in X
    # _acceleration_x: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    # ## Acceleration of the antenna in Y
    # _acceleration_y: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    # ## Acceleration of the antenna in Z
    # _acceleration_z: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    ## Battery voltage
    _battery_level: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    # ## Firmware version
    # _firmware_version: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC sampling frequency in MHz
    # _adc_sampling_frequency: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC sampling resolution in bits
    # _adc_sampling_resolution: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?
    # _adc_input_channels: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?
    # _adc_enabled_channels: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC samples callected in all channels
    # _adc_samples_count_total: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC samples callected in channel x
    # _adc_samples_count_channel_x: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC samples callected in channel y
    # _adc_samples_count_channel_y: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    # ## ADC samples callected in channel z
    # _adc_samples_count_channel_z: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("unsigned short")
    # )
    ## ADC samples callected in channels (0,1,2,3)
    _adc_samples_count_channel: StdVectorList = field(
        default_factory=lambda: StdVectorList("vector<unsigned short>")
    )
    ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    _trigger_pattern: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    _trigger_rate: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    _clock_tick: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Clock ticks per second
    _clock_ticks_per_second: StdVectorList = field(
        default_factory=lambda: StdVectorList("unsigned int")
    )
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    _gps_offset: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    ## GPS leap second
    _gps_leap_second: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## GPS status
    _gps_status: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## GPS alarms
    _gps_alarms: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## GPS warnings
    _gps_warnings: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## GPS time
    _gps_time: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Longitude
    _gps_long: StdVectorList = field(default_factory=lambda: StdVectorList("double"))
    ## Latitude
    _gps_lat: StdVectorList = field(default_factory=lambda: StdVectorList("double"))
    ## Altitude
    _gps_alt: StdVectorList = field(default_factory=lambda: StdVectorList("double"))
    ## GPS temperature
    _gps_temp: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    # ## X position in site's referential
    # _pos_x: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    # ## Y position in site's referential
    # _pos_y: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    # ## Z position in site's referential
    # _pos_z: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    # ## Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual). ToDo: Decode?
    # _digi_ctrl: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    # _digi_prepost_trig_windows: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel x properties - described in Channel property parameters in the manual. ToDo: Decode?
    # _channel_properties_x: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel y properties - described in Channel property parameters in the manual. ToDo: Decode?
    # _channel_properties_y: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel z properties - described in Channel property parameters in the manual. ToDo: Decode?
    # _channel_properties_z: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel x trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    # _channel_trig_settings_x: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel y trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    # _channel_trig_settings_y: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    # ## Channel z trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    # _channel_trig_settings_z: StdVectorList = field(
    #     default_factory=lambda: StdVectorList("vector<unsigned short>")
    # )
    ## ?? What is it? Some kind of the adc trace offset?
    _ioff: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))

    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")

    # ## Voltage trace in X direction
    # _trace_x: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Voltage trace in Y direction
    # _trace_y: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Voltage trace in Z direction
    # _trace_z: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Voltage trace in channel 0
    # _trace_0: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Voltage trace in channel 1
    # _trace_1: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Voltage trace in channel 2
    # _trace_2: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Voltage trace in channel 3
    # _trace_3: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))

    ## Voltage traces for channels 1,2,3,4 in muV
    _trace_ch: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<float>>"))

    # _trace_ch: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<Float32_t>>"))

    # def __post_init__(self):
    #     super().__post_init__()
    #
    #     if self._tree.GetName() == "":
    #         self._tree.SetName(self._tree_name)
    #     if self._tree.GetTitle() == "":
    #         self._tree.SetTitle(self._tree_name)
    #
    #     self.create_branches()
    #     logger.debug(f'Create TRawVoltage object')

    @property
    def event_size(self):
        """Event size"""
        return self._event_size[0]

    @event_size.setter
    def event_size(self, value: np.uint32) -> None:
        self._event_size[0] = value

    # @property
    # def t3_number(self):
    #     """Event in the run number"""
    #     return self._t3_number[0]
    #
    # @t3_number.setter
    # def t3_number(self, value: np.uint32) -> None:
    #     self._t3_number[0] = value

    @property
    def first_du(self):
        """First detector unit that triggered in the event"""
        return self._first_du[0]

    @first_du.setter
    def first_du(self, value: np.uint32) -> None:
        self._first_du[0] = value

    @property
    def time_seconds(self):
        """Unix time corresponding to the GPS seconds of the trigger"""
        return self._time_seconds[0]

    @time_seconds.setter
    def time_seconds(self, value: np.uint32) -> None:
        self._time_seconds[0] = value

    @property
    def time_nanoseconds(self):
        """GPS nanoseconds corresponding to the trigger of the first triggered station"""
        return self._time_nanoseconds[0]

    @time_nanoseconds.setter
    def time_nanoseconds(self, value: np.uint32) -> None:
        self._time_nanoseconds[0] = value

    # @property
    # def event_type(self):
    #     """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
    #     return self._event_type[0]
    #
    # @event_type.setter
    # def event_type(self, value: np.uint32) -> None:
    #     self._event_type[0] = value
    #
    # @property
    # def event_version(self):
    #     """Event format version of the DAQ"""
    #     return self._event_version[0]
    #
    # @event_version.setter
    # def event_version(self, value: np.uint32) -> None:
    #     self._event_version[0] = value

    @property
    def du_count(self):
        """Number of detector units in the event - basically the antennas count"""
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

    # @property
    # def event_id(self):
    #     """The T3 trigger number"""
    #     return self._event_id
    #
    # @event_id.setter
    # def event_id(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._event_id.clear()
    #         self._event_id += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._event_id._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for event_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )

    @property
    def du_id(self):
        """Detector unit (antenna) ID"""
        return self._du_id

    @du_id.setter
    def du_id(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._du_id._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def du_seconds(self):
        """Unix time of the trigger for this DU"""
        return self._du_seconds

    @du_seconds.setter
    def du_seconds(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_seconds.clear()
            self._du_seconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_seconds._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def du_nanoseconds(self):
        """Nanoseconds of the trigger for this DU"""
        return self._du_nanoseconds

    @du_nanoseconds.setter
    def du_nanoseconds(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_nanoseconds.clear()
            self._du_nanoseconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_nanoseconds._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    # @property
    # def du_t0_seconds(self):
    #     return self._du_t0_seconds
    #
    # @du_t0_seconds.setter
    # def du_t0_seconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray) or isinstance(value, StdVectorList):
    #         # Clear the vector before setting
    #         self._du_t0_seconds.clear()
    #         self._du_t0_seconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_seconds._vector = value
    #     else:
    #         raise ValueError(f"Incorrect type for du_t0_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")
    #
    # @property
    # def du_t0_nanoseconds(self):
    #     return self._du_t0_nanoseconds
    #
    # @du_t0_nanoseconds.setter
    # def du_t0_nanoseconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray) or isinstance(value, StdVectorList):
    #         # Clear the vector before setting
    #         self._du_t0_nanoseconds.clear()
    #         self._du_t0_nanoseconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_nanoseconds._vector = value
    #     else:
    #         raise ValueError(f"Incorrect type for du_t0_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    @property
    def trigger_position(self):
        """Trigger position in the trace (trigger start = nanoseconds - 2*sample number)"""
        return self._trigger_position

    @trigger_position.setter
    def trigger_position(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trigger_position.clear()
            self._trigger_position += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_position._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trigger_position {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def trigger_flag(self):
        """Same as event_type, but event_type could consist of different triggered DUs"""
        return self._trigger_flag

    @trigger_flag.setter
    def trigger_flag(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trigger_flag.clear()
            self._trigger_flag += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_flag._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trigger_flag {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def atm_temperature(self):
        """Atmospheric temperature (read via I2C)"""
        return self._atm_temperature

    @atm_temperature.setter
    def atm_temperature(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._atm_temperature.clear()
            self._atm_temperature += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._atm_temperature._vector = value
        else:
            raise ValueError(
                f"Incorrect type for atm_temperature {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    @property
    def atm_pressure(self):
        """Atmospheric pressure"""
        return self._atm_pressure

    @atm_pressure.setter
    def atm_pressure(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._atm_pressure.clear()
            self._atm_pressure += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._atm_pressure._vector = value
        else:
            raise ValueError(
                f"Incorrect type for atm_pressure {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    @property
    def atm_humidity(self):
        """Atmospheric humidity"""
        return self._atm_humidity

    @atm_humidity.setter
    def atm_humidity(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._atm_humidity.clear()
            self._atm_humidity += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._atm_humidity._vector = value
        else:
            raise ValueError(
                f"Incorrect type for atm_humidity {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    @property
    def du_acceleration(self):
        """Acceleration of the antenna in (x,y,z) in m/s2"""
        return self._du_acceleration

    @du_acceleration.setter
    def du_acceleration(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_acceleration.clear()
            self._du_acceleration += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._du_acceleration._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_acceleration {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )

    # @property
    # def acceleration_x(self):
    #     """Acceleration of the antenna in X"""
    #     return self._acceleration_x
    #
    # @acceleration_x.setter
    # def acceleration_x(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._acceleration_x.clear()
    #         self._acceleration_x += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("float")):
    #         self._acceleration_x._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for acceleration_x {type(value)}. Either a list, an array or a ROOT.vector of floats required."
    #         )
    #
    # @property
    # def acceleration_y(self):
    #     """Acceleration of the antenna in Y"""
    #     return self._acceleration_y
    #
    # @acceleration_y.setter
    # def acceleration_y(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._acceleration_y.clear()
    #         self._acceleration_y += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("float")):
    #         self._acceleration_y._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for acceleration_y {type(value)}. Either a list, an array or a ROOT.vector of floats required."
    #         )
    #
    # @property
    # def acceleration_z(self):
    #     """Acceleration of the antenna in Z"""
    #     return self._acceleration_z
    #
    # @acceleration_z.setter
    # def acceleration_z(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._acceleration_z.clear()
    #         self._acceleration_z += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("float")):
    #         self._acceleration_z._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for acceleration_z {type(value)}. Either a list, an array or a ROOT.vector of floats required."
    #         )

    @property
    def battery_level(self):
        """Battery voltage"""
        return self._battery_level

    @battery_level.setter
    def battery_level(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._battery_level.clear()
            self._battery_level += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._battery_level._vector = value
        else:
            raise ValueError(
                f"Incorrect type for battery_level {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    # @property
    # def firmware_version(self):
    #     """Firmware version"""
    #     return self._firmware_version
    #
    # @firmware_version.setter
    # def firmware_version(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._firmware_version.clear()
    #         self._firmware_version += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._firmware_version._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for firmware_version {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )
    #
    # @property
    # def adc_sampling_frequency(self):
    #     """ADC sampling frequency in MHz"""
    #     return self._adc_sampling_frequency
    #
    # @adc_sampling_frequency.setter
    # def adc_sampling_frequency(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_sampling_frequency.clear()
    #         self._adc_sampling_frequency += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_sampling_frequency._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_sampling_frequency {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )
    #
    # @property
    # def adc_sampling_resolution(self):
    #     """ADC sampling resolution in bits"""
    #     return self._adc_sampling_resolution
    #
    # @adc_sampling_resolution.setter
    # def adc_sampling_resolution(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_sampling_resolution.clear()
    #         self._adc_sampling_resolution += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_sampling_resolution._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_sampling_resolution {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )
    #
    # @property
    # def adc_input_channels(self):
    #     """ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?"""
    #     return self._adc_input_channels
    #
    # @adc_input_channels.setter
    # def adc_input_channels(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_input_channels.clear()
    #         self._adc_input_channels += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_input_channels._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_input_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )
    #
    # @property
    # def adc_enabled_channels(self):
    #     """ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?"""
    #     return self._adc_enabled_channels
    #
    # @adc_enabled_channels.setter
    # def adc_enabled_channels(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_enabled_channels.clear()
    #         self._adc_enabled_channels += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_enabled_channels._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_enabled_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )
    #
    # @property
    # def adc_samples_count_total(self):
    #     """ADC samples callected in all channels"""
    #     return self._adc_samples_count_total
    #
    # @adc_samples_count_total.setter
    # def adc_samples_count_total(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_samples_count_total.clear()
    #         self._adc_samples_count_total += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_samples_count_total._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_samples_count_total {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )
    #
    # @property
    # def adc_samples_count_channel_x(self):
    #     """ADC samples callected in channel x"""
    #     return self._adc_samples_count_channel_x
    #
    # @adc_samples_count_channel_x.setter
    # def adc_samples_count_channel_x(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_samples_count_channel_x.clear()
    #         self._adc_samples_count_channel_x += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_samples_count_channel_x._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_samples_count_channel_x {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )
    #
    # @property
    # def adc_samples_count_channel_y(self):
    #     """ADC samples callected in channel y"""
    #     return self._adc_samples_count_channel_y
    #
    # @adc_samples_count_channel_y.setter
    # def adc_samples_count_channel_y(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_samples_count_channel_y.clear()
    #         self._adc_samples_count_channel_y += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_samples_count_channel_y._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_samples_count_channel_y {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )
    #
    # @property
    # def adc_samples_count_channel_z(self):
    #     """ADC samples callected in channel z"""
    #     return self._adc_samples_count_channel_z
    #
    # @adc_samples_count_channel_z.setter
    # def adc_samples_count_channel_z(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._adc_samples_count_channel_z.clear()
    #         self._adc_samples_count_channel_z += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned short")):
    #         self._adc_samples_count_channel_z._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for adc_samples_count_channel_z {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
    #         )

    @property
    def adc_samples_count_channel(self):
        """ADC samples collected in channels (x,y,z)"""
        return self._adc_samples_count_channel

    @adc_samples_count_channel.setter
    def adc_samples_count_channel(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._adc_samples_count_channel.clear()
            self._adc_samples_count_channel += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._adc_samples_count_channel._vector = value
        else:
            raise ValueError(
                f"Incorrect type for adc_samples_count_channel {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short> required."
            )

    @property
    def trigger_pattern(self):
        """Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?"""
        return self._trigger_pattern

    @trigger_pattern.setter
    def trigger_pattern(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trigger_pattern.clear()
            self._trigger_pattern += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_pattern._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trigger_pattern {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def trigger_rate(self):
        """Trigger rate - the number of triggers recorded in the second preceding the event"""
        return self._trigger_rate

    @trigger_rate.setter
    def trigger_rate(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trigger_rate.clear()
            self._trigger_rate += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_rate._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trigger_rate {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def clock_tick(self):
        """Clock tick at which the event was triggered (used to calculate the trigger time)"""
        return self._clock_tick

    @clock_tick.setter
    def clock_tick(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._clock_tick.clear()
            self._clock_tick += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._clock_tick._vector = value
        else:
            raise ValueError(
                f"Incorrect type for clock_tick {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def clock_ticks_per_second(self):
        """Clock ticks per second"""
        return self._clock_ticks_per_second

    @clock_ticks_per_second.setter
    def clock_ticks_per_second(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._clock_ticks_per_second.clear()
            self._clock_ticks_per_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._clock_ticks_per_second._vector = value
        else:
            raise ValueError(
                f"Incorrect type for clock_ticks_per_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def gps_offset(self):
        """GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?"""
        return self._gps_offset

    @gps_offset.setter
    def gps_offset(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_offset.clear()
            self._gps_offset += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_offset._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_offset {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    @property
    def gps_leap_second(self):
        """GPS leap second"""
        return self._gps_leap_second

    @gps_leap_second.setter
    def gps_leap_second(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_leap_second.clear()
            self._gps_leap_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_leap_second._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_leap_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_status(self):
        """GPS status"""
        return self._gps_status

    @gps_status.setter
    def gps_status(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_status.clear()
            self._gps_status += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_status._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_status {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_alarms(self):
        """GPS alarms"""
        return self._gps_alarms

    @gps_alarms.setter
    def gps_alarms(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_alarms.clear()
            self._gps_alarms += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_alarms._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_alarms {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_warnings(self):
        """GPS warnings"""
        return self._gps_warnings

    @gps_warnings.setter
    def gps_warnings(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_warnings.clear()
            self._gps_warnings += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_warnings._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_warnings {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_time(self):
        """GPS time"""
        return self._gps_time

    @gps_time.setter
    def gps_time(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_time.clear()
            self._gps_time += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._gps_time._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_time {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def gps_long(self):
        """Longitude"""
        return self._gps_long

    @gps_long.setter
    def gps_long(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_long.clear()
            self._gps_long += value
        # A vector was given
        elif isinstance(value, ROOT.vector("double")):
            self._gps_long._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_long {type(value)}. Either a list, an array or a ROOT.vector of doubles required."
            )

    @property
    def gps_lat(self):
        """Latitude"""
        return self._gps_lat

    @gps_lat.setter
    def gps_lat(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_lat.clear()
            self._gps_lat += value
        # A vector was given
        elif isinstance(value, ROOT.vector("double")):
            self._gps_lat._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_lat {type(value)}. Either a list, an array or a ROOT.vector of doubles required."
            )

    @property
    def gps_alt(self):
        """Altitude"""
        return self._gps_alt

    @gps_alt.setter
    def gps_alt(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_alt.clear()
            self._gps_alt += value
        # A vector was given
        elif isinstance(value, ROOT.vector("double")):
            self._gps_alt._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_alt {type(value)}. Either a list, an array or a ROOT.vector of doubles required."
            )

    @property
    def gps_temp(self):
        """GPS temperature"""
        return self._gps_temp

    @gps_temp.setter
    def gps_temp(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._gps_temp.clear()
            self._gps_temp += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_temp._vector = value
        else:
            raise ValueError(
                f"Incorrect type for gps_temp {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    # @property
    # def pos_x(self):
    #     """X position in site's referential"""
    #     return self._pos_x
    #
    # @pos_x.setter
    # def pos_x(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._pos_x.clear()
    #         self._pos_x += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("float")):
    #         self._pos_x._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for pos_x {type(value)}. Either a list, an array or a ROOT.vector of floats required."
    #         )
    #
    # @property
    # def pos_y(self):
    #     """Y position in site's referential"""
    #     return self._pos_y
    #
    # @pos_y.setter
    # def pos_y(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._pos_y.clear()
    #         self._pos_y += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("float")):
    #         self._pos_y._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for pos_y {type(value)}. Either a list, an array or a ROOT.vector of floats required."
    #         )
    #
    # @property
    # def pos_z(self):
    #     """Z position in site's referential"""
    #     return self._pos_z
    #
    # @pos_z.setter
    # def pos_z(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._pos_z.clear()
    #         self._pos_z += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("float")):
    #         self._pos_z._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for pos_z {type(value)}. Either a list, an array or a ROOT.vector of floats required."
    #         )
    #
    # @property
    # def digi_ctrl(self):
    #     """Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual). ToDo: Decode?"""
    #     return self._digi_ctrl
    #
    # @digi_ctrl.setter
    # def digi_ctrl(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._digi_ctrl.clear()
    #         self._digi_ctrl += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._digi_ctrl._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for digi_ctrl {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def digi_prepost_trig_windows(self):
    #     """Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?"""
    #     return self._digi_prepost_trig_windows
    #
    # @digi_prepost_trig_windows.setter
    # def digi_prepost_trig_windows(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._digi_prepost_trig_windows.clear()
    #         self._digi_prepost_trig_windows += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._digi_prepost_trig_windows._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for digi_prepost_trig_windows {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_properties_x(self):
    #     """Channel x properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    #     return self._channel_properties_x
    #
    # @channel_properties_x.setter
    # def channel_properties_x(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_properties_x.clear()
    #         self._channel_properties_x += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_properties_x._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_properties_x {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_properties_y(self):
    #     """Channel y properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    #     return self._channel_properties_y
    #
    # @channel_properties_y.setter
    # def channel_properties_y(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_properties_y.clear()
    #         self._channel_properties_y += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_properties_y._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_properties_y {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_properties_z(self):
    #     """Channel z properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    #     return self._channel_properties_z
    #
    # @channel_properties_z.setter
    # def channel_properties_z(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_properties_z.clear()
    #         self._channel_properties_z += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_properties_z._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_properties_z {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_trig_settings_x(self):
    #     """Channel x trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    #     return self._channel_trig_settings_x
    #
    # @channel_trig_settings_x.setter
    # def channel_trig_settings_x(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_trig_settings_x.clear()
    #         self._channel_trig_settings_x += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_trig_settings_x._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_trig_settings_x {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_trig_settings_y(self):
    #     """Channel y trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    #     return self._channel_trig_settings_y
    #
    # @channel_trig_settings_y.setter
    # def channel_trig_settings_y(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_trig_settings_y.clear()
    #         self._channel_trig_settings_y += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_trig_settings_y._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_trig_settings_y {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )
    #
    # @property
    # def channel_trig_settings_z(self):
    #     """Channel z trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    #     return self._channel_trig_settings_z
    #
    # @channel_trig_settings_z.setter
    # def channel_trig_settings_z(self, value) -> None:
    #     # A list of strings was given
    #     if (
    #         isinstance(value, list)
    #         or isinstance(value, np.ndarray)
    #         or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._channel_trig_settings_z.clear()
    #         self._channel_trig_settings_z += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("vector<unsigned short>")):
    #         self._channel_trig_settings_z._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for channel_trig_settings_z {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
    #         )

    @property
    def ioff(self):
        """?? What is it? Some kind of the adc trace offset?"""
        return self._ioff

    @ioff.setter
    def ioff(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._ioff.clear()
            self._ioff += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._ioff._vector = value
        else:
            raise ValueError(
                f"Incorrect type for ioff {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    # @property
    # def trace_0(self):
    #     """Voltage trace channel 0"""
    #     return self._trace_0
    #
    # @trace_0.setter
    # def trace_0(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_0.clear()
    #         self._trace_0 += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_0._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_0 {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
    #         )
    #
    # @property
    # def trace_1(self):
    #     """Voltage trace in channel 1"""
    #     return self._trace_1
    #
    # @trace_1.setter
    # def trace_1(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_1.clear()
    #         self._trace_1 += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_1._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_1 {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def trace_2(self):
    #     """Voltage trace in channel 2"""
    #     return self._trace_2
    #
    # @trace_2.setter
    # def trace_2(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_2.clear()
    #         self._trace_2 += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_2._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_2 {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def trace_3(self):
    #     """Voltage trace in channel 3"""
    #     return self._trace_3
    #
    # @trace_3.setter
    # def trace_3(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_3.clear()
    #         self._trace_3 += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_3._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_3 {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )

    @property
    def trace_ch(self):
        """Voltage traces for channels (0,1,2,3) in muV"""
        return self._trace_ch

    @trace_ch.setter
    def trace_ch(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trace_ch._vector.clear()
            self._trace_ch += value

        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<vector<float>>")):
            self._trace_ch._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trace_ch {type(value)}. Either a list, an array or a ROOT.vector of vector<vector<float>> required."
            )


@dataclass
## The class for storing voltage traces and associated values for each event
class TVoltage(MotherEventTree):
    """The class for storing voltage traces and associated values at antenna feed point for each event"""

    _type: str = "voltage"

    _tree_name: str = "tvoltage"

    ## Common for the whole event
    ## First detector unit that triggered in the event
    _first_du: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Unix time corresponding to the GPS seconds of the trigger
    _time_seconds: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    _time_nanoseconds: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    _du_id: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Unix time of the trigger for this DU
    _du_seconds: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Nanoseconds of the trigger for this DU
    _du_nanoseconds: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Same as event_type, but event_type could consist of different triggered DUs
    _trigger_flag: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Acceleration of the antenna in (x,y,z) in m/s2
    _du_acceleration: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    _trigger_rate: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))

    # ## Voltage trace in X direction
    # _trace_x: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Voltage trace in Y direction
    # _trace_y: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Voltage trace in Z direction
    # _trace_z: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## Voltage traces for antenna arms (x,y,z)
    _trace: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<float>>"))
    # _trace: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<Float32_t>>"))

    ## Peak2peak amplitude (muV)
    _p2p: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## (Computed) peak time
    _time_max: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))

    @property
    def first_du(self):
        """First detector unit that triggered in the event"""
        return self._first_du[0]

    @first_du.setter
    def first_du(self, value: np.uint32) -> None:
        self._first_du[0] = value

    @property
    def time_seconds(self):
        """Unix time corresponding to the GPS seconds of the trigger"""
        return self._time_seconds[0]

    @time_seconds.setter
    def time_seconds(self, value: np.uint32) -> None:
        self._time_seconds[0] = value

    @property
    def time_nanoseconds(self):
        """GPS nanoseconds corresponding to the trigger of the first triggered station"""
        return self._time_nanoseconds[0]

    @time_nanoseconds.setter
    def time_nanoseconds(self, value: np.uint32) -> None:
        self._time_nanoseconds[0] = value

    @property
    def du_count(self):
        """Number of detector units in the event - basically the antennas count"""
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

    @property
    def du_id(self):
        """Detector unit (antenna) ID"""
        return self._du_id

    @du_id.setter
    def du_id(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._du_id._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def du_seconds(self):
        """Unix time of the trigger for this DU"""
        return self._du_seconds

    @du_seconds.setter
    def du_seconds(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_seconds.clear()
            self._du_seconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_seconds._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def du_nanoseconds(self):
        """Nanoseconds of the trigger for this DU"""
        return self._du_nanoseconds

    @du_nanoseconds.setter
    def du_nanoseconds(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_nanoseconds.clear()
            self._du_nanoseconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_nanoseconds._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def trigger_flag(self):
        """Same as event_type, but event_type could consist of different triggered DUs"""
        return self._trigger_flag

    @trigger_flag.setter
    def trigger_flag(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trigger_flag.clear()
            self._trigger_flag += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_flag._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trigger_flag {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def du_acceleration(self):
        """Acceleration of the antenna in (x,y,z) in m/s2"""
        return self._du_acceleration

    @du_acceleration.setter
    def du_acceleration(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_acceleration.clear()
            self._du_acceleration += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._du_acceleration._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_acceleration {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )

    @property
    def trigger_rate(self):
        """Trigger rate - the number of triggers recorded in the second preceding the event"""
        return self._trigger_rate

    @trigger_rate.setter
    def trigger_rate(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trigger_rate.clear()
            self._trigger_rate += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_rate._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trigger_rate {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    # @property
    # def trace_x(self):
    #     """Efield trace in X direction"""
    #     return self._trace_x
    #
    # @trace_x.setter
    # def trace_x(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_x.clear()
    #         self._trace_x += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_x._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_x {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
    #         )
    #
    # @property
    # def trace_y(self):
    #     """Efield trace in Y direction"""
    #     return self._trace_y
    #
    # @trace_y.setter
    # def trace_y(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_y.clear()
    #         self._trace_y += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_y._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_y {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def trace_z(self):
    #     """Efield trace in Z direction"""
    #     return self._trace_z
    #
    # @trace_z.setter
    # def trace_z(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_z.clear()
    #         self._trace_z += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_z._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_z {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )

    @property
    def trace(self):
        """Voltage traces in (x,y,z) directions in muV"""
        return self._trace

    @trace.setter
    def trace(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trace._vector.clear()
            self._trace += value

        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<vector<float>>")):
            self._trace._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trace {type(value)}. Either a list, an array or a ROOT.vector of vector<vector<float>> required."
            )

    @property
    def p2p(self):
        """Peak2peak amplitude (muV)"""
        return self._p2p

    @p2p.setter
    def p2p(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._p2p.clear()
            self._p2p += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._p2p._vector = value
        else:
            raise ValueError(
                f"Incorrect type for p2p {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )

    @property
    def time_max(self):
        """(Computed) peak time"""
        return self._time_max

    @time_max.setter
    def time_max(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._time_max.clear()
            self._time_max += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._time_max._vector = value
        else:
            raise ValueError(
                f"Incorrect type for time_max {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )


@dataclass
## The class for storing Efield traces and associated values for each event
class TEfield(MotherEventTree):
    """The class for storing Efield traces and associated values for each event"""

    _type: str = "efield"

    _tree_name: str = "tefield"

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
    _time_seconds: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    _time_nanoseconds: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    _event_type: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    _du_id: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned short"))
    ## Unix time of the trigger for this DU
    _du_seconds: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))
    ## Nanoseconds of the trigger for this DU
    _du_nanoseconds: StdVectorList = field(default_factory=lambda: StdVectorList("unsigned int"))

    # ## Efield trace in X direction
    # _trace_x: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Efield trace in Y direction
    # _trace_y: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## Efield trace in Z direction
    # _trace_z: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## FFT magnitude in X direction
    # _fft_mag_x: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## FFT magnitude in Y direction
    # _fft_mag_y: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## FFT magnitude in Z direction
    # _fft_mag_z: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## FFT phase in X direction
    # _fft_phase_x: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## FFT phase in Y direction
    # _fft_phase_y: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    # ## FFT phase in Z direction
    # _fft_phase_z: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))

    ## Efield traces for antenna arms (x,y,z)
    _trace: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<float>>"))
    ## FFT magnitude for antenna arms (x,y,z)
    _fft_mag: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<float>>"))
    ## FFT phase for antenna arms (x,y,z)
    _fft_phase: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<float>>"))

    ## Peak-to-peak amplitudes for X, Y, Z (muV/m)
    _p2p: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## Efield polarisation info
    _pol: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## (Computed) peak time
    _time_max: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))

    @property
    def time_seconds(self):
        """Unix time corresponding to the GPS seconds of the trigger"""
        return self._time_seconds[0]

    @time_seconds.setter
    def time_seconds(self, value: np.uint32) -> None:
        self._time_seconds[0] = value

    @property
    def time_nanoseconds(self):
        """GPS nanoseconds corresponding to the trigger of the first triggered station"""
        return self._time_nanoseconds[0]

    @time_nanoseconds.setter
    def time_nanoseconds(self, value: np.uint32) -> None:
        self._time_nanoseconds[0] = value

    @property
    def event_type(self):
        """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
        return self._event_type[0]

    @event_type.setter
    def event_type(self, value: np.uint32) -> None:
        self._event_type[0] = value

    @property
    def du_count(self):
        """Number of detector units in the event - basically the antennas count"""
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

    @property
    def du_id(self):
        """Detector unit (antenna) ID"""
        return self._du_id

    @du_id.setter
    def du_id(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._du_id._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def du_seconds(self):
        """Unix time of the trigger for this DU"""
        return self._du_seconds

    @du_seconds.setter
    def du_seconds(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_seconds.clear()
            self._du_seconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_seconds._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def du_nanoseconds(self):
        """Nanoseconds of the trigger for this DU"""
        return self._du_nanoseconds

    @du_nanoseconds.setter
    def du_nanoseconds(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_nanoseconds.clear()
            self._du_nanoseconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_nanoseconds._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    # @property
    # def du_t0_seconds(self):
    #     return self._du_t0_seconds
    #
    # @du_t0_seconds.setter
    # def du_t0_seconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray) or isinstance(value, StdVectorList):
    #         # Clear the vector before setting
    #         self._du_t0_seconds.clear()
    #         self._du_t0_seconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_seconds._vector = value
    #     else:
    #         raise ValueError(f"Incorrect type for du_t0_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")
    #
    # @property
    # def du_t0_nanoseconds(self):
    #     return self._du_t0_nanoseconds
    #
    # @du_t0_nanoseconds.setter
    # def du_t0_nanoseconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray) or isinstance(value, StdVectorList):
    #         # Clear the vector before setting
    #         self._du_t0_nanoseconds.clear()
    #         self._du_t0_nanoseconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_nanoseconds._vector = value
    #     else:
    #         raise ValueError(f"Incorrect type for du_t0_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")
    # @property
    # def trace_x(self):
    #     """Efield trace in X direction"""
    #     return self._trace_x
    #
    # @trace_x.setter
    # def trace_x(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_x.clear()
    #         self._trace_x += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_x._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_x {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
    #         )
    #
    # @property
    # def trace_y(self):
    #     """Efield trace in Y direction"""
    #     return self._trace_y
    #
    # @trace_y.setter
    # def trace_y(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_y.clear()
    #         self._trace_y += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_y._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_y {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def trace_z(self):
    #     """Efield trace in Z direction"""
    #     return self._trace_z
    #
    # @trace_z.setter
    # def trace_z(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._trace_z.clear()
    #         self._trace_z += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._trace_z._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for trace_z {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def fft_mag_x(self):
    #     """FFT magnitude in X direction"""
    #     return self._fft_mag_x
    #
    # @fft_mag_x.setter
    # def fft_mag_x(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._fft_mag_x.clear()
    #         self._fft_mag_x += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._fft_mag_x._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for fft_mag_x {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def fft_mag_y(self):
    #     """FFT magnitude in Y direction"""
    #     return self._fft_mag_y
    #
    # @fft_mag_y.setter
    # def fft_mag_y(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._fft_mag_y.clear()
    #         self._fft_mag_y += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._fft_mag_y._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for fft_mag_y {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def fft_mag_z(self):
    #     """FFT magnitude in Z direction"""
    #     return self._fft_mag_z
    #
    # @fft_mag_z.setter
    # def fft_mag_z(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._fft_mag_z.clear()
    #         self._fft_mag_z += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._fft_mag_z._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for fft_mag_z {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def fft_phase_x(self):
    #     """FFT phase in X direction"""
    #     return self._fft_phase_x
    #
    # @fft_phase_x.setter
    # def fft_phase_x(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._fft_phase_x.clear()
    #         self._fft_phase_x += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._fft_phase_x._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for fft_phase_x {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def fft_phase_y(self):
    #     """FFT phase in Y direction"""
    #     return self._fft_phase_y
    #
    # @fft_phase_y.setter
    # def fft_phase_y(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._fft_phase_y.clear()
    #         self._fft_phase_y += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._fft_phase_y._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for fft_phase_y {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )
    #
    # @property
    # def fft_phase_z(self):
    #     """FFT phase in Z direction"""
    #     return self._fft_phase_z
    #
    # @fft_phase_z.setter
    # def fft_phase_z(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._fft_phase_z.clear()
    #         self._fft_phase_z += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("vector<float>")):
    #         self._fft_phase_z._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for fft_phase_z {type(value)}. Either a list, an array or a ROOT.vector of float required."
    #         )

    @property
    def trace(self):
        """Efield traces in (x,y,z) directions in muV"""
        return self._trace

    @trace.setter
    def trace(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trace._vector.clear()
            self._trace += value

        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<vector<float>>")):
            self._trace._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trace {type(value)}. Either a list, an array or a ROOT.vector of vector<vector<float>> required."
            )

    @property
    def fft_mag(self):
        """Efield fft_mags in (x,y,z) directions in muV"""
        return self._fft_mag

    @fft_mag.setter
    def fft_mag(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._fft_mag._vector.clear()
            self._fft_mag += value

        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<vector<float>>")):
            self._fft_mag._vector = value
        else:
            raise ValueError(
                f"Incorrect type for fft_mag {type(value)}. Either a list, an array or a ROOT.vector of vector<vector<float>> required."
            )

    @property
    def fft_phase(self):
        """Efield fft_phases in (x,y,z) directions in muV"""
        return self._fft_phase

    @fft_phase.setter
    def fft_phase(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._fft_phase._vector.clear()
            self._fft_phase += value

        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<vector<float>>")):
            self._fft_phase._vector = value
        else:
            raise ValueError(
                f"Incorrect type for fft_phase {type(value)}. Either a list, an array or a ROOT.vector of vector<vector<float>> required."
            )

    @property
    def p2p(self):
        """Peak2peak amplitude (muV)"""
        return self._p2p

    @p2p.setter
    def p2p(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._p2p.clear()
            self._p2p += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._p2p._vector = value
        else:
            raise ValueError(
                f"Incorrect type for p2p {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )

    @property
    def pol(self):
        """Efield polarisation info"""
        return self._pol

    @pol.setter
    def pol(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._pol.clear()
            self._pol += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._pol._vector = value
        else:
            raise ValueError(
                f"Incorrect type for pol {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )

    @property
    def time_max(self):
        """(Computed) peak time"""
        return self._time_max

    @time_max.setter
    def time_max(self, value):
        # A list was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._time_max.clear()
            self._time_max += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._time_max._vector = value
        else:
            raise ValueError(
                f"Incorrect type for time_max {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )


@dataclass
## The class for storing reconstructed shower data common for each event
class TShower(MotherEventTree):
    """The class for storing shower data common for each event"""

    _type: str = "shower"

    _tree_name: str = "tshower"

    # Shower primary type
    _primary_type: StdString = StdString("")
    # Energy from e+- (ie related to radio emission) (GeV)
    _energy_em: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # Total energy of the primary (including muons, neutrinos, ...) (GeV)
    _energy_primary: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # Shower azimuth  (coordinates system = NWU + origin = core, "pointing to")
    _azimuth: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # Shower zenith  (coordinates system = NWU + origin = core, , "pointing to")
    _zenith: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # Direction vector (u_x, u_y, u_z)  of shower in GRAND detector ref
    _direction: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    # Shower core position in GRAND detector ref (if it is an upgoing shower, there is no core position)
    _shower_core_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    # Atmospheric model name
    _atmos_model: StdString = StdString("")
    # Atmospheric model parameters
    _atmos_model_param: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    # Magnetic field parameters: Inclination, Declination, modulus
    _magnetic_field: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    # Ground Altitude at core position (m asl)
    _core_alt: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # Shower Xmax depth  (g/cm2 along the shower axis)
    _xmax_grams: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # Shower Xmax position in GRAND detector ref
    _xmax_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float64))
    # Shower Xmax position in shower coordinates
    _xmax_pos_shc: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float64))
    # Unix time when the shower was at the core position (seconds after epoch)
    _core_time_s: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))
    # Unix time when the shower was at the core position (seconds after epoch)
    _core_time_ns: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    @property
    def primary_type(self):
        """Shower primary type"""
        return str(self._primary_type)

    @primary_type.setter
    def primary_type(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for primary_type {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._primary_type.string.assign(value)

    @property
    def energy_em(self):
        """Energy from e+- (ie related to radio emission) (GeV)"""
        return self._energy_em[0]

    @energy_em.setter
    def energy_em(self, value):
        self._energy_em[0] = value

    @property
    def energy_primary(self):
        """Total energy of the primary (including muons, neutrinos, ...) (GeV)"""
        return self._energy_primary[0]

    @energy_primary.setter
    def energy_primary(self, value):
        self._energy_primary[0] = value

    @property
    def azimuth(self):
        """Shower azimuth  (coordinates system = NWU + origin = core, "pointing to")"""
        return self._azimuth[0]

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth[0] = value

    @property
    def zenith(self):
        """Shower zenith  (coordinates system = NWU + origin = core, , "pointing to")"""
        return self._zenith[0]

    @zenith.setter
    def zenith(self, value):
        self._zenith[0] = value

    @property
    def direction(self):
        """Direction vector (u_x, u_y, u_z)  of shower in GRAND detector ref"""
        return np.array(self._direction)

    @direction.setter
    def direction(self, value):
        self._direction = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("direction", self._direction)

    @property
    def shower_core_pos(self):
        """Shower core position in GRAND detector ref (if it is an upgoing shower, there is no core position)"""
        return np.array(self._shower_core_pos)

    @shower_core_pos.setter
    def shower_core_pos(self, value):
        self._shower_core_pos = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("shower_core_pos", self._shower_core_pos)

    @property
    def atmos_model(self):
        """Atmospheric model name"""
        return str(self._atmos_model)

    @atmos_model.setter
    def atmos_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for atmos_model {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._atmos_model.string.assign(value)

    @property
    def atmos_model_param(self):
        """Atmospheric model parameters"""
        return np.array(self._atmos_model_param)

    @atmos_model_param.setter
    def atmos_model_param(self, value):
        self._atmos_model_param = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("atmos_model_param", self._atmos_model_param)

    @property
    def magnetic_field(self):
        """Magnetic field parameters: Inclination, Declination, modulus"""
        return np.array(self._magnetic_field)

    @magnetic_field.setter
    def magnetic_field(self, value):
        self._magnetic_field = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("magnetic_field", self._magnetic_field)

    @property
    def core_alt(self):
        """Ground Altitude at core position (m asl)"""
        return self._core_alt[0]

    @core_alt.setter
    def core_alt(self, value):
        self._core_alt[0] = value

    @property
    def xmax_grams(self):
        """Shower Xmax depth (g/cm2 along the shower axis)"""
        return self._xmax_grams[0]

    @xmax_grams.setter
    def xmax_grams(self, value):
        self._xmax_grams[0] = value

    @property
    def xmax_pos(self):
        """Shower Xmax position in GRAND detector ref"""
        return np.array(self._xmax_pos)

    @xmax_pos.setter
    def xmax_pos(self, value):
        self._xmax_pos = np.array(value).astype(np.float64)
        self._tree.SetBranchAddress("xmax_pos", self._xmax_pos)

    @property
    def xmax_pos_shc(self):
        """Shower Xmax position in shower coordinates."""
        return np.array(self._xmax_pos_shc)

    @xmax_pos_shc.setter
    def xmax_pos_shc(self, value):
        self._xmax_pos_shc = np.array(value).astype(np.float64)
        self._tree.SetBranchAddress("xmax_pos_shc", self._xmax_pos_shc)

    @property
    def core_time_s(self):
        """Unix time when the shower was at the core position (seconds after epoch)"""
        return self._core_time_s[0]

    @core_time_s.setter
    def core_time_s(self, value):
        self._core_time_s[0] = value

    @property
    def core_time_ns(self):
        """Unix time when the shower was at the core position (ns part)"""
        return self._core_time_ns[0]

    @core_time_ns.setter
    def core_time_ns(self, value):
        self._core_time_ns[0] = value


# @dataclass
# ## The class for storing voltage sim-only data common for a whole run
# class VoltageRunSimdataTree(MotherRunTree):
#     """The class for storing voltage sim-only data common for a whole run"""
#
#     _type: str = "runvoltagesimdata"
#
#     _tree_name: str = "trunvoltagesimdata"
#
#     _signal_sim: StdString = StdString("")  # name and model of the signal simulator
#
#     def __post_init__(self):
#         super().__post_init__()
#
#         if self._tree.GetName() == "":
#             self._tree.SetName(self._tree_name)
#         if self._tree.GetTitle() == "":
#             self._tree.SetTitle(self._tree_name)
#
#         self.create_branches()
#
#     @property
#     def signal_sim(self):
#         """Name and model of the signal simulator"""
#         return str(self._signal_sim)
#
#     @signal_sim.setter
#     def signal_sim(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._signal_sim.string.assign(value)


# @dataclass
# ## The class for storing voltage sim-only data common for each event
# class VoltageEventSimdataTree(MotherEventTree):
#     """The class for storing voltage sim-only data common for each event"""
#
#     _type: str = "eventvoltagesimdata"
#
#     _tree_name: str = "teventvoltagesimdata"
#
#     _du_id: StdVectorList = field(default_factory=lambda: StdVectorList("int"))  # Detector ID
#     _t_0: StdVectorList = field(default_factory=lambda: StdVectorList("float"))  # Time window t0
#     _p2p: StdVectorList = field(
#         default_factory=lambda: StdVectorList("float")
#     )  # peak 2 peak amplitudes (x,y,z,modulus)
#
#     # def __post_init__(self):
#     #     super().__post_init__()
#     #
#     #     if self._tree.GetName() == "":
#     #         self._tree.SetName(self._tree_name)
#     #     if self._tree.GetTitle() == "":
#     #         self._tree.SetTitle(self._tree_name)
#     #
#     #     self.create_branches()
#
#     @property
#     def du_id(self):
#         """Detector ID"""
#         return self._du_id
#
#     @du_id.setter
#     def du_id(self, value):
#
#         # A list was given
#         if (
#             isinstance(value, list)
#             or isinstance(value, np.ndarray)
#             or isinstance(value, StdVectorList)
#         ):
#             # Clear the vector before setting
#             self._du_id.clear()
#             self._du_id += value
#         # A vector of strings was given
#         elif isinstance(value, ROOT.vector("int")):
#             self._du_id._vector = value
#         else:
#             raise ValueError(
#                 f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of float required."
#             )
#
#     @property
#     def t_0(self):
#         """Time window t0"""
#         return self._t_0
#
#     @t_0.setter
#     def t_0(self, value):
#
#         # A list was given
#         if (
#             isinstance(value, list)
#             or isinstance(value, np.ndarray)
#             or isinstance(value, StdVectorList)
#         ):
#             # Clear the vector before setting
#             self._t_0.clear()
#             self._t_0 += value
#         # A vector of strings was given
#         elif isinstance(value, ROOT.vector("float")):
#             self._t_0._vector = value
#         else:
#             raise ValueError(
#                 f"Incorrect type for t_0 {type(value)}. Either a list, an array or a ROOT.vector of float required."
#             )
#
#     @property
#     def p2p(self):
#         """Peak 2 peak amplitudes (x,y,z,modulus)"""
#         return self._p2p
#
#     @p2p.setter
#     def p2p(self, value):
#
#         # A list was given
#         if (
#             isinstance(value, list)
#             or isinstance(value, np.ndarray)
#             or isinstance(value, StdVectorList)
#         ):
#             # Clear the vector before setting
#             self._p2p.clear()
#             self._p2p += value
#         # A vector of strings was given
#         elif isinstance(value, ROOT.vector("float")):
#             self._p2p._vector = value
#         else:
#             raise ValueError(
#                 f"Incorrect type for p2p {type(value)}. Either a list, an array or a ROOT.vector of float required."
#             )


@dataclass
## The class for storing Efield sim-only data common for a whole run
class TRunEfieldSim(MotherRunTree):
    """The class for storing Efield sim-only data common for a whole run"""

    _type: str = "runefieldsim"

    _tree_name: str = "trunefieldsim"

    ## Name and model of the electric field simulator
    # _field_sim: StdString = StdString("")
    ## Name of the atmospheric index of refraction model
    _refractivity_model: StdString = StdString("")
    _refractivity_model_parameters: StdVectorList = field(
        default_factory=lambda: StdVectorList("double")
    )
    ## Starting time of antenna data collection time window (because it can be a shorter trace then voltage trace, and thus these parameters can be different)
    _t_pre: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    ## Finishing time of antenna data collection time window
    _t_post: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))

    ## Simulator name (aires/corsika, etc.)
    _sim_name: StdString = StdString("")
    ## Simulator version string
    _sim_version: StdString = StdString("")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def refractivity_model(self):
        """Name of the atmospheric index of refraction model"""
        return str(self._refractivity_model)

    @refractivity_model.setter
    def refractivity_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._refractivity_model.string.assign(value)

    @property
    def refractivity_model_parameters(self):
        """Refractivity model parameters"""
        return self._refractivity_model_parameters

    @refractivity_model_parameters.setter
    def refractivity_model_parameters(self, value) -> None:
        # A list of strings was given
        if (
                isinstance(value, list)
                or isinstance(value, np.ndarray)
                or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._refractivity_model_parameters.clear()
            self._refractivity_model_parameters += value
        # A vector was given
        elif isinstance(value, ROOT.vector("double")):
            self._refractivity_model_parameters._vector = value
        else:
            raise ValueError(
                f"Incorrect type for refractivity_model_parameters {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def t_pre(self):
        """Starting time of antenna data collection time window (because it can be a shorter trace then voltage trace, and thus these parameters can be different)"""
        return self._t_pre[0]

    @t_pre.setter
    def t_pre(self, value):
        self._t_pre[0] = value

    @property
    def t_post(self):
        """Finishing time of antenna data collection time window"""
        return self._t_post[0]

    @t_post.setter
    def t_post(self, value):
        self._t_post[0] = value

    @property
    def sim_name(self):
        """Simulator name (aires/corsika, etc.)"""
        return str(self._sim_name)

    @sim_name.setter
    def sim_name(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._sim_name.string.assign(value)

    @property
    def sim_version(self):
        """Simulator version string"""
        return str(self._sim_version)

    @sim_version.setter
    def sim_version(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._sim_version.string.assign(value)


# @dataclass
# ## The class for storing Efield sim-only data common for each event
# class EfieldEventSimdataTree(MotherEventTree):
#     """The class for storing Efield sim-only data common for each event"""
#
#     _type: str = "eventefieldsimdata"
#
#     _tree_name: str = "teventefieldsimdata"
#
#     _du_id: StdVectorList = field(default_factory=lambda: StdVectorList("int"))  # Detector ID
#     _t_0: StdVectorList = field(default_factory=lambda: StdVectorList("float"))  # Time window t0
#     _p2p: StdVectorList = field(
#         default_factory=lambda: StdVectorList("float")
#     )  # peak 2 peak amplitudes (x,y,z,modulus)
#
#     # _event_size: np.ndarray = np.zeros(1, np.uint32)
#     # _start_time: StdVectorList("double") = StdVectorList("double")
#     # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
#     # _det_time: StdVectorList("double") = StdVectorList("double")
#     # _e_det_time: StdVectorList("double") = StdVectorList("double")
#     # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
#     # _sampling_speed: StdVectorList("float") = StdVectorList("float")
#
#     # def __post_init__(self):
#     #     super().__post_init__()
#     #
#     #     if self._tree.GetName() == "":
#     #         self._tree.SetName(self._tree_name)
#     #     if self._tree.GetTitle() == "":
#     #         self._tree.SetTitle(self._tree_name)
#     #
#     #     self.create_branches()
#
#     @property
#     def du_id(self):
#         """Detector ID"""
#         return self._du_id
#
#     @du_id.setter
#     def du_id(self, value):
#         # A list was given
#         if (
#             isinstance(value, list)
#             or isinstance(value, np.ndarray)
#             or isinstance(value, StdVectorList)
#         ):
#             # Clear the vector before setting
#             self._du_id.clear()
#             self._du_id += value
#         # A vector of strings was given
#         elif isinstance(value, ROOT.vector("int")):
#             self._du_id._vector = value
#         else:
#             raise ValueError(
#                 f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of float required."
#             )
#
#     @property
#     def t_0(self):
#         """Time window t0"""
#         return self._t_0
#
#     @t_0.setter
#     def t_0(self, value):
#         # A list was given
#         if (
#             isinstance(value, list)
#             or isinstance(value, np.ndarray)
#             or isinstance(value, StdVectorList)
#         ):
#             # Clear the vector before setting
#             self._t_0.clear()
#             self._t_0 += value
#         # A vector of strings was given
#         elif isinstance(value, ROOT.vector("float")):
#             self._t_0._vector = value
#         else:
#             raise ValueError(
#                 f"Incorrect type for t_0 {type(value)}. Either a list, an array or a ROOT.vector of float required."
#             )
#
#     @property
#     def p2p(self):
#         """Peak 2 peak amplitudes (x,y,z,modulus)"""
#         return self._p2p
#
#     @p2p.setter
#     def p2p(self, value):
#         # A list was given
#         if (
#             isinstance(value, list)
#             or isinstance(value, np.ndarray)
#             or isinstance(value, StdVectorList)
#         ):
#             # Clear the vector before setting
#             self._p2p.clear()
#             self._p2p += value
#         # A vector of strings was given
#         elif isinstance(value, ROOT.vector("float")):
#             self._p2p._vector = value
#         else:
#             raise ValueError(
#                 f"Incorrect type for p2p {type(value)}. Either a list, an array or a ROOT.vector of float required."
#             )


@dataclass
## The class for storing shower sim-only data common for a whole run
class TRunShowerSim(MotherRunTree):
    """Run-level info associated with simulated showers"""

    _type: str = "runshowersim"

    _tree_name: str = "trunshowersim"

    # relative thinning energy
    _rel_thin: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # maximum_weight (weight factor)
    _maximum_weight: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    """the maximum weight, computed in zhaires as PrimaryEnergy*RelativeThinning*WeightFactor/14.0 (see aires manual section 3.3.6 and 2.3.2) to make it mean the same as Corsika Wmax"""

    hadronic_thinning: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """the ratio of energy at wich thining starts in hadrons and electromagnetic particles"""
    hadronic_thinning_weight: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """the ratio of electromagnetic to hadronic maximum weights"""

    # low energy cut for electrons (GeV)
    _lowe_cut_e: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # low energy cut for gammas (GeV)
    _lowe_cut_gamma: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # low energy cut for muons (GeV)
    _lowe_cut_mu: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # low energy cut for mesons (GeV)
    _lowe_cut_meson: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # low energy cut for nuceleons (GeV)
    _lowe_cut_nucleon: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    # Site for wich the smulation was done
    _site: StdString = StdString("")
    # Simulator name (aires/corsika, etc.)
    _sim_name: StdString = StdString("")
    # Simulator version string
    _sim_version: StdString = StdString("")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def rel_thin(self):
        """relative thinning energy"""
        return self._rel_thin[0]

    @rel_thin.setter
    def rel_thin(self, value):
        self._rel_thin[0] = value

    @property
    def maximum_weight(self):
        """maximum_weight"""
        return self._maximum_weight[0]

    @maximum_weight.setter
    def maximum_weight(self, value):
        self._maximum_weight[0] = value

    @property
    def lowe_cut_e(self):
        """low energy cut for electrons(GeV)"""
        return self._lowe_cut_e[0]

    @lowe_cut_e.setter
    def lowe_cut_e(self, value):
        self._lowe_cut_e[0] = value

    @property
    def lowe_cut_gamma(self):
        """low energy cut for gammas(GeV)"""
        return self._lowe_cut_gamma[0]

    @lowe_cut_gamma.setter
    def lowe_cut_gamma(self, value):
        self._lowe_cut_gamma[0] = value

    @property
    def lowe_cut_mu(self):
        """low energy cut for muons(GeV)"""
        return self._lowe_cut_mu[0]

    @lowe_cut_mu.setter
    def lowe_cut_mu(self, value):
        self._lowe_cut_mu[0] = value

    @property
    def lowe_cut_meson(self):
        """low energy cut for mesons(GeV)"""
        return self._lowe_cut_meson[0]

    @lowe_cut_meson.setter
    def lowe_cut_meson(self, value):
        self._lowe_cut_meson[0] = value

    @property
    def lowe_cut_nucleon(self):
        """low energy cut for nucleons(GeV)"""
        return self._lowe_cut_nucleon[0]

    @lowe_cut_nucleon.setter
    def lowe_cut_nucleon(self, value):
        self._lowe_cut_nucleon[0] = value

    @property
    def site(self):
        """Name of the atmospheric index of refraction model"""
        return str(self._site)

    @site.setter
    def site(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._site.string.assign(value)

    @property
    def sim_name(self):
        """Simulator name (aires/corsika, etc.)"""
        return str(self._sim_name)

    @sim_name.setter
    def sim_name(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for sim_name {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._sim_name.string.assign(value)

    @property
    def sim_version(self):
        """Simulator version string"""
        return str(self._sim_version)

    @sim_version.setter
    def sim_version(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for sim_version {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._sim_version.string.assign(value)


@dataclass
## The class for storing a shower sim-only data for each event
class TShowerSim(MotherEventTree):
    """Event-level info associated with simulated showers"""

    _type: str = "showersim"

    _tree_name: str = "tshowersim"

    ## File name in the simulator
    _input_name: StdString = StdString("")
    ## The date for which we simulate the event (epoch)
    _event_date: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
    ## Random seed
    _rnd_seed: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))
    ## Primary energy (GeV)
    # primary_energy: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Primary particle type
    # primary_type: StdVectorListDesc = field(default=StdVectorListDesc("string"))
    ## Primary injection point in Shower Coordinates
    primary_inj_point_shc: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    ## Primary injection altitude in Shower Coordinates
    primary_inj_alt_shc: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Primary injection direction in Shower Coordinates
    primary_inj_dir_shc: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))

    ## Table of air density [g/cm3] and vertical depth [g/cm2] versus altitude [m]
    atmos_altitude: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    atmos_density: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    atmos_depth: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))

    ## High energy hadronic model (and version) used
    _hadronic_model: StdString = StdString("")
    ## Energy model (and version) used
    _low_energy_model: StdString = StdString("")
    ## Time it took for the sim
    _cpu_time: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))

    ## Slant depth of the observing levels for longitudinal development tables
    _long_depth: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    long_pd_depth: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Number of electrons
    long_pd_eminus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Number of positrons
    long_pd_eplus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Number of muons-
    long_pd_muminus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Number of muons+
    long_pd_muplus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Number of gammas
    long_pd_gamma: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Number of pions, kaons, etc.
    long_pd_hadron: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Energy in low energy gammas
    long_gamma_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Energy in low energy e+/e-
    long_e_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Energy deposited by e+/e-
    long_e_edep: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Energy in low energy mu+/mu-
    long_mu_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Energy deposited by mu+/mu-
    long_mu_edep: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Energy in low energy hadrons
    long_hadron_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Energy deposited by hadrons
    long_hadron_edep: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Energy in created neutrinos
    long_neutrino: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Core positions tested for that shower to generate the event (effective area study)
    _tested_core_positions: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))

    event_weight: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """statistical weight given to the event"""
    tested_cores: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """tested core positions"""

    @property
    def input_name(self):
        """File name in the simulator"""
        return str(self._input_name)

    @input_name.setter
    def input_name(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for input_name {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._input_name.string.assign(value)

    @property
    def event_date(self):
        """The date for which we simulate the event (epoch)"""
        return self._event_date[0]

    @event_date.setter
    def event_date(self, value):
        self._event_date[0] = value

    @property
    def rnd_seed(self):
        """Random seed"""
        return self._rnd_seed[0]

    @rnd_seed.setter
    def rnd_seed(self, value):
        self._rnd_seed[0] = value

    # @property
    # def sim_primary_energy(self):
    #     """Primary energy (GeV)"""
    #     return self._sim_primary_energy
    #
    # @sim_primary_energy.setter
    # def sim_primary_energy(self, value):
    #     # A list of strings was given
    #     if isinstance(value, list):
    #         # Clear the vector before setting
    #         self._sim_primary_energy.clear()
    #         self._sim_primary_energy += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("float")):
    #         self._sim_primary_energy._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for sim_primary_energy {type(value)}. Either a list or a ROOT.vector of floats required."
    #         )
    #
    # @property
    # def sim_primary_type(self):
    #     """Primary particle type"""
    #     return self._sim_primary_type
    #
    # @sim_primary_type.setter
    # def sim_primary_type(self, value):
    #     # A list of strings was given
    #     if isinstance(value, list):
    #         # Clear the vector before setting
    #         self._sim_primary_type.clear()
    #         self._sim_primary_type += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("string")):
    #         self._sim_primary_type._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for sim_primary_type {type(value)}. Either a list or a ROOT.vector of strings required."
    #         )
    #
    # @property
    # def sim_primary_inj_point_shc(self):
    #     """Primary injection point in Shower coordinates"""
    #     return np.array(self._sim_primary_inj_point_shc)
    #
    # @sim_primary_inj_point_shc.setter
    # def sim_primary_inj_point_shc(self, value):
    #     set_vector_of_vectors(value, "vector<float>", self._sim_primary_inj_point_shc, "sim_primary_inj_point_shc")
    #
    # @property
    # def sim_primary_inj_alt_shc(self):
    #     """Primary injection altitude in Shower Coordinates"""
    #     return self._sim_primary_inj_alt_shc
    #
    # @sim_primary_inj_alt_shc.setter
    # def sim_primary_inj_alt_shc(self, value):
    #     # A list was given
    #     if (
    #             isinstance(value, list)
    #             or isinstance(value, np.ndarray)
    #             or isinstance(value, StdVectorList)
    #     ):
    #         # Clear the vector before setting
    #         self._sim_primary_inj_alt_shc.clear()
    #         self._sim_primary_inj_alt_shc += value
    #     # A vector of strings was given
    #     elif isinstance(value, ROOT.vector("float")):
    #         self._sim_primary_inj_alt_shc._vector = value
    #     else:
    #         raise ValueError(
    #             f"Incorrect type for sim_primary_inj_alt_shc {type(value)}. Either a list, an array or a ROOT.vector of floats required."
    #         )
    #
    # @property
    # def sim_primary_inj_dir_shc(self):
    #     """Primary injection direction in Shower Coordinates"""
    #     return np.array(self._sim_primary_inj_dir_shc)
    #
    # @sim_primary_inj_dir_shc.setter
    # def sim_primary_inj_dir_shc(self, value):
    #     set_vector_of_vectors(value, "vector<float>", self._sim_primary_inj_dir_shc, "sim_primary_inj_dir_shc")

    @property
    def hadronic_model(self):
        """High energy hadronic model (and version) used"""
        return str(self._hadronic_model)

    @hadronic_model.setter
    def hadronic_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._hadronic_model.string.assign(value)

    @property
    def low_energy_model(self):
        """High energy model (and version) used"""
        return str(self._low_energy_model)

    @low_energy_model.setter
    def low_energy_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._low_energy_model.string.assign(value)

    @property
    def cpu_time(self):
        """Time it took for the sim"""
        return np.array(self._cpu_time)

    @cpu_time.setter
    def cpu_time(self, value):
        self._cpu_time = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("cpu_time", self._cpu_time)

    @property
    def long_depth(self):
        """Slant depth of the observing levels for longitudinal development tables"""
        return np.array(self._long_depth)

    @long_depth.setter
    def long_depth(self, value):
        set_vector_of_vectors(value, "float", self._long_depth, "long_depth")

    @property
    def long_eminus(self):
        """Number of electrons"""
        return np.array(self._long_eminus)

    @long_eminus.setter
    def long_eminus(self, value):
        set_vector_of_vectors(value, "float", self._long_eminus, "long_eminus")

    @property
    def long_eplus(self):
        """Number of electrons"""
        return np.array(self._long_eplus)

    @long_eplus.setter
    def long_eplus(self, value):
        set_vector_of_vectors(value, "float", self._long_eplus, "long_eplus")

    @property
    def long_muminus(self):
        """Number of electrons"""
        return np.array(self._long_muminus)

    @long_muminus.setter
    def long_muminus(self, value):
        set_vector_of_vectors(value, "float", self._long_muminus, "long_muminus")

    @property
    def long_muplus(self):
        """Number of electrons"""
        return np.array(self._long_muplus)

    @long_muplus.setter
    def long_muplus(self, value):
        set_vector_of_vectors(value, "float", self._long_muplus, "long_muplus")

    @property
    def long_gamma(self):
        """Number of electrons"""
        return np.array(self._long_gamma)

    @long_gamma.setter
    def long_gamma(self, value):
        set_vector_of_vectors(value, "float", self._long_gamma, "long_gamma")

    @property
    def long_hadron(self):
        """Number of electrons"""
        return np.array(self._long_hadron)

    @long_hadron.setter
    def long_hadron(self, value):
        set_vector_of_vectors(value, "float", self._long_hadron, "long_hadron")

    # @property
    # def long_gamma_elow(self):
    #     """Number of electrons"""
    #     return np.array(self._long_gamma_elow)
    #
    # @long_gamma_elow.setter
    # def long_gamma_elow(self, value):
    #     set_vector_of_vectors(value, "float", self._long_gamma_elow, "long_gamma_elow")
    #
    # @property
    # def long_e_elow(self):
    #     """Number of electrons"""
    #     return np.array(self._long_e_elow)
    #
    # @long_e_elow.setter
    # def long_e_elow(self, value):
    #     set_vector_of_vectors(value, "float", self._long_e_elow, "long_e_elow")
    #
    # @property
    # def long_e_edep(self):
    #     """Number of electrons"""
    #     return np.array(self._long_e_edep)
    #
    # @long_e_edep.setter
    # def long_e_edep(self, value):
    #     set_vector_of_vectors(value, "float", self._long_e_edep, "long_e_edep")
    #
    # @property
    # def long_mu_elow(self):
    #     """Number of electrons"""
    #     return np.array(self._long_mu_elow)
    #
    # @long_mu_elow.setter
    # def long_mu_elow(self, value):
    #     set_vector_of_vectors(value, "float", self._long_mu_elow, "long_mu_elow")
    #
    # @property
    # def long_mu_edep(self):
    #     """Number of electrons"""
    #     return np.array(self._long_mu_edep)
    #
    # @long_mu_edep.setter
    # def long_mu_edep(self, value):
    #     set_vector_of_vectors(value, "float", self._long_mu_edep, "long_mu_edep")
    #
    # @property
    # def long_hadron_elow(self):
    #     """Number of electrons"""
    #     return np.array(self._long_hadron_elow)
    #
    # @long_hadron_elow.setter
    # def long_hadron_elow(self, value):
    #     set_vector_of_vectors(value, "float", self._long_hadron_elow, "long_hadron_elow")
    #
    # @property
    # def long_hadron_edep(self):
    #     """Number of electrons"""
    #     return np.array(self._long_hadron_edep)
    #
    # @long_hadron_edep.setter
    # def long_hadron_edep(self, value):
    #     set_vector_of_vectors(value, "float", self._long_hadron_edep, "long_hadron_edep")
    #
    # @property
    # def long_neutrinos(self):
    #     """Number of electrons"""
    #     return np.array(self._long_neutrinos)
    #
    # @long_neutrinos.setter
    # def long_neutrinos(self, value):
    #     set_vector_of_vectors(value, "float", self._long_neutrinos, "long_neutrinos")

    @property
    def tested_core_positions(self):
        """Core positions tested for that shower to generate the event (effective area study)"""
        return np.array(self._tested_core_positions)

    @tested_core_positions.setter
    def tested_core_positions(self, value):
        set_vector_of_vectors(value, "vector<float>", self._tested_core_positions, "tested_core_positions")


## General info on the noise generation
@dataclass
class TRunNoise(MotherRunTree):
    """General info on the noise generation"""

    _type: str = "runnoise"

    _tree_name: str = "trunnoise"

    ## Info to retrieve the map of galactic noise
    _gal_noise_map: StdString = StdString("")
    ## LST time when we generate the noise
    _gal_noise_LST: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    ## Noise std dev for each arm of each antenna
    _gal_noise_sigma: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))

    @property
    def gal_noise_map(self):
        """Info to retrieve the map of galactic noise"""
        return str(self._gal_noise_map)

    @gal_noise_map.setter
    def gal_noise_map(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for gal_noise_map {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._gal_noise_map.string.assign(value)

    @property
    def gal_noise_LST(self):
        """LST time when we generate the noise"""
        return self._gal_noise_LST[0]

    @gal_noise_LST.setter
    def gal_noise_LST(self, value: np.float32) -> None:
        self._gal_noise_LST[0] = value

    @property
    def gal_noise_sigma(self):
        """Noise std dev for each arm of each antenna"""
        return np.array(self._gal_noise_sigma)

    @gal_noise_sigma.setter
    def gal_noise_sigma(self, value):
        set_vector_of_vectors(value, "vector<float>", self._gal_noise_sigma, "gal_noise_sigma")


# @dataclass
# ## The class for storing shower data for each event specific to ZHAireS only
# class ShowerEventZHAireSTree(MotherEventTree):
#     """The class for storing shower data for each event specific to ZHAireS only"""
#
#     _type: str = "eventshowerzhaires"
#
#     _tree_name: str = "teventshowerzhaires"
#
#     # ToDo: we need explanations of these parameters
#
#     _relative_thining: StdString = StdString("")
#     _weight_factor: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))
#     _gamma_energy_cut: StdString = StdString("")
#     _electron_energy_cut: StdString = StdString("")
#     _muon_energy_cut: StdString = StdString("")
#     _meson_energy_cut: StdString = StdString("")
#     _nucleon_energy_cut: StdString = StdString("")
#     _other_parameters: StdString = StdString("")
#
#     # def __post_init__(self):
#     #     super().__post_init__()
#     #
#     #     if self._tree.GetName() == "":
#     #         self._tree.SetName(self._tree_name)
#     #     if self._tree.GetTitle() == "":
#     #         self._tree.SetTitle(self._tree_name)
#     #
#     #     self.create_branches()
#
#     @property
#     def relative_thining(self):
#         """Relative thinning energy"""
#         return str(self._relative_thining)
#
#     @relative_thining.setter
#     def relative_thining(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for relative_thining {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._relative_thining.string.assign(value)
#
#     @property
#     def weight_factor(self):
#         """Weight factor"""
#         return self._weight_factor[0]
#
#     @weight_factor.setter
#     def weight_factor(self, value: np.float64) -> None:
#         self._weight_factor[0] = value
#
#     @property
#     def gamma_energy_cut(self):
#         """Low energy cut for gammas(GeV)"""
#         return str(self._gamma_energy_cut)
#
#     @gamma_energy_cut.setter
#     def gamma_energy_cut(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for gamma_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._gamma_energy_cut.string.assign(value)
#
#     @property
#     def electron_energy_cut(self):
#         """Low energy cut for electrons (GeV)"""
#         return str(self._electron_energy_cut)
#
#     @electron_energy_cut.setter
#     def electron_energy_cut(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for electron_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._electron_energy_cut.string.assign(value)
#
#     @property
#     def muon_energy_cut(self):
#         """Low energy cut for muons (GeV)"""
#         return str(self._muon_energy_cut)
#
#     @muon_energy_cut.setter
#     def muon_energy_cut(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for muon_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._muon_energy_cut.string.assign(value)
#
#     @property
#     def meson_energy_cut(self):
#         """Low energy cut for mesons (GeV)"""
#         return str(self._meson_energy_cut)
#
#     @meson_energy_cut.setter
#     def meson_energy_cut(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for meson_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._meson_energy_cut.string.assign(value)
#
#     @property
#     def nucleon_energy_cut(self):
#         """Low energy cut for nucleons (GeV)"""
#         return str(self._nucleon_energy_cut)
#
#     @nucleon_energy_cut.setter
#     def nucleon_energy_cut(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for nucleon_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._nucleon_energy_cut.string.assign(value)
#
#     @property
#     def other_parameters(self):
#         """Other parameters"""
#         return str(self._other_parameters)
#
#     @other_parameters.setter
#     def other_parameters(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for other_parameters {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._other_parameters.string.assign(value)


# ## A class wrapping around TTree describing the detector information, like position, type, etc. It works as an array for readout in principle
# @dataclass
# class DetectorInfo(DataTree):
#     """A class wrapping around TTree describing the detector information, like position, type, etc. It works as an array for readout in principle"""
#
#     _tree_name: str = "tdetectorinfo"
#
#     ## Detector Unit id
#     _du_id: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
#     ## Currently read out unit. Not publicly visible
#     _cur_du_id: int = -1
#     ## Detector longitude
#     _long: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
#     ## Detector latitude
#     _lat: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
#     ## Detector altitude
#     _alt: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
#     ## Detector type
#     _type: StdString = StdString("antenna")
#     ## Detector description
#     _description: StdString = StdString("")
#
#     def __post_init__(self):
#         super().__post_init__()
#
#         if self._tree.GetName() == "":
#             self._tree.SetName(self._tree_name)
#         if self._tree.GetTitle() == "":
#             self._tree.SetTitle(self._tree_name)
#
#         self.create_branches()
#
#     def __len__(self):
#         return self._tree.GetEntries()
#
#     # def __delitem__(self, index):
#     #     self.vector.erase(index)
#     #
#     # def insert(self, index, value):
#     #     self.vector.insert(index, value)
#     #
#     # def __setitem__(self, index, value):
#     #     self.vector[index] = value
#
#     def __getitem__(self, index):
#         # Read the unit if not read already
#         if self._cur_du_id != index:
#             self._tree.GetEntryWithIndex(index)
#             self._cur_du_id = index
#         return self
#
#     ## Don't really add friends, just generates an index
#     def add_proper_friends(self):
#         """Don't really add friends, just generates an index"""
#         # Create the index
#         self._tree.BuildIndex("du_id")
#
#     ## Fill the tree
#     def fill(self):
#         """Fill the tree"""
#         self._tree.Fill()
#
#     @property
#     def du_id(self):
#         """Detector Unit id"""
#         return self._du_id[0]
#
#     @du_id.setter
#     def du_id(self, value: int) -> None:
#         self._du_id[0] = value
#
#     @property
#     def long(self):
#         """Detector longitude"""
#         return self._long[0]
#
#     @long.setter
#     def long(self, value: np.float32) -> None:
#         self._long[0] = value
#
#     @property
#     def lat(self):
#         """Detector latitude"""
#         return self._lat[0]
#
#     @lat.setter
#     def lat(self, value: np.float32) -> None:
#         self._lat[0] = value
#
#     @property
#     def alt(self):
#         """Detector altitude"""
#         return self._alt[0]
#
#     @alt.setter
#     def alt(self, value: np.float32) -> None:
#         self._alt[0] = value
#
#     @property
#     def type(self):
#         """Detector type"""
#         return str(self._type)
#
#     @type.setter
#     def type(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for type {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._type.string.assign(value)
#
#     @property
#     def description(self):
#         """Detector description"""
#         return str(self._description)
#
#     @description.setter
#     def description(self, value):
#         # Not a string was given
#         if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
#             raise ValueError(
#                 f"Incorrect type for description {type(value)}. Either a string or a ROOT.std.string is required."
#             )
#
#         self._description.string.assign(value)


## Exception raised when an already existing event/run is added to a tree
class NotUniqueEvent(Exception):
    """Exception raised when an already existing event/run is added to a tree"""

    pass


## General setter for vectors of vectors
def set_vector_of_vectors(value, vec_type, variable, variable_name):
    # A list was given
    if isinstance(value, list) or isinstance(value, np.ndarray) or isinstance(value, StdVectorList):
        # Clear the vector before setting
        variable.clear()
        variable += value
    # A vector of vectors was given
    elif isinstance(value, ROOT.vector(vec_type)):
        variable._vector = value
    else:
        if "vector" in vec_type:
            raise ValueError(
                f"Incorrect type for {variable_name} {type(value)}. Either a list of lists, a list of arrays or a ROOT.vector of vectors required."
            )
        else:
            raise ValueError(
                f"Incorrect type for {variable_name} {type(value)}. Either a list, an array or a ROOT.vector required."
            )


## Class holding the information about GRAND data in a directory
class DataDirectory:
    """Class holding the information about GRAND data in a directory"""

    def __init__(self, dir_name: str, recursive: bool = False):
        """
        @param dir_name: the name of the directory to be scanned
        @param recursive: if to scan the directory recursively
        """

        # Make the path absolute
        self.dir_name = os.path.abspath(dir_name)

        # Get the file list
        self.file_list = self.get_list_of_files(recursive)

        # Get the file handle list
        self.file_handle_list = self.get_list_of_files_handles()

        # Create chains and set them as attributes
        self.create_chains()

    def get_list_of_files(self, recursive: bool = False):
        """Gets list of files in the directory"""
        return sorted(glob.glob(os.path.join(self.dir_name, "*.root"), recursive=recursive))

    def get_list_of_files_handles(self):
        """Go through the list of files in the directory and open all of them"""
        file_handle_list = []

        for filename in self.file_list:
            file_handle_list.append(DataFile(filename))

        return file_handle_list

    def create_chains(self):
        chains_dict = {}
        tree_types = set()
        # Loop through the list of file handles
        for i, f in enumerate(self.file_handle_list):
            # Collect the tree types
            tree_types.update(*f.tree_types.keys())

            # Select the highest analysis level trees for each class and store these trees as main attributes
            for key in f.tree_types:
                if key == "run":
                    setattr(self, "trun", f.dict_of_trees["trun"])
                else:
                    max_analysis_level = -1
                    for key1 in f.tree_types[key].keys():
                        el = f.tree_types[key][key1]
                        chain_name = el["name"]
                        if "analysis_level" in el:
                            if el["analysis_level"] > max_analysis_level or el["analysis_level"] == 0:
                                max_analysis_level = el["analysis_level"]
                                max_anal_chain_name = chain_name

                                setattr(self, key + "_" + str(el["analysis_level"]), f.dict_of_trees[el["name"]])

                        if chain_name not in chains_dict:
                            chains_dict[chain_name] = ROOT.TChain(chain_name)
                        chains_dict[chain_name].Add(self.file_list[i])

                        # In case there is no analysis level info in the tree (old trees), just take the last one
                        if max_analysis_level == -1:
                            max_anal_chain_name = el["name"]

                    setattr(self, "t" + key, chains_dict[max_anal_chain_name])

    def print(self, recursive=True):
        """Prints all the information about all the data"""
        pass

    def get_list_of_chains(self):
        """Gets list of TTree chains of specific type from the directory"""
        pass

    def print_list_of_chains(self):
        """Prints list of TTree chains of specific type from the directory"""
        pass


class DataFile:
    """Class holding the information about GRAND TTrees in the specified file"""

    # Holds all the trees in the file, by tree name
    dict_of_trees = {}
    # Holds the list of trees in the file, but just with maximal level
    list_of_trees = []
    # Holds dict of tree types, each containing a dict of tree names with tree meta-data as values
    tree_types = defaultdict(dict)

    def __init__(self, filename):
        """filename can be either a string or a ROOT.TFile"""
        # If a string given, open the file
        if type(filename) is str:
            f = ROOT.TFile(filename)
            self.f = f
        elif type(filename) is ROOT.TFile:
            self.f = filename
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

            # Add the tree to a dict for this tree class
            self.tree_types[tree_info["type"]][tree_info["name"]] = tree_info

            self.dict_of_trees[tree_info["name"]] = t

        # Select the highest analysis level trees for each class and store these trees as main attributes
        # Loop through tree types
        for key in self.tree_types:
            if key == "run":
                setattr(self, "trun", self.dict_of_trees["trun"])
            else:
                max_analysis_level = -1
                # Loop through trees in the current tree type
                for key1 in self.tree_types[key].keys():
                    el = self.tree_types[key][key1]
                    tree_class = getattr(thismodule, el["type"])
                    tree_instance = tree_class(_tree_name=self.dict_of_trees[el["name"]])
                    # If there is analysis level info in the tree, attribute each level and max level
                    if "analysis_level" in el:
                        if el["analysis_level"] > max_analysis_level or el["analysis_level"] == 0:
                            max_analysis_level = el["analysis_level"]
                            max_anal_tree_name = el["name"]
                            max_anal_tree_type = el["type"]
                        setattr(self, tree_class.get_default_tree_name() + "_" + str(el["analysis_level"]), tree_instance)
                    # In case there is no analysis level info in the tree (old trees), just take the last one
                    elif max_analysis_level == -1:
                        max_anal_tree_name = el["name"]
                        max_anal_tree_type = el["type"]

                    traces_lenghts = self._get_traces_lengths(tree_instance)
                    if traces_lenghts is not None:
                        el["traces_lengths"] = traces_lenghts

                    dus = self._get_list_of_all_used_dus(tree_instance)
                    if dus is not None:
                        el["dus"] = dus

                    el["mem_size"], el["disk_size"] = tree_instance.get_tree_size()

                tree_class = getattr(thismodule, max_anal_tree_type)
                tree_instance = tree_class(_tree_name=self.dict_of_trees[max_anal_tree_name])
                setattr(self, tree_class.get_default_tree_name(), tree_instance)
                self.list_of_trees.append(self.dict_of_trees[max_anal_tree_name])

    def print(self):
        """Prints the information about the TTrees in the file"""

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
