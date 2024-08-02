"""
Read/Write python interface to GRAND data (real and simulated) stored in Cern ROOT TTrees.

This is the interface for accessing GRAND ROOT TTrees that do not require the user (reader/writer of the TTrees) to have any knowledge of ROOT. It also hides the internals from the data generator, so that the changes in the format are not concerning the user.
"""

from logging import getLogger
import sys
import datetime
import os
from pathlib import Path

import ROOT
import numpy as np
import glob
import array

from collections import defaultdict

# Load the C++ macros for vector filling from numpy arrays
ROOT.gROOT.LoadMacro(os.path.dirname(os.path.realpath(__file__))+"/vector_filling.C")

# Conversion between numpy dtype and array.array typecodes
numpy_to_array_typecodes = {np.dtype('int8'): 'b', np.dtype('int16'): 'h', np.dtype('int32'): 'i', np.dtype('int64'): 'q', np.dtype('uint8'): 'B', np.dtype('uint16'): 'H', np.dtype('uint32'): 'I', np.dtype('uint64'): 'Q', np.dtype('float32'): 'f', np.dtype('float64'): 'd', np.dtype('complex64'): 'F', np.dtype('complex128'): 'D', np.dtype('int16'): 'h'}
# numpy_to_array_typecodes = {np.int8: 'b', np.int16: 'h', np.int32: 'i', np.int64: 'q', np.uint8: 'B', np.uint16: 'H', np.uint32: 'I', np.uint64: 'Q', np.float32: 'f', np.float64: 'd', np.complex64: 'F', np.complex128: 'D', "int8": 'b', "int16": 'h', "int32": 'i', "int64": 'q', "uint8": 'B', "uint16": 'H', "uint32": 'I', "uint64": 'Q', "float32": 'f', "float64": 'd', "complex64": 'F', "complex128": 'D'}

# Conversion between C++ type and array.array typecodes
cpp_to_array_typecodes = {'char': 'b', 'short': 'h', 'int': 'i', 'long long': 'q', 'unsigned char': 'B', 'unsigned short': 'H', 'unsigned int': 'I', 'unsigned long long': 'Q', 'float': 'f', 'double': 'd', 'string': 'u'}

cpp_to_numpy_typecodes = {'char': np.dtype('int8'), 'short': np.dtype('int16'), 'int': np.dtype('int32'), 'long long': np.dtype('int64'), 'unsigned char': np.dtype('uint8'), 'unsigned short': np.dtype('uint16'), 'unsigned int': np.dtype('uint32'), 'unsigned long long': np.dtype('uint64'), 'float': np.dtype('float32'), 'double': np.dtype('float64'), 'string': np.dtype('U')}

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

    def __init__(self, vec_type, value=[], sec_vec_type=None):
        """
        Args:
            vec_type: C++ type for the std::vector (eg. "float", "string", etc.)
            value: list with which to init the vector
            sec_vec_type: second possible vector type to switch to. Needed in case of branch of specific name having two possible types due to changes in the hardware binary blob format and maitaining compatibility through such changes.
        """
        self._vector = ROOT.vector(vec_type)(value)
        if sec_vec_type is not None:
            self._sec_vector = ROOT.vector(sec_vec_type)()
        #: C++ type for the std::vector (eg. "float", "string", etc.)
        self.vec_type = vec_type
        self.sec_vec_type = sec_vec_type
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
                try:
                    return np.array(self._vector[index])
                except:
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
    # function modified by Jelena to fix the negative issue, use at own risk
        try:
            if isinstance(value, np.ndarray):
                if self.ndim == 1: ROOT.fill_vec_1D[self.basic_vec_type](np.ascontiguousarray(value), np.array(value.shape).astype(np.int32), self._vector)
                if self.ndim == 2: ROOT.fill_vec_2D[self.basic_vec_type](np.ascontiguousarray(value), np.array(value.shape).astype(np.int32), self._vector)
                if self.ndim == 3: ROOT.fill_vec_3D[self.basic_vec_type](np.ascontiguousarray(value), np.array(value.shape).astype(np.int32), self._vector)
            else:
                if (isinstance(value, list) and self.basic_vec_type.split()[-1] == "float"):
                    if self.ndim == 1: value = array.array(cpp_to_array_typecodes[self.basic_vec_type], value)
                    if self.ndim == 2: value = [array.array(cpp_to_array_typecodes[self.basic_vec_type], el) for el in value]
                    if self.ndim == 3: value = [[array.array(cpp_to_array_typecodes[self.basic_vec_type], el1) for el1 in el] for el in value]
                elif not isinstance(value, StdVectorList):
                    value = list(value)

                # The list needs to have simple Python types - ROOT.vector does not accept numpy types
                try:
                    if isinstance(value, StdVectorList):
                        # ToDo: Maybe faster than +=, but... to be checked
                        self._vector.assign(value._vector)
                    else:
                        self._vector += value
                except TypeError:
                    # Slow conversion to simple types. No better idea for now
                    if self.basic_vec_type.split()[-1] in ["int", "long", "short", "char", "float"]:
                        if self.ndim == 1: value = array.array(cpp_to_array_typecodes[self.basic_vec_type], value)
                        if self.ndim == 2: value = [array.array(cpp_to_array_typecodes[self.basic_vec_type], el) for el in value]
                        if self.ndim == 3: value = [[array.array(cpp_to_array_typecodes[self.basic_vec_type], el1) for el1 in el] for el in value]

                    self._vector += value

        except OverflowError:
            # Handle the OverflowError here, e.g., by logging a message or taking an appropriate action.
            if isinstance(value, (list, np.ndarray)):
                # Use signed integer types to allow for negative values
                signed_type = 'l' if self.basic_vec_type.split()[-1] == "int" else self.basic_vec_type
                if self.ndim == 1:
                    value = array.array(signed_type, value)
                if self.ndim == 2:
                    value = [array.array(signed_type, el) for el in value]
                if self.ndim == 3:
                    value = [[array.array(signed_type, el1) for el1 in el] for el in value]
            else:
                value = list(value)

        return self

    def switch_to_sec_vec_type(self):
        """
        Change the vector type to sec_vec_type. Expected to be ran only on branch creation.
        """
        self._vector = self._sec_vector
        #: C++ type for the std::vector (eg. "float", "string", etc.)
        self.vec_type = self.sec_vec_type
        # Basic type of the vector - different than vec_type in case of vector of vectors
        if "<" in self.vec_type:
            self.basic_vec_type = self.vec_type.split("<")[-1].split(">")[0]
        else:
            self.basic_vec_type = self.vec_type
        # The number of dimensions of this vector
        self.ndim = self.vec_type.count("vector") + 1

class StdVectorListDesc:
    """A descriptor for StdVectorList - makes use of it possible in dataclasses without setting property and setter"""
    def __init__(self, vec_type, sec_vec_type=None):
        self.factory = lambda: StdVectorList(vec_type, sec_vec_type=sec_vec_type)

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

        inst[0] = value

class TTreeArrayDesc:
    """A descriptor for numpy arrays stored in TTrees. Ensures the type and converts to array (in case of for eg. list). Makes use of it possible in dataclasses without setting property and setter"""
    def __init__(self, shape, dtype):
        self.factory = lambda: np.zeros(shape, dtype)
        self.dtype = dtype

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
        if isinstance(value, TTreeArrayDesc):
            value = getattr(obj, self.attrname)
        inst = getattr(obj, self.attrname)

        inst[:] = np.array(value).astype(self.dtype)

class StdString:
    """A python string interface to ROOT's std::string"""

    def __init__(self, value):
        self.string = ROOT.string(value)

    def __len__(self):
        return len(str(self.string))

    def __repr__(self):
        return str(self.string)

class StdStringDesc:
    """A descriptor for strings assigned to TTrees as python strings - makes use of it possible in dataclasses without setting property and setter"""

    def __init__(self, value=""):
        self.factory = lambda: ROOT.string(value)

    def __set_name__(self, type, name):
        self.name = name
        self.attrname = f"_{name}"

    def create_default(self, obj):
        setattr(obj, self.attrname, self.factory())

    def __get__(self, obj, obj_type):
        if not hasattr(obj, self.attrname):
            self.create_default(obj)
        return str(getattr(obj, self.attrname))

    def __set__(self, obj, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string) or isinstance(value, StdStringDesc)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        if not hasattr(obj, self.attrname):
            self.create_default(obj)
        # This is needed for default init as a field of an upper class
        if isinstance(value, StdStringDesc):
            value = getattr(obj, self.attrname)
        inst = getattr(obj, self.attrname)

        inst.assign(value)


@dataclass
class DataTree:
    """
    Mother class for GRAND Tree data classes
    """

    ## File handle
    _file: ROOT.TFile = None
    """File handle"""
    ## File name
    _file_name: str = None
    """File name"""
    ## Tree object
    _tree: ROOT.TTree = None
    """Tree object"""
    ## Tree name
    _tree_name: str = ""
    """Tree name"""
    ## Tree type
    _type: str = ""
    """Tree type"""
    ## A list of run_numbers or (run_number, event_number) pairs in the Tree
    _entry_list: list = field(default_factory=list)
    """A list of run_numbers or (run_number, event_number) pairs in the Tree"""
    ## Comment - if needed, added by user
    _comment: str = ""
    """Comment - if needed, added by user"""
    ## TTree creation date/time in UTC - a naive time, without timezone set
    _creation_datetime: datetime.datetime = None
    """TTree creation date/time in UTC - a naive time, without timezone set"""
    ## Modification history - JSON
    _modification_history: str = ""
    """Modification history - JSON"""

    ## Unix creation datetime of the source tree; 0 s means no source
    _source_datetime: datetime.datetime = None
    """Unix creation datetime of the source tree; 0 s means no source"""
    ## The tool used to generate this tree's values from another tree
    _modification_software: str = ""
    """The tool used to generate this tree's values from another tree"""
    ## The version of the tool used to generate this tree's values from another tree
    _modification_software_version: str = ""
    """The version of the tool used to generate this tree's values from another tree"""
    ## The analysis level of this tree
    _analysis_level: int = 0
    """The analysis level of this tree"""


    ## Fields that are not branches
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
    """Fields that are not branches"""

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
        """Return the iterable over self"""
        # Always start the iteration with the first entry
        current_entry = 0

        while current_entry < self._tree.GetEntries():
            self.get_entry(current_entry)
            yield self
            current_entry += 1

    ## Set the tree's file
    def _set_file(self, f):
        """Set the tree's file"""
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
        """Init/readout the tree from a file"""
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

                # Make the tree save itself in this file
                self._tree.SetDirectory(self._file)

            else:
                logger.info(f"creating tree {self._tree_name} {self._file}")
                self._create_tree()


        self.assign_metadata()

        # Fill the runs/events numbers from the tree (important if it already existed)
        self.fill_entry_list()

    ## Create the tree
    def _create_tree(self, tree_name=""):
        """Create the tree"""
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
            # print(field, self.__dict__[field], type(self.__dict__[field]))
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
        elif type(value) == StdVectorList or type(value) == StdVectorListDesc:
            # For two-type vectors, check if a switch to the second vector type is needed
            if getattr(self, value_name).sec_vec_type is not None:
                # If the second vector type is the type of the branch, switch to the second vector type
                if self._tree.GetLeaf(branch_name) != None:
                    if getattr(self, value_name).sec_vec_type in self._tree.GetLeaf(branch_name).GetTypeName():
                        getattr(self, value_name).switch_to_sec_vec_type()
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
                    logger.info(f"Could not find branch {branch_name} in tree {self.tree_name}. This branch will not be filled.")
        elif type(value) == StdString:
            # Create the branch
            if not set_branches:
                # self._tree.Branch(value.name[1:], getattr(self, value.name).string)
                self._tree.Branch(branch_name, getattr(self, value_name).string)
            # Or set its address
            else:
                # self._tree.SetBranchAddress(value.name[1:], getattr(self, value.name).string)
                self._tree.SetBranchAddress(branch_name, getattr(self, value_name).string)
        elif isinstance(value, ROOT.string):
            # Create the branch
            if not set_branches:
                # self._tree.Branch(value.name[1:], getattr(self, value.name).string)
                self._tree.Branch(branch_name, getattr(self, value_name))
            # Or set its address
            else:
                # self._tree.SetBranchAddress(value.name[1:], getattr(self, value.name).string)
                self._tree.SetBranchAddress(branch_name, getattr(self, value_name))
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
            try:
                u = getattr(self._tree, field_name)
                # print("*", field[1:], self.__dataclass_fields__[field].name, u, type(u), id(u))
            except:
                logger.info(f"Could not find {field_name} in tree {self.tree_name}. This field won't be assigned.")
            else:
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
        # ToDo: stupid, because default values are generated here and in the class fields definitions. But definition of the class field does not call the setter, which is needed to attach these fields to the tree.
        self.source_datetime = datetime.datetime.fromtimestamp(0)
        self.modification_software = ""
        self.modification_software_version = ""
        self.analysis_level = 0

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

    def stop_using(self):
        """Stop using this instance: remove it from the tree list"""
        grand_tree_list.remove(self)


## A mother class for classes with Run values
@dataclass
class MotherRunTree(DataTree):
    """A mother class for classes with Run values"""

    run_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))

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

    run_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    # ToDo: it seems instances propagate this number among them without setting (but not the run number!). I should find why...
    event_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    # ## Create metadata for the tree
    # def create_metadata(self):
    #     """Create metadata for the tree"""
    #     # First add the medatata of the mother class
    #     super().create_metadata()
    #     # ToDo: stupid, because default values are generated here and in the class fields definitions. But definition of the class field does not call the setter, which is needed to attach these fields to the tree.
    #     self.source_datetime = datetime.datetime.fromtimestamp(0)
    #     self.modification_software = ""
    #     self.modification_software_version = ""
    #     self.analysis_level = 0

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
        # For now, do not add friends
        return 0

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

    def get_dus_indices_in_run(self, trun):
        """Gets an array of the indices of DUs of the current event in the TRun tree"""

        return np.nonzero(np.isin(np.asarray(trun.du_id), np.asarray(self.du_id)))[0]


## A class wrapping around a TTree holding values common for the whole run
@dataclass
class TRun(MotherRunTree):
    """A class wrapping around a TTree holding values common for the whole run"""

    _type: str = "run"

    _tree_name: str = "trun"

    ## Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment
    run_mode: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment"""
    ## Run's first event
    first_event: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Run's first event"""
    ## First event time
    first_event_time: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """First event time"""
    ## Run's last event
    last_event: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Run's last event"""
    ## Last event time
    last_event_time: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Last event time"""

    # These are not from the hardware
    ## Data source: detector, sim, other
    data_source: StdStringDesc = field(default=StdStringDesc("detector"))
    """Data source: detector, sim, other"""
    ## Data generator: gtot (in this case)
    data_generator: StdStringDesc = field(default=StdStringDesc("GRANDlib"))
    """Data generator: gtot (in this case)"""
    ## Generator version: gtot version (in this case)
    data_generator_version: StdStringDesc = field(default=StdStringDesc("0.1.0"))
    """Generator version: gtot version (in this case)"""
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    event_type: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
    ## Event format version of the DAQ
    event_version: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Event format version of the DAQ"""
    ## Site name
    # _site: StdVectorList("string") = StdVectorList("string")
    site: StdStringDesc = field(default=StdStringDesc())
    """Site name"""
    ## Site layout
    site_layout: StdStringDesc = field(default=StdStringDesc())
    """Site layout"""
    ## Origin of the coordinate system used for the array
    origin_geoid: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Origin of the coordinate system used for the array"""

    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("int"))
    """Detector unit (antenna) ID"""
    ## Detector unit (antenna) (lat,lon,alt) position
    du_geoid: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Detector unit (antenna) (lat,lon,alt) position"""
    ## Detector unit (antenna) (x,y,z) position in site's referential
    du_xyz: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Detector unit (antenna) (x,y,z) position in site's referential"""
    ## Detector unit type
    du_type: StdVectorListDesc = field(default=StdVectorListDesc("string"))
    """Detector unit type"""
    ## Detector unit (antenna) angular tilt
    du_tilt: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Detector unit (antenna) angular tilt"""
    ## Angular tilt of the ground at the antenna
    du_ground_tilt: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Angular tilt of the ground at the antenna"""
    ## Detector unit (antenna) nut ID
    du_nut: StdVectorListDesc = field(default=StdVectorListDesc("int"))
    """Detector unit (antenna) nut ID"""
    ## Detector unit (antenna) FrontEnd Board ID
    du_feb: StdVectorListDesc = field(default=StdVectorListDesc("int"))
    """Detector unit (antenna) FrontEnd Board ID"""
    ## Time bin size in ns (for hardware, computed as 1/adc_sampling_frequency)
    t_bin_size: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Time bin size in ns (for hardware, computed as 1/adc_sampling_frequency)"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


## General info on the voltage common to all events.
@dataclass
class TRunVoltage(MotherRunTree):
    """General info on the voltage common to all events."""

    _type: str = "runvoltage"

    _tree_name: str = "trunvoltage"

    ## Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual).
    digi_ctrl: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual)."""
    ## Firmware version
    firmware_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Firmware version"""
    ## Nominal trace length in units of samples
    trace_length: StdVectorListDesc = field(default=StdVectorListDesc("vector<int>"))
    """Nominal trace length in units of samples"""
    ## ADC sampling frequency in MHz
    adc_sampling_frequency: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling frequency in MHz"""
    ## ADC sampling resolution in bits
    adc_sampling_resolution: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling resolution in bits"""
    ## ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?
    adc_input_channels: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?"""
    ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?
    adc_enabled_channels: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?"""
    ## Value of the Variable gain amplification on the board
    gain: StdVectorListDesc = field(default=StdVectorListDesc("vector<int>"))
    """Value of the Variable gain amplification on the board"""
    ## Conversion factor from bits to V for ADC
    adc_conversion: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Conversion factor from bits to V for ADC"""
    ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    digi_prepost_trig_windows: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?"""
    ## Channel x properties - described in Channel property parameters in the manual. ToDo: Decode?
    channel_properties_x: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel x properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    ## Channel y properties - described in Channel property parameters in the manual. ToDo: Decode?
    channel_properties_y: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel y properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    ## Channel z properties - described in Channel property parameters in the manual. ToDo: Decode?
    channel_properties_z: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel z properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    ## Channel x trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    channel_trig_settings_x: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel x trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    ## Channel y trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    channel_trig_settings_y: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel y trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    ## Channel z trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    channel_trig_settings_z: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel z trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


@dataclass
## The class for storing ADC traces and associated values for each event
class TADC(MotherEventTree):
    """The class for storing ADC traces and associated values for each event"""

    _type: str = "adc"

    _tree_name: str = "tadc"

    ## Common for the whole event
    ## Event size
    """Common for the whole event"""
    event_size: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    ## Event in the run number
    t3_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Event in the run number"""
    ## First detector unit that triggered in the event
    first_du: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """First detector unit that triggered in the event"""
    ## Unix time corresponding to the GPS seconds of the first triggered station
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Unix time corresponding to the GPS seconds of the first triggered station"""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """GPS nanoseconds corresponding to the trigger of the first triggered station"""
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    event_type: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
    ## Event format version of the DAQ
    event_version: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Event format version of the DAQ"""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detector units in the event - basically the antennas count"""

    ## Specific for each Detector Unit
    ## The T3 trigger number
    event_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """The T3 trigger number"""
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Detector unit (antenna) ID"""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Unix time of the trigger for this DU"""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Nanoseconds of the trigger for this DU"""
    ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger position in the trace (trigger start = nanoseconds - 2*sample number)"""
    ## Same as event_type, but event_type could consist of different triggered DUs
    trigger_flag: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Same as event_type, but event_type could consist of different triggered DUs"""
    ## Atmospheric temperature (read via I2C)
    atm_temperature: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Atmospheric temperature (read via I2C)"""
    ## Atmospheric pressure
    atm_pressure: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Atmospheric pressure"""
    ## Atmospheric humidity
    atm_humidity: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Atmospheric humidity"""
    ## Acceleration of the antenna in X
    acceleration_x: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Acceleration of the antenna in X"""
    ## Acceleration of the antenna in Y
    acceleration_y: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Acceleration of the antenna in Y"""
    ## Acceleration of the antenna in Z
    acceleration_z: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Acceleration of the antenna in Z"""
    ## Battery voltage
    battery_level: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Battery voltage"""
    ## Firmware version
    firmware_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Firmware version"""
    ## ADC sampling frequency in MHz
    adc_sampling_frequency: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling frequency in MHz"""
    ## ADC sampling resolution in bits
    adc_sampling_resolution: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling resolution in bits"""

    adc_input_channels_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """ADC input channels"""

    adc_enabled_channels_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """ADC enabled channels"""

    adc_samples_count_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    trigger_pattern_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    trigger_pattern_ch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_notch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_redch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_ch2_ch3: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_calibration: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_10s: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_external_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))

    ## Trigger rate - the number of triggers recorded in the second preceding the event
    trigger_rate: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger rate - the number of triggers recorded in the second preceding the event"""
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    clock_tick: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock tick at which the event was triggered (used to calculate the trigger time)"""
    ## Clock ticks per second
    clock_ticks_per_second: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock ticks per second"""
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    gps_offset: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?"""
    ## GPS leap second
    gps_leap_second: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS leap second"""
    ## GPS status
    gps_status: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS status"""
    ## GPS alarms
    gps_alarms: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS alarms"""
    ## GPS warnings
    gps_warnings: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS warnings"""
    ## GPS time
    gps_time: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """GPS time"""
    ## Longitude
    gps_long: StdVectorListDesc = field(default=StdVectorListDesc("unsigned long long"))
    """Longitude"""
    ## Latitude
    gps_lat: StdVectorListDesc = field(default=StdVectorListDesc("unsigned long long"))
    """Latitude"""
    ## Altitude
    gps_alt: StdVectorListDesc = field(default=StdVectorListDesc("unsigned long long"))
    """Altitude"""
    ## GPS temperature
    gps_temp: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """GPS temperature"""

    ## Digital control register
    enable_auto_reset_timeout: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Digital control register"""
    force_firmware_reset: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_filter_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    enable_1PPS: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_DAQ: StdVectorListDesc = field(default=StdVectorListDesc("bool"))

    ## Trigger enable mask register
    enable_trigger_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """Trigger enable mask register"""
    enable_trigger_ch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_notch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_redch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_ch2_ch3: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_calibration: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_10s: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_external_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))

    ## Test pulse rate divider and channel readout enable
    enable_readout_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """Test pulse rate divider and channel readout enable"""
    fire_single_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    test_pulse_rate_divider: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))

    ## Common coincidence readout time window
    common_coincidence_time: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Common coincidence readout time window"""

    ## Input selector for readout channel
    selector_readout_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """Input selector for readout channel"""

    pre_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    post_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    gain_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    integration_time_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>","vector<unsigned char>"))
    offset_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    base_maximum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    base_minimum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    signal_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    noise_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    tper_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    tprev_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    ncmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    tcmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    qmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    ncmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    qmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))

    ## ?? What is it? Some kind of the adc trace offset?
    ioff: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """?? What is it? Some kind of the adc trace offset?"""

    ## ADC traces for channels (0,1,2,3)
    trace_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<short>>"))
    """ADC traces for channels (0,1,2,3)"""

    ## PPS-ID
    pps_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """PPS-ID"""

    ## FPGA temperature
    fpga_temp: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """FPGA temperature"""

    ## ADC temperature
    adc_temp: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """ADC temperature"""

    ## Hardware ID
    hardware_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Hardware ID"""

    ## Trigger status
    trigger_status: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger status"""

    ## Trigger DDR storage
    trigger_ddr_storage: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger DDR storage"""

    ## Data format version
    data_format_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Data format version"""

    ## ADAQ version
    adaq_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADAQ version"""

    ## DUDAQ version
    dudaq_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """DUDAQ version"""

    ## Trigger selection: ch0&ch1&ch2
    trigger_pattern_ch0_ch1_ch2: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Trigger selection: ch0&ch1&ch2"""

    ## Trigger selection: ch0&ch1&~ch2
    trigger_pattern_ch0_ch1_notch2: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Trigger selection: ch0&ch1&~ch2"""

    ## Trigger selection: 20 Hz
    trigger_pattern_20Hz: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Trigger selection: 20 Hz"""

    ## External pulse trigger period
    trigger_external_test_pulse_period: StdVectorListDesc = field(default=StdVectorListDesc("int"))
    """External pulse trigger period"""

    ## GPS seconds since Sunday 00:00
    gps_sec_sun: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """GPS seconds since Sunday 00:00"""

    ## GPS week number
    gps_week_num: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS week number"""

    ## GPS receiver mode
    gps_receiver_mode: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))
    """GPS receiver mode"""

    ## GPS disciplining mode
    gps_disciplining_mode: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))
    """GPS disciplining mode"""

    ## GPS self-survey progress
    gps_self_survey: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))
    """GPS self-survey progress"""

    ## GPS minor alarms
    gps_minor_alarms: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS minor alarms"""

    ## GPS GNSS decoding
    gps_gnss_decoding: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))
    """GPS GNSS decoding"""

    ## GPS disciplining activity
    gps_disciplining_activity: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS disciplining activity"""

    ## Notch filter number
    notch_filters_no_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """Notch filter number"""

@dataclass
## The class for storing voltage traces and associated values for each event
class TRawVoltage(MotherEventTree):
    """The class for storing voltage traces and associated values at ADC input level for each event. Derived from TADC but in human readable format and physics units."""

    _type: str = "rawvoltage"

    _tree_name: str = "trawvoltage"
    ## Common for the whole event
    ## Event size
    """Common for the whole event"""
    event_size: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    ## First detector unit that triggered in the event
    first_du: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """First detector unit that triggered in the event"""
    ## Unix time corresponding to the GPS seconds of the trigger
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Unix time corresponding to the GPS seconds of the trigger"""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """GPS nanoseconds corresponding to the trigger of the first triggered station"""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detector units in the event - basically the antennas count"""

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Detector unit (antenna) ID"""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Unix time of the trigger for this DU"""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Nanoseconds of the trigger for this DU"""
    ## Same as event_type, but event_type could consist of different triggered DUs
    trigger_flag: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Same as event_type, but event_type could consist of different triggered DUs"""
    ## Trigger position in the trace, in samples
    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger position in the trace, in samples"""
    ## Atmospheric temperature (read via I2C)
    atm_temperature: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Atmospheric temperature (read via I2C)"""
    ## Atmospheric pressure
    atm_pressure: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Atmospheric pressure"""
    ## Atmospheric humidity
    atm_humidity: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Atmospheric humidity"""
    ## Acceleration of the antenna in (x,y,z) in m/s2
    du_acceleration: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Acceleration of the antenna in (x,y,z) in m/s2"""
    ## Battery voltage
    battery_level: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Battery voltage"""
    ## ADC samples callected in channels (0,1,2,3)
    adc_samples_count_channel: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """ADC samples callected in channels (0,1,2,3)"""
    ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    trigger_pattern: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?"""
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    trigger_rate: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger rate - the number of triggers recorded in the second preceding the event"""
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    clock_tick: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock tick at which the event was triggered (used to calculate the trigger time)"""
    ## Clock ticks per second
    clock_ticks_per_second: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock ticks per second"""
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    gps_offset: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?"""
    ## GPS leap second
    gps_leap_second: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS leap second"""
    ## GPS status
    gps_status: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS status"""
    ## GPS alarms
    gps_alarms: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS alarms"""
    ## GPS warnings
    gps_warnings: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS warnings"""
    ## GPS time
    gps_time: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """GPS time"""
    ## Longitude
    gps_long: StdVectorListDesc = field(default=StdVectorListDesc("double"))
    """Longitude"""
    ## Latitude
    gps_lat: StdVectorListDesc = field(default=StdVectorListDesc("double"))
    """Latitude"""
    ## Altitude
    gps_alt: StdVectorListDesc = field(default=StdVectorListDesc("double"))
    """Altitude"""
    ## GPS temperature
    gps_temp: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """GPS temperature"""

    ## ?? What is it? Some kind of the adc trace offset?
    ioff: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """?? What is it? Some kind of the adc trace offset?"""

    ## Voltage traces for channels 1,2,3,4 in muV
    trace_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Voltage traces for channels 1,2,3,4 in muV"""


@dataclass
## The class for storing voltage traces and associated values for each event
class TVoltage(MotherEventTree):
    """The class for storing voltage traces and associated values at antenna feed point for each event"""

    _type: str = "voltage"

    _tree_name: str = "tvoltage"

    ## Common for the whole event
    ## First detector unit that triggered in the event
    first_du: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """First detector unit that triggered in the event"""
    ## Unix time corresponding to the GPS seconds of the trigger
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Unix time corresponding to the GPS seconds of the trigger"""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """GPS nanoseconds corresponding to the trigger of the first triggered station"""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detector units in the event - basically the antennas count"""

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Detector unit (antenna) ID"""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Unix time of the trigger for this DU"""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Nanoseconds of the trigger for this DU"""
    ## Same as event_type, but event_type could consist of different triggered DUs
    trigger_flag: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Same as event_type, but event_type could consist of different triggered DUs"""
    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger position in the trace, in samples"""    
    
    ## Acceleration of the antenna in (x,y,z) in m/s2
    du_acceleration: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Acceleration of the antenna in (x,y,z) in m/s2"""
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    trigger_rate: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger rate - the number of triggers recorded in the second preceding the event"""

    ## Voltage traces for antenna arms (x,y,z)
    trace: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Voltage traces for antenna arms (x,y,z)"""
    # _trace: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<Float32_t>>"))

    ## Peak2peak amplitude (muV)
    p2p: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Peak2peak amplitude (muV)"""
    ## (Computed) peak time
    time_max: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """(Computed) peak time"""


@dataclass
## The class for storing Efield traces and associated values for each event
class TEfield(MotherEventTree):
    """The class for storing Efield traces and associated values for each event"""

    _type: str = "efield"

    _tree_name: str = "tefield"

    ## Common for the whole event
    ## Unix time corresponding to the GPS seconds of the trigger
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Unix time corresponding to the GPS seconds of the trigger"""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """GPS nanoseconds corresponding to the trigger of the first triggered station"""
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    event_type: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detector units in the event - basically the antennas count"""

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Detector unit (antenna) ID"""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Unix time of the trigger for this DU"""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Nanoseconds of the trigger for this DU"""

    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger position in the trace, in samples"""


    ## Efield traces for antenna arms (x,y,z)
    trace: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Efield traces for antenna arms (x,y,z)"""
    ## FFT magnitude for antenna arms (x,y,z)
    fft_mag: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """FFT magnitude for antenna arms (x,y,z)"""
    ## FFT phase for antenna arms (x,y,z)
    fft_phase: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """FFT phase for antenna arms (x,y,z)"""

    ## Peak-to-peak amplitudes for X, Y, Z (muV/m)
    p2p: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Peak-to-peak amplitudes for X, Y, Z (muV/m)"""
    ## Efield polarisation info
    pol: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Efield polarisation info"""
    ## (Computed) peak time
    time_max: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """(Computed) peak time"""


@dataclass
## The class for storing reconstructed shower data common for each event
class TShower(MotherEventTree):
    """The class for storing shower data common for each event"""

    _type: str = "shower"

    _tree_name: str = "tshower"

    ## Shower primary type
    primary_type: StdStringDesc = field(default=StdStringDesc(""))
    """Shower primary type"""
    ## Energy from e+- (ie related to radio emission) (GeV)
    energy_em: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Energy from e+- (ie related to radio emission) (GeV)"""
    ## Total energy of the primary (including muons, neutrinos, ...) (GeV)
    energy_primary: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Total energy of the primary (including muons, neutrinos, ...) (GeV)"""
    ## Shower azimuth  (coordinates system = NWU + origin = core, "pointing to")
    azimuth: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Shower azimuth  (coordinates system = NWU + origin = core, "pointing to")"""
    ## Shower zenith  (coordinates system = NWU + origin = core, , "pointing to")
    zenith: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Shower zenith  (coordinates system = NWU + origin = core, , "pointing to")"""
    ## Direction vector (u_x, u_y, u_z)  of shower in GRAND detector ref
    direction: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Direction vector (u_x, u_y, u_z)  of shower in GRAND detector ref"""
    ## Shower core position in GRAND detector ref (if it is an upgoing shower, there is no core position)
    shower_core_pos: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Shower core position in GRAND detector ref (if it is an upgoing shower, there is no core position)"""
    ## Atmospheric model name
    atmos_model: StdStringDesc = field(default=StdStringDesc(""))
    """Atmospheric model name"""
    ## Atmospheric model parameters
    atmos_model_param: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Atmospheric model parameters"""
    ## Magnetic field parameters: Inclination, Declination, modulus
    magnetic_field: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Magnetic field parameters: Inclination, Declination, modulus"""
    ## Ground Altitude at core position (m asl)
    core_alt: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Ground Altitude at core position (m asl)"""
    ## Shower Xmax depth  (g/cm2 along the shower axis)
    xmax_grams: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Shower Xmax depth  (g/cm2 along the shower axis)"""
    ## Shower Xmax position in GRAND detector ref
    xmax_pos: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Shower Xmax position in GRAND detector ref"""
    ## Shower Xmax position in shower coordinates
    xmax_pos_shc: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Shower Xmax position in shower coordinates"""
    ## Unix time when the shower was at the core position (seconds after epoch)
    core_time_s: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float64))
    """Unix time when the shower was at the core position (seconds after epoch)"""
    ## Unix time when the shower was at the core position (seconds after epoch)
    core_time_ns: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float64))
    """Unix time when the shower was at the core position (seconds after epoch)"""


@dataclass
## The class for storing Efield sim-only data common for a whole run
class TRunEfieldSim(MotherRunTree):
    """The class for storing Efield sim-only data common for a whole run"""

    _type: str = "runefieldsim"

    _tree_name: str = "trunefieldsim"

    ## Name of the atmospheric index of refraction model
    refractivity_model: StdStringDesc = field(default=StdStringDesc())
    """Name of the atmospheric index of refraction model"""
    refractivity_model_parameters: StdVectorListDesc = field(default=StdVectorListDesc("double"))
    ## Starting time of antenna data collection time window (because it can be a shorter trace then voltage trace, and thus these parameters can be different)
    t_pre: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Starting time of antenna data collection time window (because it can be a shorter trace then voltage trace, and thus these parameters can be different)"""
    ## Finishing time of antenna data collection time window
    t_post: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Finishing time of antenna data collection time window"""

    ## Simulator name (aires/corsika, etc.)
    sim_name: StdStringDesc = field(default=StdStringDesc())
    """Simulator name (aires/corsika, etc.)"""
    ## Simulator version string
    sim_version: StdStringDesc = field(default=StdStringDesc())
    """Simulator version string"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


@dataclass
## The class for storing shower sim-only data common for a whole run
class TRunShowerSim(MotherRunTree):
    """Run-level info associated with simulated showers"""

    _type: str = "runshowersim"

    _tree_name: str = "trunshowersim"

    ## relative thinning energy
    rel_thin: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """relative thinning energy"""
    # maximum_weight (weight factor)
    maximum_weight: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """the maximum weight, computed in zhaires as PrimaryEnergy*RelativeThinning*WeightFactor/14.0 (see aires manual section 3.3.6 and 2.3.2) to make it mean the same as Corsika Wmax"""

    hadronic_thinning: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """the ratio of energy at wich thining starts in hadrons and electromagnetic particles"""
    hadronic_thinning_weight: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """the ratio of electromagnetic to hadronic maximum weights"""

    ## low energy cut for electrons (GeV)
    lowe_cut_e: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for electrons (GeV)"""
    ## low energy cut for gammas (GeV)
    lowe_cut_gamma: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for gammas (GeV)"""
    ## low energy cut for muons (GeV)
    lowe_cut_mu: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for muons (GeV)"""
    ## low energy cut for mesons (GeV)
    lowe_cut_meson: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for mesons (GeV)"""
    ## low energy cut for nuceleons (GeV)
    lowe_cut_nucleon: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for nuceleons (GeV)"""
    ## Site for wich the smulation was done
    site: StdStringDesc = field(default=StdStringDesc())
    """Site for wich the smulation was done"""
    ## Simulator name (aires/corsika, etc.)
    sim_name: StdStringDesc = field(default=StdStringDesc())
    """Simulator name (aires/corsika, etc.)"""
    ## Simulator version string
    sim_version: StdStringDesc = field(default=StdStringDesc())
    """Simulator version string"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


@dataclass
## The class for storing a shower sim-only data for each event
class TShowerSim(MotherEventTree):
    """Event-level info associated with simulated showers"""

    _type: str = "showersim"

    _tree_name: str = "tshowersim"

    ## File name in the simulator
    input_name: StdStringDesc = field(default=StdStringDesc())
    """File name in the simulator"""
    ## The date for which we simulate the event (epoch)
    event_date: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """The date for which we simulate the event (epoch)"""
    ## Random seed
    rnd_seed: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float64))
    """Random seed"""
    ## Primary energy (GeV)
    # primary_energy: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Primary particle type
    # primary_type: StdVectorListDesc = field(default=StdVectorListDesc("string"))
    ## Primary injection point in Shower Coordinates
    primary_inj_point_shc: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Primary injection point in Shower Coordinates"""
    ## Primary injection altitude in Shower Coordinates
    primary_inj_alt_shc: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Primary injection altitude in Shower Coordinates"""
    ## Primary injection direction in Shower Coordinates
    primary_inj_dir_shc: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Primary injection direction in Shower Coordinates"""

    ## Table of air density [g/cm3] and vertical depth [g/cm2] versus altitude [m]
    atmos_altitude: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Table of air density [g/cm3] and vertical depth [g/cm2] versus altitude [m]"""
    atmos_density: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    atmos_depth: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))

    ## High energy hadronic model (and version) used
    hadronic_model: StdStringDesc = field(default=StdStringDesc())
    """High energy hadronic model (and version) used"""
    ## Energy model (and version) used
    low_energy_model: StdStringDesc = field(default=StdStringDesc())
    """Energy model (and version) used"""
    ## Time it took for the sim
    cpu_time: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Time it took for the sim"""

    ## Slant depth of the observing levels for longitudinal development tables
    long_depth: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Slant depth of the observing levels for longitudinal development tables"""
    long_pd_depth: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Number of electrons
    long_pd_eminus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of electrons"""
    ## Number of positrons
    long_pd_eplus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of positrons"""
    ## Number of muons-
    long_pd_muminus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of muons-"""
    ## Number of muons+
    long_pd_muplus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of muons+"""
    ## Number of gammas
    long_pd_gamma: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of gammas"""
    ## Number of pions, kaons, etc.
    long_pd_hadron: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of pions, kaons, etc."""
    ## Energy in low energy gammas
    long_gamma_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in low energy gammas"""
    ## Energy in low energy e+/e-
    long_e_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in low energy e+/e-"""
    ## Energy deposited by e+/e-
    long_e_edep: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy deposited by e+/e-"""
    ## Energy in low energy mu+/mu-
    long_mu_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in low energy mu+/mu-"""
    ## Energy deposited by mu+/mu-
    long_mu_edep: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy deposited by mu+/mu-"""
    ## Energy in low energy hadrons
    long_hadron_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in low energy hadrons"""
    ## Energy deposited by hadrons
    long_hadron_edep: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy deposited by hadrons"""
    ## Energy in created neutrinos
    long_neutrino: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in created neutrinos"""
    ## Core positions tested for that shower to generate the event (effective area study)
    tested_core_positions: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Core positions tested for that shower to generate the event (effective area study)"""

    event_weight: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """statistical weight given to the event"""
    tested_cores: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """tested core positions"""


## General info on the noise generation
@dataclass
class TRunNoise(MotherRunTree):
    """General info on the noise generation"""

    _type: str = "runnoise"

    _tree_name: str = "trunnoise"

    ## Info to retrieve the map of galactic noise
    gal_noise_map: StdStringDesc = field(default=StdStringDesc())
    """Info to retrieve the map of galactic noise"""
    ## LST time when we generate the noise
    gal_noise_LST: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """LST time when we generate the noise"""
    ## Noise std dev for each arm of each antenna
    gal_noise_sigma: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Noise std dev for each arm of each antenna"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


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

        if sim2root_structure:
            self.init_sim2root_structure()
        else:
            logger.warning("Sorry, non sim2root directories not supported yet")

        # Create chains and set them as attributes
        # self.create_chains()

    def __getattr__(self, name):
        """For non-existing tree files or tree parameters, return None instead of rising an exception"""
        trees_to_check = ["trun", "trunvoltage", "trawvoltage", "tadc", "tvoltage", "tefield", "tshower", "trunefieldsim", "trunshowersim", "tshowersim", "trunnoise"]
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

        for filename in self.file_list:
            file_handle_list.append(DataFile(filename))

        return file_handle_list

    # Init the instance with sim2root structure files
    def init_sim2root_structure(self):

        # Loop through groups of files with tree types expected in the directory
        for flistname in ["ftruns", "ftrunshowersims", "ftrunefieldsims", "ftefields", "ftshowers", "ftshowersims", "ftvoltages", "ftadcs", "ftrawvoltages", "ftrunnoises"]:
            # Assign the list of files with specific tree type to the class instance
            setattr(self, flistname, {int(Path(el.filename).name.split("_")[2][1:]): el for el in self.file_handle_list if Path(el.filename).name.startswith(flistname[2:-1]+"_")})
            max_level = -1
            for (l, f) in getattr(self, flistname).items():
                # Assign the file with the tree with the specific analysis level to the class instance
                setattr(self, f"{flistname[:-1]}_l{l}", f)
                # Assign the tree with the specific analysis level to the class instance
                setattr(self, f"{flistname[1:-1]}_l{l}", getattr(f, f"{flistname[1:-1]}_l{l}"))
                if (l>max_level and self.analysis_level==-1) or l==self.analysis_level:
                    max_level = l
                    # Assign the file with the highest or requested analysis level as default to the class instance
                    # ToDo: This may assign all files until it goes to the max level. Probably could be avoided
                    setattr(self, f"{flistname[:-1]}", f)
                    # Assign the tree with the highest or requested analysis level as default to the class instance
                    setattr(self, f"{flistname[1:-1]}", getattr(f, f"{flistname[1:-1]}"))

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

                    tree_class = getattr(thismodule, el["type"])
                    setattr(self, tree_class.get_default_tree_name(), chains_dict[max_anal_chain_name])

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

    ## Holds all the trees in the file, by tree name
    dict_of_trees = {}
    """Holds all the trees in the file, by tree name"""
    ## Holds the list of trees in the file, but just with maximal level
    list_of_trees = []
    """Holds the list of trees in the file, but just with maximal level"""
    ## Holds dict of tree types, each containing a dict of tree names with tree meta-data as values
    tree_types = defaultdict(dict)
    """Holds dict of tree types, each containing a dict of tree names with tree meta-data as values"""

    def __init__(self, filename):
        """filename can be either a string or a ROOT.TFile"""

        # Need to init here, so that different instances do not share the same data
        self.dict_of_trees = {}
        self.list_of_trees = []
        self.tree_types = defaultdict(dict)

        # If a string given, open the file
        if type(filename) is str:
            f = ROOT.TFile(filename)
            self.f = f
            self.filename = filename
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

            # Add the tree to a dict for this tree class
            self.tree_types[tree_info["type"]][tree_info["name"]] = tree_info

            self.dict_of_trees[tree_info["name"]] = t

        # Select the highest analysis level trees for each class and store these trees as main attributes
        # Loop through tree types
        self.tree_instances = []
        # ToDo: make sure that this is for the instance, not the class
        self.max_tree_instance = None
        for key in self.tree_types:
            max_analysis_level = -1
            # Loop through trees in the current tree type
            for key1 in self.tree_types[key].keys():
                el = self.tree_types[key][key1]
                tree_class = getattr(thismodule, el["type"])
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

class DataFileChain:
    """Class holding a number of DataFiles with the same TTree type, TChaining the trees together"""

    self.list_of_files = []
    """The list of DataFiles in the chain"""

    self.chain = None
    """The main TChain"""

    def __init__(self, files, tree_type):

        # Create the ROOT TChain and the list of files
        self.chain = ROOT.TChain(tree_type)
        for f in files:
            self.list_of_files.append(f)
            self.chain.Add(f.filename)


