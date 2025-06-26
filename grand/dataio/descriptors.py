# Created by Lech Wiktor Piotrowski at 13/03/2025
import array
import sys
import ROOT
import os

import numpy as np

# Load the C++ macros for vector filling from numpy arrays
ROOT.gROOT.LoadMacro(os.path.dirname(os.path.realpath(__file__))+"/vector_filling.C")

# Conversion between numpy dtype and array.array typecodes
numpy_to_array_typecodes = {np.dtype('int8'): 'b', np.dtype('int16'): 'h', np.dtype('int32'): 'i', np.dtype('int64'): 'q', np.dtype('uint8'): 'B', np.dtype('uint16'): 'H', np.dtype('uint32'): 'I', np.dtype('uint64'): 'Q', np.dtype('float32'): 'f', np.dtype('float64'): 'd', np.dtype('complex64'): 'F', np.dtype('complex128'): 'D', np.dtype('int16'): 'h'}

# Conversion between C++ type and array.array typecodes
cpp_to_array_typecodes = {'char': 'b', 'short': 'h', 'int': 'i', 'long long': 'q', 'unsigned char': 'B', 'unsigned short': 'H', 'unsigned int': 'I', 'unsigned long long': 'Q', 'float': 'f', 'double': 'd', 'string': 'u'}

# Conversion between C++ type and numpy typecodes
cpp_to_numpy_typecodes = {'char': np.dtype('int8'), 'short': np.dtype('int16'), 'int': np.dtype('int32'), 'long long': np.dtype('int64'), 'unsigned char': np.dtype('uint8'), 'unsigned short': np.dtype('uint16'), 'unsigned int': np.dtype('uint32'), 'unsigned long long': np.dtype('uint64'), 'float': np.dtype('float32'), 'double': np.dtype('float64'), 'string': np.dtype('U')}


high_root_version = ROOT.gROOT.GetVersionInt()>=63600
higher_root_version = ROOT.gROOT.GetVersionInt()>=63004

# This import changes in Python 3.10
if sys.version_info.major >= 3 and sys.version_info.minor < 10:
    from collections import MutableSequence
else:
    from collections.abc import MutableSequence


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
                    # Conversion of char arrays with numpy gives a bool array :/ Need to go with the looping
                    if "char" in self.basic_vec_type:
                        raise
                    return np.array(self._vector[index]).tolist()
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
        # To avoid memory leak when getting a numpy array
        if isinstance(value, np.ndarray):
            fill_stdvectorlist_with_array(self, value)
            return self

        try:
            if higher_root_version:
                try:
                    if isinstance(value, StdVectorList):
                        if high_root_version:
                            self._vector.assign(value._vector)
                        else:
                            self._vector += value._vector
                    else:
                        if high_root_version:
                            self._vector.assign(value)
                        else:
                            self._vector += value
                    return self
                except:
                    try:
                        if isinstance(value, StdVectorList):
                            if "char" in value.basic_vec_type.split()[-1]:
                                tmp_array = np.array(chars_to_uint8_array(value._vector)).astype(cpp_to_numpy_typecodes[self.basic_vec_type])
                                fill_stdvectorlist_with_array(self, tmp_array)
                                return self
                                # if high_root_version:
                                #     tmp_array = np.array(chars_to_uint8_array(value._vector)).astype(cpp_to_numpy_typecodes[self.basic_vec_type])
                                #     fill_stdvectorlist_with_array(self, tmp_array)
                                #     self._vector.assign()
                                # else:
                                #     self._vector += np.array(chars_to_uint8_array(value._vector)).astype(cpp_to_numpy_typecodes[self.basic_vec_type])
                            else:
                                try:
                                    self._vector.assign(value._vector)
                                except:
                                    self._vector += value._vector
                        else:
                            tmp_array = np.ascontiguousarray(value).astype(cpp_to_numpy_typecodes[self.basic_vec_type])
                            fill_stdvectorlist_with_array(self, tmp_array)

                            # self._vector.assign(np.ascontiguousarray(value).astype(cpp_to_numpy_typecodes[self.basic_vec_type]))
                        return self
                    except Exception as e:
                        # Basically only for ROOT <6.36 for 3D lists that can't be converted to arrays (traces of non-homogenous lenght)
                        if self.ndim == 1: value = array.array(cpp_to_array_typecodes[self.basic_vec_type], value)
                        if self.ndim == 2: value = [array.array(cpp_to_array_typecodes[self.basic_vec_type], el) for el in value]
                        if self.ndim == 3: value = [[array.array(cpp_to_array_typecodes[self.basic_vec_type], el1) for el1 in el] for el in value]

                        self._vector += value

                        return self
            else:
                # For ROOT <6.30.06
                if isinstance(value, np.ndarray):
                    # Do not set empty values
                    if value.size==0: return
                    # Sometimes, for example, int is given in place of unsigned int, and C++ fuction does not convert it, so python conversion is needed
                    value = value.astype(cpp_to_numpy_typecodes[self.basic_vec_type])
                    if self.ndim == 1: ROOT.fill_vec_1D[self.basic_vec_type](np.ascontiguousarray(value), np.array(value.shape).astype(np.int32), self._vector)
                    if self.ndim == 2: ROOT.fill_vec_2D[self.basic_vec_type](np.ascontiguousarray(value), np.array(value.shape).astype(np.int32), self._vector)
                    if self.ndim == 3: ROOT.fill_vec_3D[self.basic_vec_type](np.ascontiguousarray(value), np.array(value.shape).astype(np.int32), self._vector)
                else:
                    if (isinstance(value, list) and self.basic_vec_type.split()[-1] == "float"):
                        # Do not set empty values
                        if not value: return
                        if self.ndim == 1: value = array.array(cpp_to_array_typecodes[self.basic_vec_type], value)
                        if self.ndim == 2: value = [array.array(cpp_to_array_typecodes[self.basic_vec_type], el) for el in value]
                        if self.ndim == 3: value = [[array.array(cpp_to_array_typecodes[self.basic_vec_type], el1) for el1 in el] for el in value]
                    elif not isinstance(value, StdVectorList):
                        value = list(value)

                    # The list needs to have simple Python types - ROOT.vector does not accept numpy types
                    try:
                        if isinstance(value, StdVectorList):
                            # Do not set empty values
                            if not value: return
                            # ToDo: Maybe faster than +=, but... to be checked
                            if high_root_version:
                                try:
                                    # That will not work if we have different types of value and self._vector
                                    self._vector += value._vector
                                except:
                                    # For different types, convert the type with numpy, but it won't simply work for chars

                                    # For chars is ugly, using a special function
                                    if "char" in value.basic_vec_type.split()[-1]:
                                        self._vector.assign(np.array(chars_to_uint8_array(value._vector)).astype(cpp_to_numpy_typecodes[self.basic_vec_type]))
                                    else:
                                        self._vector.assign(np.array(value._vector).astype(cpp_to_numpy_typecodes[self.basic_vec_type]))

                            else:
                                self._vector += value._vector
                                # self._vector.assign(value._vector)
                        else:
                            self._vector += value
                    except TypeError:
                        if high_root_version:
                            # Slow conversion to simple types. No better idea for now
                            if self.basic_vec_type.split()[-1] in ["int", "long", "short", "char", "float"]:
                                if self.ndim == 1:
                                    value = list(value)
                                if self.ndim == 2:
                                    value = [list(el) for el in value]
                                if self.ndim == 3: value = [[list(el1) for el1 in el] for el in value]
                                self._vector += value
                        else:
                            if "char" in value.basic_vec_type.split()[-1]:
                                self._vector += chars_to_uint8_array(value._vector)
                            elif self.basic_vec_type.split()[-1] in ["int", "long", "short", "char", "float"]:
                                if self.ndim == 1: value = array.array(cpp_to_array_typecodes[self.basic_vec_type], value)
                                if self.ndim == 2: value = [array.array(cpp_to_array_typecodes[self.basic_vec_type], el) for el in value]
                                if self.ndim == 3: value = [[array.array(cpp_to_array_typecodes[self.basic_vec_type], el1) for el1 in el] for el in value]
                                # self._vector.assign(value)
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
            # Do not set empty values
            if not value: return
        inst = getattr(obj, self.attrname)
        vector = inst._vector
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray) or isinstance(value, StdVectorList):
            # Clear the vector before setting
            vector.clear()
            inst += value
        # A vector of vectors was given
        elif isinstance(value, ROOT.vector(inst.vec_type)):
            inst._vector = value
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

from collections.abc import Iterable

def chars_to_uint8_array(nested_chars):
    # 1) Remember the original shape
    shape = np.shape(nested_chars)

    # 2) Flatten arbitrarily-nested lists of single‐char strings
    def _flatten(xs):
        for x in xs:
            if (isinstance(x, Iterable) or "std.vector" in str(type(x))) and not isinstance(x, (str, bytes)):
                yield from _flatten(x)
            else:
                yield x

    # 3) Build one bytes object via Latin-1 encoding
    flat_str = ''.join(_flatten(nested_chars))
    buf = flat_str.encode('latin-1')

    # 4) View that buffer as uint8 and restore shape
    return np.frombuffer(buf, dtype=np.uint8).reshape(shape)

def convert_deepest_lists_to_arrays(data):
    """
    Recursively traverse a nested list structure and convert only
    the deepest lists (which contain non-lists) into numpy arrays.
    """
    if isinstance(data, list):
        # If elements are not lists themselves, this is the deepest level
        if not any(isinstance(el, list) for el in data):
            return np.array(data)
        else:
            return [convert_deepest_lists_to_arrays(el) for el in data]
    else:
        return data  # Base case: not a list

def split_2d_arrays_to_rows(data):
    """
    Traverse `data`, and for every 2D numpy array found,
    replace it with a list of its 1D row arrays.
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            # turn M×N array into [array(shape=(N,)), …]
            return [row for row in data]
        else:
            # leave arrays of other dims untouched
            return data

    elif isinstance(data, list):
        # recurse into each element of the list
        return [split_2d_arrays_to_rows(el) for el in data]

    else:
        # any non‐list, non‐ndarray is passed through
        return data

# Fills the StdVectorList's vector with numpy array using C++ functions to avoid a memory leak
def fill_stdvectorlist_with_array(target, value):
    if isinstance(value, np.ndarray):
        # Do not set empty values
        if value.size == 0: return
        # Sometimes, for example, int is given in place of unsigned int, and C++ fuction does not convert it, so python conversion is needed
        value = value.astype(cpp_to_numpy_typecodes[target.basic_vec_type])
        if target.ndim == 1: ROOT.fill_vec_1D[target.basic_vec_type](np.ascontiguousarray(value), np.array(value.shape).astype(np.int32), target._vector)
        if target.ndim == 2: ROOT.fill_vec_2D[target.basic_vec_type](np.ascontiguousarray(value), np.array(value.shape).astype(np.int32), target._vector)
        if target.ndim == 3: ROOT.fill_vec_3D[target.basic_vec_type](np.ascontiguousarray(value), np.array(value.shape).astype(np.int32), target._vector)


## Exception raised when an already existing event/run is added to a tree
class NotUniqueEvent(Exception):
    """Exception raised when an already existing event/run is added to a tree"""

    pass
