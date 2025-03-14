# Created by Lech Wiktor Piotrowski at 14/03/2025
import datetime
import glob
import os
from dataclasses import dataclass, field
from logging import getLogger
import ROOT

import numpy as np

from grand.dataio import StdVectorList, StdVectorListDesc, StdString

logger = getLogger(__name__)

## A list of generated Trees
grand_tree_list = []
"""Internal list of generated Trees"""

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

    ## Is the tree read from TChain
    is_tchain: bool = False
    """Is the tree read from TChain"""


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
        "__setattr__",
        "is_tchain"
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
        # If the filename string is given, check if chain, if not open/create the ROOT file with this name
        elif isinstance(f, str):
            # Check if a chain - filename string resolves to a list longer than 1 (due to wildcards)
            flist = glob.glob(f)
            if len(flist) > 1:
                f = flist
            # Otherwise, it was a single file
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
        else:
            raise ValueError(f"Unsupported filename {f}. Can't open/create a file with a tree.")

        # If a list is given, it's a Chain
        if isinstance(f, list):
            self.is_tchain = True
            # Create the TChain
            self._tree = ROOT.TChain(self._tree_name, self._tree_name)
            # Assign files to the chain
            for el in f:
                self._tree.Add(el)


    ## Init/readout the tree from a file
    def _set_tree(self, t):
        """Init/readout the tree from a file"""
        # If the ROOT TTree is given, just use it
        if isinstance(t, ROOT.TTree) or isinstance(t, ROOT.TChain):
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

    def get_current_file(self):
        """Get's the current TFile the TTree is in"""
        return self._tree.GetCurrentFile()

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
            raise ValueError(f"Unsupported type {type(value)}. Can't create a branch {branch_name}.")

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

        # ToDo: this should create some lists of values for TChains

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
