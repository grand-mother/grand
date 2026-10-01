# Created by Lech Wiktor Piotrowski at 14/03/2025
import datetime
import glob
import os
from dataclasses import dataclass, field
from logging import getLogger
import weakref
import ROOT

import numpy as np

from grand.basis import validate as _validate

from grand.dataio import StdVectorList, StdVectorListDesc, StdString
from grand.dataio import file_lock as _file_lock

logger = getLogger(__name__)

## A list of generated Trees
grand_tree_list = []
"""Internal list of generated Trees"""

## ROOT files that the trees opened themselves from a file name, keyed by the TFile address
_files_opened_by_trees = {}
"""ROOT files that the trees opened themselves from a file name, keyed by the TFile address.

``stop_using()`` closes a file listed here once no tree in ``grand_tree_list`` uses it.
A ``ROOT.TFile`` handed in by the caller is never listed, so the caller keeps control of it.
This is needed because PyROOT stops owning a TFile once ``TTree.SetDirectory(file)`` is called
with it, so dropping the last Python reference does not close it (GitHub issue #71)."""


def _register_opened_file(f):
    """Record a TFile that a tree opened itself, so that ``stop_using()`` may close it"""
    _files_opened_by_trees[ROOT.addressof(f)] = f


def _forget_opened_file(f):
    """Remove a TFile from the record of files the trees opened themselves"""
    if f is not None:
        _files_opened_by_trees.pop(ROOT.addressof(f), None)
        # Closed: another process may write it now (issue #281)
        _file_lock.release(f.GetName())

@dataclass
class DataTree:
    """
    Mother class for GRAND Tree data classes

    Every instance is kept in ``grand_tree_list`` until ``stop_using()`` is
    called, so a loop over many files must release each tree, either with
    ``tree.stop_using()`` or by using the tree as a context manager::

        for path in paths:
            with TADC(path) as tadc:
                ...
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
        r"""Sets an attribute, recording the change in the modification history.

        Parameters
        ----------
        key : str
            Attribute name.
        value : object
            New value.
        """
        if not hasattr(self, "_attributes_and_properties"):
            super().__setattr__("_attributes_and_properties", set([el1 for el in type(self).__mro__[:-1] for el1 in list(el.__dict__.keys()) + list(el.__annotations__.keys())]))
        # If the attribute not in the list of class's attributes and properties, don't add it
        if key not in self._attributes_and_properties:
            raise AttributeError(f"{key} attribute for class {type(self)} doesn't exist.")
        super().__setattr__(key, value)

    @property
    def tree(self):
        """The ROOT TTree in which the variables' values are stored

        Returns
        -------
        ROOT.TTree
            The underlying ROOT tree.
        """
        return self._tree

    @property
    def type(self):
        """The type of the tree

        Returns
        -------
        str
            The tree type, as stored in its metadata.
        """
        return self._type

    @type.setter
    def type(self, val: str) -> None:
        # The meta field does not exist, add it
        r"""Sets the tree type.

        Parameters
        ----------
        val : object
            The new value.
        """
        if not (el:=self._tree.GetUserInfo().FindObject("type")):
            self._tree.GetUserInfo().Add(ROOT.TNamed("type", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)
        # Update the property
        self._type = val

    @property
    def file(self):
        """The ROOT TFile in which the tree is stored

        Returns
        -------
        ROOT.TFile
            The file this tree lives in.
        """
        return self._file

    @file.setter
    def file(self, val: ROOT.TFile) -> None:
        r"""Sets the file this tree belongs to.

        Parameters
        ----------
        val : object
            The new value.
        """
        self._set_file(val)

    @property
    def tree_name(self):
        """The name of the TTree

        Returns
        -------
        str
            Name of the tree in the file.

        Parameters
        ----------
        val : str
            Name to give the tree.
        """
        return self._tree_name

    @tree_name.setter
    def tree_name(self, val):
        """Set the tree name


        Parameters
        ----------
        val : str
            Name to give the tree.
        """
        # ToDo: enforce the name to start with the type!
        self._tree_name = val
        self._tree.SetName(val)
        self._tree.SetTitle(val)

    @property
    def file_name(self):
        """The file in which the TTree is stored

        Returns
        -------
        str
            Path of the file this tree lives in.
        """
        return self._file_name

    @property
    def comment(self):
        """Comment - if needed, added by user

        Returns
        -------
        str
            Free-text comment stored with the tree.
        """
        return self._comment

    @comment.setter
    def comment(self, val: str) -> None:
        # The meta field does not exist, add it
        r"""Sets the free-text comment stored with the tree.

        Parameters
        ----------
        val : object
            The new value.
        """
        if not (el:=self._tree.GetUserInfo().FindObject("comment")):
            self._tree.GetUserInfo().Add(ROOT.TNamed("comment", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)

        # Update the property
        self._comment = val

    @property
    def creation_datetime(self):
        r"""Returns the time the tree was created.

        Returns
        -------
        datetime
            Creation time, as recorded in the file.

        Parameters
        ----------
        val : datetime or int
            Creation time to record.
        """
        return self._creation_datetime

    @creation_datetime.setter
    def creation_datetime(self, val: datetime.datetime) -> None:
        # If datetime was given, convert it to int
        r"""Returns the time the tree was created.

        Parameters
        ----------
        val : datetime or int
            Creation time to record.
        """
        if type(val) == datetime.datetime:
            # Keep the datetime for the getter; ROOT stores the integer.  The
            # two lines were the other way round, so reading back after
            # setting a datetime gave an int (found while triaging #136).
            val_dt = val
            val = int(val.timestamp())
        elif type(val) == int:
            val_dt = datetime.datetime.fromtimestamp(val)
        else:
            raise ValueError(f"Unsupported type {type(val)} for creation_datetime!")

        # The meta field does not exist, add it
        if not (el := self._tree.GetUserInfo().FindObject("creation_datetime")):
            self._tree.GetUserInfo().Add(ROOT.TParameter(int)("creation_datetime", val))
        # The meta field exists, change the value
        else:
            el.SetVal(val)

        self._creation_datetime = val_dt

    @property
    def modification_history(self):
        """Modification_history - if needed, added by user

        Returns
        -------
        str
            Record of the modifications made to this tree.
        """
        return self._modification_history

    @modification_history.setter
    def modification_history(self, val: str) -> None:
        # The meta field does not exist, add it
        r"""Sets the record of modifications.

        Parameters
        ----------
        val : object
            The new value.
        """
        if not (el:=self._tree.GetUserInfo().FindObject("modification_history")):
            self._tree.GetUserInfo().Add(ROOT.TNamed("modification_history", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)

        # Update the property
        self._modification_history = val

    @property
    def source_datetime(self):
        """Unix creation datetime of the source tree; 0 s means no source

        Returns
        -------
        datetime
            Timestamp of the data this tree derives from.
        """
        # Convert from ROOT's TDatime into Python's datetime object
        # return datetime.datetime.fromtimestamp(self._tree.GetUserInfo().At(3).GetVal())
        return self._source_datetime

    @source_datetime.setter
    def source_datetime(self, val: datetime.datetime) -> None:
        # Remove the existing datetime
        r"""Sets the timestamp of the data this tree derives from.

        Parameters
        ----------
        val : object
            The new value.
        """
        self._tree.GetUserInfo().Remove(self._tree.GetUserInfo().FindObject("source_datetime"))

        # If datetime was given
        if type(val) == datetime.datetime:
            # Keep the datetime for the getter; ROOT stores the integer.  The
            # two lines were the other way round, so reading back after
            # setting a datetime gave an int (found while triaging #136).
            val_dt = val
            val = int(val.timestamp())
        # If timestamp was given - this happens when initialising with self.assign_metadata()
        elif type(val) == int:
            val_dt = datetime.datetime.fromtimestamp(val)
        else:
            raise ValueError(f"Unsupported type {type(val)} for source_datetime!")

        # The meta field does not exist, add it
        if not (el:=self._tree.GetUserInfo().FindObject("source_datetime")):
            self._tree.GetUserInfo().Add(ROOT.TParameter(int)("source_datetime", val))
        # The meta field exists, change the value
        else:
            el.SetVal(val)

        self._source_datetime = val_dt

    @property
    def modification_software(self):
        """The tool used to generate this tree's values from another tree

        Returns
        -------
        str
            Name of the software that last modified the tree.
        """
        return self._modification_software

    @modification_software.setter
    def modification_software(self, val: str) -> None:
        # The meta field does not exist, add it
        r"""Sets the name of the software that last modified the tree.

        Parameters
        ----------
        val : object
            The new value.
        """
        if not (el:=self._tree.GetUserInfo().FindObject("modification_software")):
            self._tree.GetUserInfo().Add(ROOT.TNamed("modification_software", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)

        self._modification_software = val

    @property
    def modification_software_version(self):
        """The tool used to generate this tree's values from another tree

        Returns
        -------
        str
            Version of that software.
        """
        return self._modification_software_version

    @modification_software_version.setter
    def modification_software_version(self, val: str) -> None:
        # The meta field does not exist, add it
        r"""Sets the version of the software that last modified the tree.

        Parameters
        ----------
        val : object
            The new value.
        """
        if not (el:=self._tree.GetUserInfo().FindObject("modification_software_version")):
            self._tree.GetUserInfo().Add(ROOT.TNamed("modification_software_version", val))
        # The meta field exists, change the value
        else:
            el.SetTitle(val)

        self._modification_software_version = val

    @property
    def analysis_level(self):
        """The analysis level of this tree

        Returns
        -------
        int
            How far through the processing chain this data has been taken.
        """
        return self._analysis_level

    @analysis_level.setter
    def analysis_level(self, val: int) -> None:
        # The meta field does not exist, add it
        r"""Sets the analysis level.

        Parameters
        ----------
        val : int
            How far through the processing chain this data has been taken.
            It is part of how files are grouped and named, so changing it
            changes which files are read together.
        """
        if not (el:=self._tree.GetUserInfo().FindObject("analysis_level")):
            self._tree.GetUserInfo().Add(ROOT.TParameter(int)("analysis_level", val))
        # The meta field exists, change the value
        else:
            el.SetVal(val)

        self._analysis_level = val

    @classmethod
    def get_default_tree_name(cls):
        """Gets the default name of the tree of the class

        Returns
        -------
        str
            The conventional name for this tree class, such as ``"trun"``.
        """
        return cls._tree_name

    def __post_init__(self):
        r"""Completes initialisation after the dataclass fields are set.

        """
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

        self.__setattr__ = weakref.proxy(self.mod_setattr)
        # self.__setattr__ = self.mod_setattr

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
        """Set the tree's file

        Parameters
        ----------
        f : ROOT.TFile or str
            File to attach the tree to.
        """
        # If the ROOT TFile is given, just use it
        if isinstance(f, ROOT.TFile):
            self._file = f
            self._file_name = self._file.GetName()
        # If the filename string is given, check if chain, if not open/create the ROOT file with this name
        elif isinstance(f, (str, os.PathLike)):
            f = os.fspath(f)
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
                    where = type(self).__name__
                    if os.path.isfile(self._file_name):
                        # Not while another process writes it, and remember its
                        # state, to refuse a stale write later (#281)
                        try:
                            with _file_lock.opening(self._file_name, where):
                                self._file = ROOT.TFile(self._file_name, "read")
                        except OSError as error:
                            if "another process" in str(error):
                                raise
                            raise OSError(_validate.message(
                                where, "cannot open %s: it is not a ROOT file, or it is "
                                "damaged" % self._file_name)) from None
                    elif os.path.isdir(self._file_name):
                        raise IsADirectoryError(_validate.message(
                            where, "%s is a directory; give a ROOT file, or use "
                            "DataDirectory for a directory" % self._file_name))
                    # If the file does not exist, create it (this is how trees
                    # are written), but not in a directory that does not exist:
                    # that is almost always a mistyped path.
                    else:
                        parent = os.path.dirname(os.path.abspath(self._file_name))
                        if not os.path.isdir(parent):
                            raise FileNotFoundError(_validate.message(
                                where, "no such file: %s, and its directory %s does not exist "
                                "either, so it cannot be created" % (self._file_name, parent)))
                        self._file = ROOT.TFile(self._file_name, "create")
                        # Created to be written: no other process may write it meanwhile (#281)
                        _file_lock.lock_for_writing(self._file_name, where, fresh=True)
                    # Opened here, so stop_using() may close it
                    _register_opened_file(self._file)
        else:
            raise TypeError(_validate.message(
                type(self).__name__, "the file must be a file name or a ROOT.TFile, got %s"
                % type(f).__name__))

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
        """Init/readout the tree from a file

        Parameters
        ----------
        t : ROOT.TTree or str
            Tree to wrap, or its name.
        """
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
                if not self._tree:
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
        """Create the tree

        Parameters
        ----------
        tree_name : str, optional
            Name for the new tree; the class default if omitted.
        """
        if tree_name != "":
            self._tree_name = tree_name
        self._tree = ROOT.TTree(self._tree_name, self._tree_name)

        self.create_metadata()

    def fill(self):
        """Adds the current variable values as a new event to the tree"""
        pass

    def write(self, *args, close_file=True, overwrite=False, force_close_file=False, **kwargs):
        """Write the tree to the file

        Parameters
        ----------
        close_file : bool, optional
            Close the file after writing.
        overwrite : bool, optional
            Replace an existing tree of the same name.
        force_close_file : bool, optional
            Close the file even if other trees reference it.
        """
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
                # One writer at a time, with an up-to-date view of the file (#281)
                _file_lock.lock_for_writing(self._file_name, type(self).__name__)
                # File exists, but reopen the file in the update mode in case it was read only
                self._file.ReOpen("update")
            # Create a new TFile object
            else:
                creating_file = True
                # One writer at a time (#281); opened afresh, so no stale view
                if os.path.isfile(args[0]):
                    _file_lock.lock_for_writing(args[0], type(self).__name__, fresh=True)
                # Overwrite requested
                # ToDo: this does not really seem to work now
                if overwrite:
                    self._file = ROOT.TFile(args[0], "recreate")
                else:
                    # By default append
                    self._file = ROOT.TFile(args[0], "update")
                _file_lock.lock_for_writing(args[0], type(self).__name__, fresh=True)
                # Opened here, so stop_using() may close it if it is not closed below
                _register_opened_file(self._file)
            # Make the tree save itself in this file
            self._tree.SetDirectory(self._file)
            # args passed to the TTree::Write() should be the following
            args = args[1:]
        # File exists, so reopen the file in the update mode in case it was read only
        else:
            # One writer at a time, with an up-to-date view of the file (#281)
            _file_lock.lock_for_writing(self._file.GetName(), type(self).__name__)
            ret = self._file.ReOpen("update")

        # ToDo: For now, I don't know how to do that: Check if the entries in possible tree in the file do not already contain entries from the current tree

        # If the writing options are not explicitly specified, add kWriteDelete option, that deletes the old cycle after writing the new cycle in the TFile
        if len(args) < 2:
            args = ["", ROOT.TObject.kWriteDelete]
        # ToDo: make sure that the tree has different name than the trees existing in the file!
        # self._tree.Write(*args)
        self._tree.GetCurrentFile().Write(*args)

        # If TFile was created here, close it
        if (creating_file and close_file) or force_close_file:
            # Need to set 0 directory so that closing of the file does not delete the internal TTree
            self._tree.SetDirectory(ROOT.nullptr)
            self._file.Close()
            _forget_opened_file(self._file)

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
        """Read into memory the ev_no entry of the tree

        Parameters
        ----------
        ev_no : int
            Entry index to load.

        Returns
        -------
        int
            Bytes read; zero when the entry does not exist.
        """
        res = self._tree.GetEntry(ev_no)
        self.assign_branches()
        return res

    @staticmethod
    def _reset_read_cache(tree):
        r"""Clears the TTreeCache's position bookkeeping before a full pass.

        Call it before every ``Draw()`` or ``BuildIndex()`` of a whole tree.

        Parameters
        ----------
        tree : ROOT.TTree or ROOT.TChain
            The tree about to be read in full.

        Notes
        -----
        The cache remembers the entry window it last prefetched, cut at the
        number of entries the tree had then.  A tree that is being appended
        to -- the output of ``convert_efield2voltage.py``, which is drawn by
        ``fill_entry_list()`` when reopened and indexed by ``write()`` after
        every event -- grows past that window, and once it has more entries
        than the cache's learning phase (100) the next pass prints, for
        every event::

            Error in <TTreeCache::FillBuffer>: Inconsistency:
            fCurrentClusterStart=0 fEntryCurrent=176 fNextClusterStart=178 ...

        (grand-mother/grand#89).  ROOT recovers by itself and the values
        read are correct, but the message is noise.  ``ResetCache()`` forgets
        only that window: the branches the cache has learnt are kept, and no
        value read changes.  Trees without a file or a cache are left alone.
        """
        f = tree.GetCurrentFile()
        if not f:
            return
        cache = tree.GetReadCache(f)
        if cache:
            cache.ResetCache()

    def _kept_buffers(self, *expressions):
        r"""Saves the values held for the branches named in ``expressions``.

        ``TTree::Draw()`` reads every entry into the branch buffers this
        instance is bound to, so afterwards the fields it touched held the last
        entry's values while the others kept theirs -- whether those came from
        ``get_entry()`` or had just been set for the next ``fill()`` (#196).
        Use as ``with self._kept_buffers(varexp, selection): ...Draw(...)``.

        Parameters
        ----------
        *expressions : str
            The ROOT expressions the draw evaluates.

        Returns
        -------
        contextlib.AbstractContextManager
            Restores the saved values on exit.
        """
        import contextlib
        import re

        names = set(re.findall(r"[A-Za-z_]\w*", " ".join(e for e in expressions if e)))
        saved = []
        for name in names:
            store = self.__dict__.get("_" + name)
            if isinstance(store, np.ndarray):
                saved.append((store, store.copy()))
            elif isinstance(store, StdVectorList):
                saved.append((store._vector, type(store._vector)(store._vector)))
            elif isinstance(store, StdString):
                saved.append((store.string, ROOT.string(store.string)))

        @contextlib.contextmanager
        def keep():
            try:
                yield
            finally:
                for target, copy in saved:
                    if isinstance(target, np.ndarray):
                        target[...] = copy
                    else:
                        target.swap(copy)       # std::vector / std::string, in place
        return keep()

    def draw(self, varexp, selection, option="", nentries=ROOT.TTree.kMaxEntries, firstentry=0, delete_temp_histogram=True):
        """An interface to TTree::Draw(). Allows for drawing specific TTree columns or getting their values with get_vX().

        Parameters
        ----------
        varexp : str
            Expression to plot, in ROOT syntax.
        selection : str, optional
            Cut applied before plotting.
        option : str, optional
            ROOT draw option.
        nentries : int, optional
            Maximum entries to use.
        firstentry : int, optional
            First entry to use.
        delete_temp_histogram : bool, optional
            Remove the temporary histogram ROOT creates.

        Returns
        -------
        int
            Number of entries drawn.
        """

        self._reset_read_cache(self._tree)
        # The values of the loaded or about-to-be-filled entry are kept (#196)
        with self._kept_buffers(varexp, selection):
            count = self._tree.Draw(varexp, selection, option, nentries, firstentry)

        # Delete the temporary histogram created by draw, so it is not saved in a file
        if delete_temp_histogram:
            if tmph := ROOT.gDirectory.Get("htemp"):
                tmph.SetDirectory(0)

        return count

    def get_v1(self):
        '''Get first vector of results from draw()

        Returns
        -------
        ndarray
            First variable of the last :meth:`draw`.
        '''
        return self._tree.GetV1()

    def get_v2(self):
        '''Get second vector of results from draw()

        Returns
        -------
        ndarray
            Second variable of the last :meth:`draw`.
        '''
        return self._tree.GetV2()

    def get_v3(self):
        '''Get third vector of results from draw()

        Returns
        -------
        ndarray
            Third variable of the last :meth:`draw`.
        '''
        return self._tree.GetV3()

    def get_v4(self):
        '''Get fourth vector of results from draw()

        Returns
        -------
        ndarray
            Fourth variable of the last :meth:`draw`.
        '''
        return self._tree.GetV4()

    ## All three methods below return the number of entries
    def get_entries(self):
        """Return the number of events in the tree

        Returns
        -------
        int
            Number of entries in the tree.
        """
        return self._tree.GetEntries()

    def get_number_of_entries(self):
        """Return the number of events in the tree

        Returns
        -------
        int
            Number of entries in the tree.
        """
        return self.get_entries()

    def get_number_of_events(self):
        """Return the number of events in the tree

        Returns
        -------
        int
            Number of distinct events, which differs from the entry count when a tree holds several entries per event.
        """
        return self.get_number_of_entries()

    def add_friend(self, value, filename=""):
        # ToDo: Due to a bug discovered during DC1, disable adding of the friends for now
        r"""Attaches another tree as a ROOT friend, so its branches are readable here.

        Parameters
        ----------
        value : DataTree or str
            The tree to attach, or its name.
        filename : str, optional
            File holding it, when it is not in this one.

        Returns
        -------
        ROOT.TTree
            The friend that was attached.
        """
        return 0
        """Add a friend to the tree"""
        self._tree.AddFriend(value, filename)

    def remove_friend(self, value):
        """Remove a friend from the tree

        Parameters
        ----------
        value : DataTree or str
            Friend to detach.
        """
        self._tree.RemoveFriend(value)

    def set_tree_index(self, value):
        """Set the tree index (necessary for working with friends)

        Parameters
        ----------
        value : str
            Index expression, typically run and event number.
        """
        self._tree.SetTreeIndex(value)

    def get_current_file(self):
        """Get's the current TFile the TTree is in

        Returns
        -------
        ROOT.TFile
            The file currently being read, which differs from :attr:`file` when the tree is a chain spanning several.
        """
        return self._tree.GetCurrentFile()

    ## Create branches of the TTree based on the class fields
    def create_branches(self, set_if_exists=True):
        """Create branches of the TTree based on the class fields

        Parameters
        ----------
        set_if_exists : bool, optional
            Attach to existing branches rather than failing.
        """
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
        """Create a specific branch of a TTree computing its type from the corresponding class field

        Parameters
        ----------
        value : object
            Dataclass field to create a branch for.
        set_branches : bool, optional
            Bind the branch immediately.
        value_name : str, optional
            Branch name; the field name by default.
        """
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
                if self._tree.GetLeaf(branch_name):
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
                try:
                    self._tree.SetBranchAddress(branch_name, getattr(self, value_name).string)
                except:
                    logger.warning(f"The branch {branch_name} was not found in the source file and will not be filled.")
        elif isinstance(value, ROOT.string):
            # Create the branch
            if not set_branches:
                # self._tree.Branch(value.name[1:], getattr(self, value.name).string)
                self._tree.Branch(branch_name, getattr(self, value_name))
            # Or set its address
            else:
                # self._tree.SetBranchAddress(value.name[1:], getattr(self, value.name).string)
                try:
                    self._tree.SetBranchAddress(branch_name, getattr(self, value_name))
                except:
                    logger.warning(f"The branch {branch_name} was not found in the source file and will not be filled.")
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
        # Which code wrote the tree (issue #137).  A tool that derives this tree
        # from another may overwrite both, as extract_events.py does.
        from grand import provenance
        self.modification_software = provenance.SOFTWARE_NAME
        self.modification_software_version = provenance.current()
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
        """Get the event with run_no and evt_no

        Parameters
        ----------
        run_no : int
            Run number.
        evt_no : int
            Event number.

        Returns
        -------
        int
            Bytes read; zero when the pair matches no entry.
        """
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
        """Print out the tree scheme

        Returns
        -------
        None
            Prints a summary to standard output.
        """
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
        """Get the meta information as a dictionary

        Parameters
        ----------
        tree : DataTree, optional
            Tree to read; this one by default.

        Returns
        -------
        dict
            The metadata fields and their values.
        """

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
        The source has to have some field the same as this tree. For example EventEfieldTree and EventVoltageTree

        Parameters
        ----------
        source : DataTree
            Tree to copy entries from.
        """
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
        """Get the tree size in memory and on disk, similar to what comes from the Print()

        Returns
        -------
        int
            Size of the tree on disk, in bytes.
        """
        mem_size = self._tree.GetDirectory().GetKey(self._tree_name).GetKeylen()
        mem_size += self._tree.GetTotBytes()
        b = ROOT.TBufferFile(ROOT.TBuffer.kWrite, 10000)
        self._tree.IsA().WriteBuffer(b, self._tree)
        mem_size += b.Length()

        disk_size = self._tree.GetZipBytes()
        disk_size += self._tree.GetDirectory().GetKey(self._tree_name).GetNbytes()

        return mem_size, disk_size

    def close_file(self):
        """Close the file associated to the tree

        The file is closed even if other tree instances still read from it.
        To release a tree when you are done with it, prefer ``stop_using()``
        or the ``with`` form, which close the file only when no other tree uses it.
        """
        self._file.Close()
        _forget_opened_file(self._file)

    def stop_using(self, close_file=True):
        """Stop using this instance and release the memory it holds

        Every tree instance is kept in the module-level ``grand_tree_list``,
        so it is never garbage collected on its own. When reading many files
        in a loop, call ``stop_using()`` on each tree at the end of the loop,
        or use the tree as a context manager, which calls it on exit::

            with TADC("file.root") as tadc:
                for event, run in tadc.get_list_of_events():
                    tadc.get_event(event, run)

        This removes the instance from ``grand_tree_list`` and, if the tree
        opened its ROOT file itself from a file name, closes that file once no
        other tree in ``grand_tree_list`` uses it. A ``ROOT.TFile`` passed in by
        the caller (for example by ``DataFile``) is left open. Calling it again
        does nothing.

        Do not use the instance after this call: if its file was closed, the
        underlying TTree is gone and ``tree`` is ``None``. A tree that was
        filled but not written is not saved; call ``write()`` first.

        Parameters
        ----------
        close_file : bool, optional
            Close the ROOT file the tree opened, if no other tree uses it.
            Pass ``False`` to only remove the instance from ``grand_tree_list``.
        """
        # Remove by identity: the dataclass __eq__ compares field values and could match another instance
        for i, inst in enumerate(grand_tree_list):
            if inst is self:
                del grand_tree_list[i]
                break

        if close_file:
            self._release_file()

    def _release_file(self):
        """Close the file this tree opened itself, if no other tree in ``grand_tree_list`` uses it"""
        f = self._file
        if f is None or self.is_tchain:
            return
        addr = ROOT.addressof(f)
        # A file handed in by the caller: the caller closes it
        if addr not in _files_opened_by_trees:
            return
        # Another live tree reads from the same file (it was reused through gROOT's list of files)
        for inst in grand_tree_list:
            if inst._file is None:
                continue
            try:
                other = ROOT.addressof(inst._file)
            except TypeError:
                # That tree's file was already deleted (closed elsewhere), so it
                # cannot be sharing this one
                continue
            if other == addr:
                return
        del _files_opened_by_trees[addr]
        if f.IsOpen():
            # Closing deletes the TTree that lives in the file, so drop the handle to it
            f.Close()
        _file_lock.release(f.GetName())
        self._tree = None

    def __enter__(self):
        """Use the tree as a context manager: ``with TADC(file_name) as tadc: ...``

        Returns
        -------
        DataTree
            This tree.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Call ``stop_using()`` when leaving the ``with`` block

        Parameters
        ----------
        exc_type : type
            Type of the exception raised in the block, if any.
        exc_val : BaseException
            The exception raised in the block, if any.
        exc_tb : traceback
            Traceback of the exception, if any.

        Returns
        -------
        bool
            ``False``, so an exception raised in the block propagates.
        """
        self.stop_using()
        return False
