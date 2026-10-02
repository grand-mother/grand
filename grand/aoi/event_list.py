# Created by Lech Wiktor Piotrowski at 14/03/2025
import logging
import os
from pathlib import Path
import ROOT
import numpy as np

from grand.aoi.event import Event
from grand.dataio import DataDirectory, DataFile
from grand.basis import validate as _validate

logger = logging.getLogger(__name__)


class EventList:
    """A class giving access/iteration over multiple events

    Every call to :meth:`get_event`, and every step of an iteration, fills and
    returns the *same* :class:`Event` object, so ``list(EventList(d))`` holds
    that one object several times, showing the last event. Copy what you need
    from each event before reading the next.

    Examples
    --------
    Loop over the events of a simulation folder.  The same ``Event`` object is
    reused, so copy out what you need rather than keeping ``event``:

    .. jupyter-execute::

        from pathlib import Path

        import grand
        from grand.aoi.event_list import EventList

        sample = (Path(grand.__file__).parents[1]
                  / "sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
        for event in EventList(str(sample)):
            print("event %5d: zenith %5.2f deg, %2d antennas"
                  % (event.event_number, event.simshower.zenith, len(event.antennas)))
    """

    ## The instance of the file with TTrees containing the event. ToDo: this should allow for multiple files holding different TTrees and TChains in the future
    file: ROOT.TFile = None
    """The instance of the file with TTrees containing the event."""

    directory: DataDirectory = None
    """The instance of the directory with files with TTrees containing the event."""

    def __init__(self, inp_name, start_event = None, start_entry = None, tefield_level = None, gps_origin = None, **kwargs):

        r"""Opens a file or directory and prepares to iterate its events.

        Parameters
        ----------
        inp_name : str
            File or directory to read.
        start_event : int, optional
            Event number to begin at.
        start_entry : int, optional
            Entry index to begin at, used instead of `start_event`.
        tefield_level : int, optional
            Analysis level of the electric field to read, for every event
            unless a call to :meth:`get_event` asks for another.
        gps_origin : None, "run" or (float, float, float), optional
            Origin of antenna positions computed from GPS (GP300, GP80,
            GP13): see :attr:`grand.aoi.event.Event.gps_origin`.
        """
        self.event_list = None

        if isinstance(inp_name, os.PathLike):
            inp_name = os.fspath(inp_name)
        accepted = tuple(t for t in (str, ROOT.TFile, DataDirectory, DataFile) if isinstance(t, type))
        if not isinstance(inp_name, accepted):
            raise TypeError(_validate.message(
                "EventList", "the input must be a file or directory name, a ROOT.TFile, a "
                "DataFile or a DataDirectory, got %s" % type(inp_name).__name__))

        # If TFile was given.  It is wrapped like a file name is: the rest of
        # the class reads the file through a DataFile (``self.file.f``).
        if isinstance(inp_name, ROOT.TFile):
            self.file_name = inp_name.GetName()
            self.file = DataFile(inp_name)
            self.event_list = self.file.get_max_list_of_events()
        # If DataDirectory was given
        elif isinstance(inp_name, DataDirectory):
            self.directory_name = inp_name.dir_name
            self.directory = inp_name
            self.event_list = self.directory.get_max_list_of_events()
        # String with file name or directory name was given
        elif isinstance(inp_name, str):
            # If file name was given
            if Path(inp_name).is_file():
                # DataFile opens it with a clear error for an empty, text or
                # damaged file, rather than cppyy's bare OSError (#235)
                self.file = DataFile(inp_name)
                # self.file = ROOT.TFile(inp_name, "read")
                self.event_list = self.file.get_max_list_of_events()
            # If directory name was given
            elif Path(inp_name).is_dir():
                if not any(Path(inp_name).glob("*.root")):
                    raise FileNotFoundError(_validate.message(
                        "EventList", "no ROOT files (*.root) in %s" % inp_name))
                self.directory = DataDirectory(inp_name)
                self.event_list = self.directory.get_max_list_of_events()
                # Nothing recognized: this failed later on None, as "'NoneType'
                # object has no attribute 'f'" (#236)
                if self.event_list is None:
                    names = [os.path.basename(name) for name in self.directory.unrecognised_files]
                    raise FileNotFoundError(_validate.message(
                        "EventList", "no GRAND event files recognised in %s%s; files must be named "
                        "<type>_<events>_L<level>_<serial>.root, e.g. efield_1-2_L0_0000.root, or be "
                        "opened one at a time" % (inp_name, " (found %s)" % ", ".join(names[:5]) if names else "")))
            else:
                raise FileNotFoundError(_validate.message(
                    "EventList", "no such file or directory: %s" % inp_name))
        # If DataFile was given
        elif isinstance(inp_name, DataFile):
            self.file_name = inp_name.f.GetName()
            self.file = inp_name
            self.event_list = self.file.get_max_list_of_events()

        if start_event is not None and start_entry is not None:
            raise ValueError(_validate.message(
                "EventList", "give 'start_event' or 'start_entry', not both"))
        self.start_event = start_event
        self.start_entry = start_entry
        self.tefield_level = tefield_level

        # The arguments to be passed to Event.fill_event_from_trees()
        self.init_kwargs = kwargs

        self.event = Event(tefield_level = tefield_level, gps_origin = gps_origin)
        self.init_trees = True

        # No need to init trees if using a DataDirectory (which inits the trees)
        if self.directory:
            self.init_trees = False

    def get_event(self, event_number=None, run_number=None, entry_number=None, fill_event=True, **kwargs):
        """Get specified event from the event list

        Parameters
        ----------
        event_number : int, optional
            Event number.
        run_number : int, optional
            Run number.
        entry_number : int, optional
            Entry index, instead of the pair above.
        fill_event : bool, optional
            Populate the event from every tree, rather than only locating it.

        Returns
        -------
        Event
            The event.

        Raises
        ------
        ValueError
            If both an entry and an event/run number are given, or a run
            number without an event number.
        LookupError
            If the input holds no such event.
        """

        # Don't allow specifying entry and event/run at the same time, because... what to chose?
        if entry_number is not None and (run_number is not None or event_number is not None):
            raise ValueError(_validate.message(
                "EventList.get_event", "give entry_number, or event_number and run_number, not both"))

        e = self.event

        if self.file is not None:
            e.file = self.file.f
        elif self.directory is not None:
            e.directory = self.directory
        else:
            raise RuntimeError(_validate.message("EventList", "no file or directory to read"))

        # If entry/event/run number not specified, take the first entry
        run_entry_number = None
        if entry_number is None and run_number is None and event_number is None:
            entry_number = 0

        if entry_number is not None:
            # Checked here: out of range it failed deep in the reader with
            # "zero-size array to reduction operation minimum", and a bool or
            # a float reached cppyy (#235)
            if isinstance(entry_number, bool) or not isinstance(entry_number, (int, np.integer)):
                raise TypeError(_validate.message(
                    "EventList.get_event", "'entry_number' must be an integer, got %r" % (entry_number,)))
            count = self.get_number_of_events()
            if count is None and self.event_list is not None:
                count = len(self.event_list)
            if entry_number < 0 or (count is not None and entry_number >= count):
                raise IndexError(_validate.message(
                    "EventList.get_event", "entry_number %s is out of range: the input holds %s "
                    "events" % (entry_number, count)))
            e._entry_number = int(entry_number)
        else:
            if run_number is None:
                run_number = 0
            if event_number is not None:
                # An event that is not in the input used to crash deep in the
                # reader (a zero-size minimum) or, after a valid event, to
                # come back labeled with the requested number but holding
                # the previous event's traces (issue #95).  The list of events
                # is known when a file or directory name was given.
                if (self.event_list is not None
                        and (event_number, run_number) not in
                        {(int(ev), int(run)) for ev, run in self.event_list}):
                    raise LookupError(_validate.message(
                        "EventList.get_event", "no event with event number %s and run number %s; "
                        "this input holds %d events: %s"
                        % (event_number, run_number, len(self.event_list), self.event_list[:10])))
                e.run_number=run_number
                e.event_number=event_number
            else:
                raise ValueError(_validate.message(
                    "EventList.get_event", "give event_number with run_number, or entry_number"))

        # Fill the event
        if fill_event:
            # Overwrite the init kwargs with kwargs given here
            options = dict(kwargs) if len(kwargs) > 0 else dict(self.init_kwargs)
            # The level is passed on every call, so a per-call level is used
            # and does not stick to later calls (#213)
            options.setdefault("tefield_level", self.tefield_level)
            e.fill_event_from_trees(init_trees=self.init_trees, event_number = event_number, run_number = run_number, **options)

            # Don't init trees anymore
            self.init_trees = False

        return e

    def get_number_of_events(self):
        """Get the number of events in the list

        Returns
        -------
        int
            Number of events available.
        """

        # ToDo: at the moment assumes the same number of events in all the trees
        # Read directory if given
        if self.directory:
            data_input = self.directory
        elif self.file:
            # Already a DataFile; wrapping it again raised a TypeError
            data_input = self.file
        else:
            raise RuntimeError(_validate.message("EventList", "no file or directory to read"))

        # First, try to get the number of events from tshower
        # if hasattr(data_input, "tshower"):
        if hasattr(data_input, "tshower") and data_input.tshower:
            return data_input.tshower.get_entries()
        elif hasattr(data_input, "tefield") and data_input.tefield:
            return data_input.tefield.get_entries()
        elif hasattr(data_input, "tvoltage") and data_input.tvoltage:
            return data_input.tvoltage.get_entries()
        elif hasattr(data_input, "trawvoltage") and data_input.trawvoltage:
            return data_input.trawvoltage.get_entries()
        #elif hasattr(data_input, "trecons") and data_input.trecons:
        #    return data_input.trecons.get_entries()
        else:
            # None, as distinct from 0 entries; logged, not printed (#256)
            logger.warning("Can not find any tree to provide the number of events in the file.")
            return None

    ## Return the iterable over self
    def __iter__(self):
        r"""Yields each event in turn, from `start_event` or `start_entry` if given.

        Yields
        ------
        Event
            The next event, fully populated.  It is the same object each time,
            refilled: see the class description.

        Raises
        ------
        ValueError
            If `start_event` or `start_entry` is not in the input.
        """
        # start_event and start_entry were stored but ignored (#213)
        events = list(self.event_list)
        first = 0
        if self.start_entry is not None:
            if not 0 <= self.start_entry < len(events):
                raise ValueError(_validate.message(
                    "EventList", "start_entry %s is out of range: the input holds %d events"
                    % (self.start_entry, len(events))))
            first = self.start_entry
        elif self.start_event is not None:
            numbers = [int(ev) for ev, _ in events]
            if self.start_event not in numbers:
                raise ValueError(_validate.message(
                    "EventList", "start_event %s is not in the input; it holds %s"
                    % (self.start_event, numbers[:10])))
            first = numbers.index(self.start_event)
        for event_num, run_num in events[first:]:
            yield self.get_event(event_number=event_num, run_number=run_num)

        # # If this is the first event, and start_entry was specified
        # if self.start_entry:
        #     current_entry = self.start_entry
        # else:
        #     # Always start the iteration with the first entry
        #     current_entry = 0
        #
        #
        # entries_cnt = self.get_number_of_events()
        #
        # while current_entry < entries_cnt:
        #     # If this is the first event, and start_entry was specified
        #     if current_entry == 0 and self.start_event:
        #         self.event._entry_number = None
        #         yield self.get_event(event_number=self.start_event)
        #         # ToDo: We need to get the entry for this event. This is a dirty hack, to check which tree is available
        #         if self.event.tvoltage:
        #             current_entry = self.event.tvoltage._tree.GetReadEntry()
        #         elif self.event.tefield:
        #             current_entry = self.event.tefield._tree.GetReadEntry()
        #         elif self.event.tshower:
        #             current_entry = self.event.tshower._tree.GetReadEntry()
        #         else:
        #             print("No tree available to iterate on")
        #             exit()
        #     else:
        #         # Standard iterations
        #         yield self.get_event(entry_number=current_entry)
        #
        #     current_entry += 1
