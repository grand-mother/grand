# Created by Lech Wiktor Piotrowski at 14/03/2025
from pathlib import Path
import ROOT

from grand.aoi.event import Event
from grand.dataio import DataDirectory, DataFile


class EventList:
    """A class giving access/iteration over multiple events"""

    ## The instance of the file with TTrees containing the event. ToDo: this should allow for multiple files holding different TTrees and TChains in the future
    file: ROOT.TFile = None
    """The instance of the file with TTrees containing the event."""

    directory: DataDirectory = None
    """The instance of the directory with files with TTrees containing the event."""

    def __init__(self, inp_name, start_event = None, start_entry = None, **kwargs):

        r"""Opens a file or directory and prepares to iterate its events.

        Parameters
        ----------
        inp_name : str
            File or directory to read.
        start_event : int, optional
            Event number to begin at.
        start_entry : int, optional
            Entry index to begin at, used instead of `start_event`.
        """
        self.event_list = None

        # If TFile was given
        if isinstance(inp_name, ROOT.TFile):
            self.file_name = inp_name.GetName()
            self.file = inp_name
        # If DataDirectory was given
        elif isinstance(inp_name, DataDirectory):
            self.directory_name = inp_name.dir_name
            self.directory = inp_name
        # String with file name or directory name was given
        elif isinstance(inp_name, str):
            # If file name was given
            if Path(inp_name).is_file():
                self.file = DataFile(ROOT.TFile(inp_name, "read"))
                # self.file = ROOT.TFile(inp_name, "read")
                self.event_list = self.file.get_max_list_of_events()
            # If directory name was given
            elif Path(inp_name).is_dir():
                self.directory = DataDirectory(inp_name)
                self.event_list = self.directory.get_max_list_of_events()
            else:
                print("Please provide proper file or directory name.")
                exit()

        if start_event is not None and start_entry is not None:
            print("Please provide only start event or start entry.")
            exit()
        self.start_event = start_event
        self.start_entry = start_entry

        # The arguments to be passed to Event.fill_event_from_trees()
        self.init_kwargs = kwargs

        self.event = Event()
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
            The event, or ``None`` when it was not found.
        """

        # Don't allow specifying entry and event/run at the same time, because... what to chose?
        if entry_number is not None and (run_number is not None or event_number is not None):
            print("Please provide only entry_number or event/run_number!")
            return None

        e = self.event

        if self.file is not None:
            e.file = self.file.f
        elif self.directory is not None:
            e.directory = self.directory
        else:
            print("No directory or file provided!")
            exit()

        # If entry/event/run number not specified, take the first entry
        run_entry_number = None
        if entry_number is None and run_number is None and event_number is None:
            entry_number = 0

        if entry_number is not None:
            e._entry_number = entry_number
        else:
            if run_number is None:
                run_number = 0
            if event_number is not None:
                e.run_number=run_number
                e.event_number=event_number
            else:
                print("Please provide event_number and run_number, or entry_number")
                return None

        # Fill the event
        if fill_event:
            # Overwrite the init kwargs with kwargs given here
            if len(kwargs)>0:
                e.fill_event_from_trees(init_trees=self.init_trees, event_number = event_number, run_number = run_number, **kwargs)
            else:
                e.fill_event_from_trees(init_trees=self.init_trees, event_number = event_number, run_number = run_number, **self.init_kwargs)

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
            data_input = DataFile(self.file)
        else:
            print("Please provide data directory or file.")
            exit()

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
        else:
            print("Can not find any tree to provide the number of events in the file.")
            return None

    ## Return the iterable over self
    def __iter__(self):
        r"""Yields each event in turn.

        Yields
        ------
        Event
            The next event, fully populated.
        """
        for event_num, run_num in self.event_list:
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
