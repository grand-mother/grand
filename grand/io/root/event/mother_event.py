"""

"""

from grand.io.root.base import *

## A mother class for classes with Event values
@dataclass
class MotherEventTree(DataTree):
    """A mother class for classes with Event values"""

    _run_number: np.ndarray = np.zeros(1, np.uint32)
    # ToDo: it seems instances propagate this number among them without setting (but not the run number!). I should find why...
    _event_number: np.ndarray = np.zeros(1, np.uint32)

    @property
    def run_number(self):
        """The run number of the current event"""
        return self._run_number[0]

    @run_number.setter
    def run_number(self, val: np.uint32) -> None:
        self._run_number[0] = val

    @property
    def event_number(self):
        """The event number of the current event"""
        return self._event_number[0]

    @event_number.setter
    def event_number(self, val: np.uint32) -> None:
        self._event_number[0] = val

    def fill(self):
        """Adds the current variable values as a new event to the tree"""
        # If the current run_number and event_number already exist, raise an exception
        if not self.is_unique_event():
            raise NotUniqueEvent(
                f"An event with (run_number,event_number)=({self.run_number},{self.event_number}) already exists in the TTree {self._tree.GetName()}."
            )

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
            if type(inst) is RunTree:
                run_trees.append(inst)
        # If any Run tree was found
        if len(run_trees) > 0:
            # Warning if there is more than 1 RunTree in memory
            if len(run_trees) > 1:
                print(
                    f"More than 1 RunTree detected in memory. Adding the last one {run_trees[-1]} as a friend"
                )
            # Add the last one RunTree as a friend
            run_tree = run_trees[-1]

            # Add the Run TTree as a friend
            self.add_friend(run_tree.tree)

        # Do not add ADCEventTree as a friend to itself
        if not isinstance(self, ADCEventTree):
            # Add the ADC tree as a friend if exists already
            adc_trees = []
            for inst in grand_tree_list:
                if type(inst) is ADCEventTree:
                    adc_trees.append(inst)
            # If any ADC tree was found
            if len(adc_trees) > 0:
                # Warning if there is more than 1 ADCEventTree in memory
                if len(adc_trees) > 1:
                    print(
                        f"More than 1 ADCEventTree detected in memory. Adding the last one {adc_trees[-1]} as a friend"
                    )
                # Add the last one ADCEventTree as a friend
                adc_tree = adc_trees[-1]

                # Add the ADC TTree as a friend
                self.add_friend(adc_tree.tree)

        # Do not add VoltageEventTree as a friend to itself
        if not isinstance(self, VoltageEventTree):
            # Add the Voltage tree as a friend if exists already
            voltage_trees = []
            for inst in grand_tree_list:
                if type(inst) is VoltageEventTree:
                    voltage_trees.append(inst)
            # If any ADC tree was found
            if len(voltage_trees) > 0:
                # Warning if there is more than 1 VoltageEventTree in memory
                if len(voltage_trees) > 1:
                    print(
                        f"More than 1 VoltageEventTree detected in memory. Adding the last one {voltage_trees[-1]} as a friend"
                    )
                # Add the last one VoltageEventTree as a friend
                voltage_tree = voltage_trees[-1]

                # Add the Voltage TTree as a friend
                self.add_friend(voltage_tree.tree)

        # Do not add EfieldEventTree as a friend to itself
        if not isinstance(self, EfieldEventTree):
            # Add the Efield tree as a friend if exists already
            efield_trees = []
            for inst in grand_tree_list:
                if type(inst) is EfieldEventTree:
                    efield_trees.append(inst)
            # If any ADC tree was found
            if len(efield_trees) > 0:
                # Warning if there is more than 1 EfieldEventTree in memory
                if len(efield_trees) > 1:
                    print(
                        f"More than 1 EfieldEventTree detected in memory. Adding the last one {efield_trees[-1]} as a friend"
                    )
                # Add the last one EfieldEventTree as a friend
                efield_tree = efield_trees[-1]

                # Add the Efield TTree as a friend
                self.add_friend(efield_tree.tree)

        # Do not add ShowerEventTree as a friend to itself
        if not isinstance(self, ShowerEventTree):
            # Add the Shower tree as a friend if exists already
            shower_trees = []
            for inst in grand_tree_list:
                if type(inst) is ShowerEventTree:
                    shower_trees.append(inst)
            # If any ADC tree was found
            if len(shower_trees) > 0:
                # Warning if there is more than 1 ShowerEventTree in memory
                if len(shower_trees) > 1:
                    print(
                        f"More than 1 ShowerEventTree detected in memory. Adding the last one {shower_trees[-1]} as a friend"
                    )
                # Add the last one ShowerEventTree as a friend
                shower_tree = shower_trees[-1]

                # Add the Shower TTree as a friend
                self.add_friend(shower_tree.tree)

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
            print(
                f"No event with event number {ev_no} and run number {run_no} in the tree. Please provide proper numbers."
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
