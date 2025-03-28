"Analysis Oriented Interface" - event-based analysis interface examples 
===========================

The Analysis Oriented Interface (AOI) is created for single-event-based analysis tasks, such as Xmax/energy/direction reconstruction, where the analysis task takes much longer than reading the event into memory and writing out the results. For fast I/O the Data Oriented Interface (the dataio module) should be used instead.

In AOI, the most commonly used GRANDROOT data is read to an instance of an Event class. From there the user can access traces, shower parameters, etc. The Event can be initialised manually from a GRANDROOT directory with the provided event and run number, or, preferably, should be obtained from EventList class instance, that iterates over the events in the directory.

The AOI classes are defined in the grand/aoi module.

How to test
-----------

**Generation of (random events)**

Run:
`python event_generation.py`

It will generate a dummy_example_events.root file containing 10 events with random values for Voltage and Efield traces and random values for Shower parameters.

Shows how to fill instances of an Event class and write it to HDD.

**Reading the test data**

Run, after generating the dummy_example_events.root file above:
`python data_play.py dummy_example_events.root`

Rudimentary readout of an Event with event_number=0 and run_number=0 from a file. Prints out the basic information about event and the included traces.

**Reading and display of the real GRAND events**

For real data run:
`python browse_gp13_events_example.py GRANDROOT_data_directory`

For simulated data run:
`python browse_sim2root_events_example.py GRANDROOT_data_directory`

The scripts loop through the events using EventList class and then draw parts of the data of the event, such as traces, on the screen.
