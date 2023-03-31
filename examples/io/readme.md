GRANDROOT basic information
===========================

This is the interface for accessing GRAND ROOT TTrees that (in its final future state) will not require the user (reader/writer of the TTrees) to have any knowledge of ROOT. It also will hide the internals from the data generator, so that the changes in the format are not concerning the user.

The TTree interface classes are defined in the GRANDROOTTrees.py file.

How to test
-----------

**Storing of the data test**

Run:
python data_storing.py

It will generate a stored_data.root file containing GRANDADCCounts, GRANDVoltage and GRANDEfield TTrees with random data, that are associated as TTree friends (loading an event from Efield should load a corresponding event from Voltage and ADCCounts, etc.)

**Reading the test data**

Run, after generating the stored_data.root file above:
python data_reading.py

Currently the reading example is very basic.