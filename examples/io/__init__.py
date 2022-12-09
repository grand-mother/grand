"""!
@section How to test
-----------

**Storing of the data test**

Run:
python DataStoringExample.py

It will generate a _stored_data.root_ file containing GRANDADCCounts, GRANDVoltage and GRANDEfield TTrees with random data, that are associated as TTree friends (loading an event from Efield should load a corresponding event from Voltage and ADCCounts, etc.)

**Reading od the data test**

Run, after generating the _stored_data.root_ file above:
python DataReadingExample.py

Currently the reading example is very basic.

"""
