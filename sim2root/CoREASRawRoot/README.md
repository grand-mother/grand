# Coreas to Raw Root Converter
## How to run just CoreasToRawRoot
python3 CoreasToRawRoot <directory with Coreas Sim>

## How to run the whole CoreasToRawRoot + sim2root + efield2voltage
python3 coreas_pipeline.py -d <directory with Coreas Sim>

WARNING: Currently, this only works when there is one shower per directory.

## Overview
### CoreasToRawRoot.py
This Python code defines a function called `CoreasToRawRoot` that performs several tasks related to processing and converting data from a CORSIKA simulation with Coreas output into a ROOT file format. Here's a brief summary/explanation of the code:

1. Importing Libraries:
   - The code begins by importing several Python modules, including `sys`, `glob`, `time`, and custom functions from `CorsikaInfoFuncs` and `raw_root_trees`.

2. Function Definition:
   - The `CoreasToRawRoot` function is defined, which takes a single argument `path`. This function is responsible for converting Coreas output data into a ROOT file.

3. Checking Input Files:
   - The function checks for the existence of specific files (e.g., `.reas`, `.inp`, `.dat`, and `.log`) in the specified directory indicated by `path`.

4. Extracting Information:
   - Various information is extracted from the input files, including simulation parameters, energy, positions, and particle distributions.

5. Creating and Filling ROOT Trees:
   - The code creates and fills several ROOT trees (e.g., `RawShower`, `RawEfield`, and `SimCoreasShower`) with the extracted information.

6. Saving Results:
   - The ROOT trees are written to a ROOT file with a specific filename based on the simulation parameters.

7. Main Function:
   - The code checks if it's being run as a standalone script (not imported as a module) and extracts the `path` argument from the command line. It then calls the `CoreasToRawRoot` function with the provided `path`.

Overall, this code is used to convert Coreas simulation output data into a structured ROOT file format for further analysis and processing. It involves reading various input files, extracting relevant information, and organizing it into ROOT trees within a single output file.


### CorsikaInfoFuncs.py
This Python script provides various functions to read and process data from Corsika simulation files. Corsika is a particle physics simulation software used for studying high-energy cosmic rays.

The script includes the following functions:

- `find_input_vals(line)`: Reads single numerical values from Corsika `SIM.reas` or `RUN.inp` files.

- `find_input_vals_list(line)`: Reads lists of numerical values from Corsika `SIM.reas` or `RUN.inp` files.

- `read_params(input_file, param)`: Reads a single numerical value associated with a specified parameter from Corsika `SIM.reas` or `RUN.inp` files.

- `read_list_of_params(input_file, param)`: Reads a list of numerical values associated with a specified parameter from Corsika `SIM.reas` or `RUN.inp` files.

- `read_atmos(input_file)`: Reads atmospheric information from a Corsika `RUN.inp` file.

- `read_date(input_file)`: Reads the date information from a Corsika `RUN.inp` file.

- `read_site(input_file)`: Reads the site information (e.g., Dunhuang, Lenghu) from a Corsika `RUN.inp` file.

- `read_first_interaction(log_file)`: Reads the height of the first interaction from a Corsika log file.

- `read_HADRONIC_INTERACTION(log_file)`: Reads information about the hadronic interaction model used from a Corsika log file.

- `read_coreas_version(log_file)`: Reads the CoREAS version used from a Corsika log file.

- `antenna_positions_dict(pathAntennaList)`: Parses antenna positions from a Corsika `SIM??????.list` file and stores them in a dictionary.

- `get_antenna_position(pathAntennaList, antenna)`: Retrieves the position of a specific antenna from a Corsika `SIM??????.list` file.

- `read_long(pathLongFile)`: Reads the longitudinal profile data from a Corsika `.long` output file.
