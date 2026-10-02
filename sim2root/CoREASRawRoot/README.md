# Coreas to Raw Root Converter

## How to run just CoreasToRawROOT.py
### Convert Multiple Showers in a Directory
To convert data from multiple CoREAS showers located in the same directory, use the following command:
`python3 CoreasToRawROOT.py -d <path_to_directory>`
The code will search for CoREAS .reas files in the specified directory and convert each of them into the GRANDROOT format.

### Convert a Single CoREAS Shower
To convert data from a single CoREAS shower, use the following command:
`python3 CoreasToRawROOT.py --file <path_to_SIMxxxxxx.reas>`
The code will convert the specified CoREAS simulation into GRANDROOT format.

### Where the output goes
Each shower is written to `Coreas_<simID>.rawroot` in the current folder, or in
the folder given with `-o <folder>` (for a single shower `-o` may also name the
file). An existing output file is refused, rather than appended to; give
`--overwrite` to replace it. This folder already holds the committed sample
`Coreas_004100.rawroot`, so converting `proton/` here needs `-o` or
`--overwrite`.


## How to run the whole chain (CoreasToRawROOT + sim2root + efield2voltage)
There is no single script for it any more. `../Common/RunSimPipeNoJitter.py`
runs the steps after this converter on a folder of `.rawroot` files:\
`python3 ../Common/RunSimPipeNoJitter.py <folder with .rawroot files> <extra> -sl GP300`


## Overview
### CoreasToRawROOT.py
This Python code defines a function called `CoreasToRawRoot` that performs several tasks related to processing and converting data from a CORSIKA simulation with Coreas output into a ROOT file format. Here's a brief summary/explanation of the code:

1. Importing Libraries:
   - The code begins by importing several Python modules, including `sys`, `glob`, `time`, and custom functions from `CorsikaInfoFuncs` and `raw_root_trees`.

2. Function Definition:
   - `CoreasToRawRoot(file, simID=None, output=".", overwrite=False)` converts one simulation, given by its `SIM??????.reas` file, into `Coreas_<simID>.rawroot`.

3. Checking Input Files:
   - The function checks for the existence of specific files (e.g., `.reas`, `.inp`, `.dat`, and `.log`) beside the `.reas` file.

4. Extracting Information:
   - Various information is extracted from the input files, including simulation parameters, energy, positions, and particle distributions.

5. Creating and Filling ROOT Trees:
   - The code creates and fills several ROOT trees (e.g., `RawShower`, `RawEfield`, and `SimCoreasShower`) with the extracted information.
   - All of them use run number = the `.reas` `EventNumber` and event number = the simulation number (4100 for `SIM004100`).
   - The magnetic field strength is in µT, as from ZHAireS (#232); files converted before that hold it in mT (`.reas` path) or Gauss (`.inp` path).
   - CoREAS gives no event time, so it is written as 0 and sim2root uses the simulation date; an unknown first-interaction height or injection altitude is NaN.

6. Saving Results:
   - The ROOT trees are written to a ROOT file with a specific filename based on the simulation parameters.

7. Main Function:
   - Run as a script, it reads `--file`, `-d`, `-o` and `--overwrite` (see above) and converts each simulation.

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
