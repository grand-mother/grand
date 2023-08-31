#!/usr/bin/python
## Extra functions used for conversion of Coreas simulations to GRANDRaw ROOT files
## by Jelena Köhler

import re
from re import search
import io
import numpy as np


# read single values from SIM.reas or RUN.inp
def find_input_vals(line):
  return search(r'[-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', line)



# read a list of values from SIM.reas or RUN.inp
def find_input_vals_list(line):
  # basically search for up to four values with the same rules as in find_input_vals
  match1 = search(r'[-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', line)
  match2 = search(r'([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?)', line)
  match3 = search(r'([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?)', line)
  match4 = search(r'([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?)', line)
  match5 = search(r'([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?) ([-+]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?)', line)
  
  if match5:
    return [match5.group(1), match5.group(2), match5.group(3), match5.group(4), match5.group(5)]
  elif match4:
    return [match4.group(1), match4.group(2), match4.group(3), match4.group(4)]
  elif match3:
    return [match3.group(1), match3.group(2), match3.group(3)]
  elif match2:
    return [match2.group(1), match2.group(2)]
  else:
     return match1
  



# read single values from SIM.reas or RUN.inp
def read_params(input_file, param):
  # works for both SIM.reas and RUN.inp, as long as you are looking for numbers
  val = ""
  with open(input_file, "r") as datafile:
    for line in datafile:
      if param in line:
        line = line.lstrip()
        if find_input_vals(line):
          val = find_input_vals(line).group()
          print(param, "=", val)
          break 
          # this is a problem for AutomaticTimeBoundaries, because it also shows up in other comments
          # therefore, just break after the first one is found. this can definitely be improved
  return float(val)



# read a list of values from SIM.reas or RUN.inp
def read_list_of_params(input_file, param):
    # works for both SIM.reas and RUN.inp, as long as you are looking for numbers
    with open(input_file, "r") as datafile:
        for line in datafile:
            if param in line:
                line = line.lstrip()
                if find_input_vals_list(line):
                    list = find_input_vals_list(line)
                    print(param, "=", list)
                    break 
                # this is a problem for AutomaticTimeBoundaries, because it also shows up in other comments
                # therefore, just break after the first one is found. this can definitely be improved
    return list





def read_atmos(input_file):
    # RUN.inp only
    with open(input_file, mode="r") as datafile:
        for line in datafile:
            if "ATMFILE" in line:
                atmos = line.split("/")[-1]
    return atmos



def read_site(input_file):
    # RUN.inp only
    atmos = read_atmos(input_file)
    if "Dunhuang" in atmos:
        site = "Dunhuang"
    elif "Lenghu" in atmos:
        site = "Lenghu"
    else:
        site = atmos
    return site



def read_first_interaction(log_file):
    """
    For now, get this from the log.
    (Yes, this is ugly and unreliable, but it's good enough for now)
    I want to change the Coreas output so this is included in the reas file instead.
    """
    with open(log_file, mode="r") as datafile:
        for line in datafile:
            if "height of first interaction" in line:
                val = line.split("interaction")[-1]
                first_interaction = find_input_vals(val).group()
                print("first interaction =", first_interaction)
    return float(first_interaction)



def read_HADRONIC_INTERACTION(log_file):
    with open(log_file, mode="r") as datafile:
        for line in datafile:
            if "S I B Y L L  2.3d" in line:
                hadr_interaction = "Sibyll 2.3d"
                print("hadronic interaction model =", hadr_interaction)
            else:
                hadr_interaction = "n/a"
                print("hadronic interaction model =", hadr_interaction)
    return str(hadr_interaction)



def read_coreas_version(log_file):
    with open(log_file, mode="r") as datafile:
        for line in datafile:
            if "CoREAS V1.4" in line:
                coreas_version = "CoREAS V1.4"
                print("CoREAS version =", coreas_version)
            else:
                coreas_version = "n/a"
                print("CoREAS version =", coreas_version)
    return str(coreas_version)



def antenna_positions_dict(pathAntennaList):
    """
    get antenna positions from SIM??????.list and store in a dictionary
    .list files are structured like "AntennaPosition = x y z name"

    """
    antennaInfo = {} # store info in a dict

    # get antenna positions from file
    file = np.genfromtxt(pathAntennaList, dtype = "str")
    # file[:,0] and file[:,1] are useless (they are simply "AntennaPosition" and "=")
    
    # get the x, y and z positions
    antennaInfo["x"] = file[:,2].astype(float) * 100 # convert to m
    antennaInfo["y"] = file[:,3].astype(float) * 100 # convert to m
    antennaInfo["z"] = file[:,4].astype(float) * 100 # convert to m
    # get the IDs of the antennas
    antennaInfo["name"] = file[:,5].astype(str)

    # Extract the IDs of the antennas from the names
    antennaInfo["ID"] = []
    # Use a counter to give generic IDs to antennas with unknown names
    generic_id_counter = 1

    for name in file[:, 5]:
        GP_match = search(r'gp_(\d+)', name, flags=re.IGNORECASE)  # Extract digits after last underscore

        if GP_match: # match GP13
            antennaInfo["ID"].append(int(GP_match.group(1))) # Group 1 contains the ID

        else: # Give generic IDs to antennas with other names
            antennaInfo["ID"].append(int(generic_id_counter))
            generic_id_counter += 1

    return antennaInfo



def get_antenna_position(pathAntennaList, antenna):
    """
    get the position for one antenna from SIM??????.list
    .list files are structured like "AntennaPosition = x y z name"

    """
    file = np.genfromtxt(pathAntennaList, dtype = "str")
    # get antenna positions from file
    # file[:,0] and file[:,1] are useless (they are simply "AntennaPosition" and "=")
    # get the x, y and z positions
    x = file[:,2].astype(float) * 100 # convert to m
    y = file[:,3].astype(float) * 100 # convert to m
    z = file[:,4].astype(float) * 100 # convert to m
    # get the names of the antennas
    name = file[:,5]

    return x, y, z



def read_long(pathLongFile):
    """
    read the .long Corsika output file, which contains the "longitudinal profile"
    more specifically, it contains the energy deposit and particle numbers for different particles
    since the files are set up in two blocks, this function helps read the data from it
    
    this function is mostly taken from corsika_long_parser.py in the coreasutilities module by Felix Schlüter

    TODO: fix hillas_parameter - something's not working yet
    """
    with open(pathLongFile, mode="r") as file:
        # create a temporary file to write the corrected contents
        temp_file = io.StringIO()

        for line in file:
            # use a regex to search for a minus sign that is not part of an exponent
            if search(r"(?<!e)(-)(?=\d)", line):
                # if the minus sign is not part of an exponent, replace it with a space and a minus sign
                line = line.replace("-", " -")
            # write the corrected line to the temporary file
            temp_file.write(line)

        # set the file pointer to the beginning of the temporary file
        temp_file.seek(0)

        # read the contents of the temporary file into a list of strings
        lines = temp_file.readlines()



    n_steps = int(lines[0].rstrip().split()[3])

    # store n table
    n_data_str = io.StringIO()
    n_data_str.writelines(lines[2:(n_steps + 2)])
    n_data_str.seek(0)
    n_data = np.genfromtxt(n_data_str)

    # store dE table
    dE_data_str = io.StringIO()
    dE_data_str.writelines(lines[(n_steps + 4):(2 * n_steps + 4)])
    dE_data_str.seek(0)
    dE_data = np.genfromtxt(dE_data_str)

    # read out hillas fit
    hillas_parameter = []
    # for line in lines:
    #     if bool(search("PARAMETERS", line)):
    #         hillas_parameter = [float(x) for x in line.split()[2:]]
    #     if bool(search("CHI", line)):
    #         hillas_parameter.append(float(line.split()[2]))


    print("The file", pathLongFile, "has been separated into energy deposit and particle distribution.")
    return n_data, dE_data, hillas_parameter
