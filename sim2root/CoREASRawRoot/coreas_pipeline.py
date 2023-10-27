# author: Jelena
import subprocess
from optparse import OptionParser
import glob
from CorsikaInfoFuncs import read_params
import sys
import re

parser = OptionParser()
parser.add_option("--directory", "--dir", "-d", type="str",
                  help="Specify the full path to the (inp) directory of the Coreas simulation set.\
                  ")
parser.add_option("--output", "--out", "-o", type="str",
                  help="Specify the where you want to store the converted simulation set.\
                  ")

(options, args) = parser.parse_args()

if __name__ == '__main__':
    if options.directory:
        path = f"{options.directory}/"
        # find reas files in directory
        if glob.glob(path + "SIM??????.reas"):
            available_reas_files = glob.glob(path + "SIM??????.reas")
        else:
            print("No showers found. Please check your input and try again.")
            sys.exit()
        
        print(f"Found {len(available_reas_files)} shower(s)")
        print(available_reas_files)
        # loop over all reas files

        for reas_file in available_reas_files:
            shower_match = re.search(r'SIM(\d{6})\.reas', reas_file)
            if shower_match:
                simID = shower_match.group(1)
                print(f"run number: {simID}")
            else:
                print(f"No simID found for {reas_file}. Please check your input and try again.")
                sys.exit()
            print("********************************")
            print(f"Now analyzing {reas_file}")
            
            # get zenith from inp file:
            zenith = int(read_params(reas_file.split(".reas")[0] + ".inp", "THETAP"))
            print(f"Zenith: {zenith} degrees")
            # get obslevel from reas file:
            obslevel = int(read_params(reas_file, "CoreCoordinateVertical")) / 100 # from cm to m
            print(f"Observation level: {obslevel} meters")
            
            print("* - * - * - * - * - * - * - * - * - *")
            print(f"Converting Coreas Simulation {simID} to RawRoot format...")

            # Run CoreasToRawROOT.py
            print("executing CoreasToRawROOT.py")
            CoreasToRawROOT = [
                'python3', 'CoreasToRawROOT.py', '-d', str(options.directory)
            ]
            subprocess.run(CoreasToRawROOT, check=True)
            print(f"Created Coreas_Run_{simID}.root")

            print("* - * - * - * - * - * - * - * - * - *")
            print(f"Converting from RawRoot to GRANDroot format...")

            # Run sim2root.py
            print("executing sim2root.py")
            sim2root = [
                'python3', '../Common/sim2root.py', f"Coreas_Run_{simID}.root"
            ]
            subprocess.run(sim2root, check=True)
            print(f"Created gr_Coreas_Run_{simID}.root")

            print("* - * - * - * - * - * - * - * - * - *")
            print(f"Converting traces from efield to voltage...")

            print("executing convert_efield2voltage.py")
            # Run convert_efield2voltage.py
            sim2root = [
                'python3', '../../scripts/convert_efield2voltage.py', f"gr_Coreas_Run_{simID}.root",\
                f"-o {options.output}efield_gr_Coreas_Run_{simID}.root"
            ]
            subprocess.run(sim2root, check=True)
            print(f"Created efield_gr_Coreas_Run_{simID}.root")
            print("********************************")
            pass

        print(f"Finished analyzing files in {options.directory}")
        print("********************************")
