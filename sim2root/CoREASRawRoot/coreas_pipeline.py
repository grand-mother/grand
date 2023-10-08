# author: Jelena
import subprocess
from optparse import OptionParser
import glob
from CorsikaInfoFuncs import read_params

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
        print(f"Searching directory {options.directory} for .reas files")
        # find .reas files with glob
        reas_names = glob.glob(options.directory + "/SIM??????.reas")
        # use ** if you want to go through all subdirectories, use * if you want to go only one level deeper
        print(f"Found {len(reas_names)} shower(s)")
        print(reas_names)
        # loop over all reas files
        #! currently CoreasToRawROOT only takes one shower per directory
        


        for reas_filename in reas_names:
            print("********************************")
            print(f"Now analyzing {reas_filename}")
            # get run number from inp file:
            runID = int(read_params(reas_filename.split(".reas")[0] + ".inp", "RUNNR"))
            print(f"run number: {runID}")
            # get zenith from inp file:
            zenith = int(read_params(reas_filename.split(".reas")[0] + ".inp", "THETAP"))
            print(f"Zenith: {zenith} degrees")
            # get obslevel from reas file:
            obslevel = int(read_params(reas_filename, "CoreCoordinateVertical")) / 100 # from cm to m
            print(f"Observation level: {obslevel} meters")
            
            print("* - * - * - * - * - * - * - * - * - *")
            print(f"Converting Coreas Simulation {runID} to RawRoot format...")

            # Run CoreasToRawROOT.py
            CoreasToRawROOT = [
                'python3', 'CoreasToRawROOT.py', str(options.directory)
            ]
            subprocess.run(CoreasToRawROOT, check=True)
            print(f"Created Coreas_Run_{runID}.root")

            print("* - * - * - * - * - * - * - * - * - *")
            print(f"Converting from RawRoot to GRANDroot format...")

            # Run sim2root.py
            sim2root = [
                'python3', '../Common/sim2root.py', f"Coreas_Run_{runID}.root"
            ]
            subprocess.run(sim2root, check=True)
            print(f"Created gr_Coreas_Run_{runID}.root")

            print("* - * - * - * - * - * - * - * - * - *")
            print(f"Converting traces from efield to voltage...")

            # Run convert_efield2voltage.py
            sim2root = [
                'python3', '../../scripts/convert_efield2voltage.py', f"gr_Coreas_Run_{runID}.root",\
                f"-o {options.output}efield_gr_Coreas_Run_{runID}.root"
            ]
            subprocess.run(sim2root, check=True)
            print(f"Created efield_gr_Coreas_Run_{runID}.root")
            print("********************************")
            pass

        print(f"Finished analyzing files in {options.directory}")
        print("********************************")
