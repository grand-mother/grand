# author: Jelena
import subprocess
from optparse import OptionParser
import glob
from CorsikaInfoFuncs import read_params
import sys
import re
import os

parser = OptionParser()
parser.add_option("--directory", "--dir", "-d", type="str",
                  help="Specify the full path to the (inp) directory of the Coreas simulation set."
                  )
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
            print(reas_file)
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
            
            # * - * - * - * - * - * - *
            # * produce RawROOT
            # Run CoreasToRawROOT.py
            print("producing RawROOT file...")
            CoreasToRawROOT = [
                "python3", "./CoreasToRawROOT.py", "-f", f"{str(reas_file)}"
            ]
            subprocess.run(CoreasToRawROOT, check=True)
            print(f"Created Coreas_{simID}.rawroot")
            
            print("* - * - * - * - * - * - * - * - * - *")
            print(f"Converting from RawRoot to GRANDroot format...")
            
            # * - * - * - * - * - * - *
            # * produce GRANDroot
            print("********************************")
            print("producing GRANDroot files...")
            # Run sim2root.py
            sim2root = [
                "python3", "../Common/sim2root.py", f"Coreas_{simID}.rawroot", "-o", f"{str(options.directory)}", "--target_duration_us=4.096", "--trigger_time_ns=800"
            ]
            result=subprocess.run(sim2root, check=True, stdout=subprocess.PIPE, text=True)
            print(f"Created grandroot trees in {str(options.directory)}")
            
            # get the path to the efield.root file
            sim2root_file = glob.glob(str(options.directory)+"/sim_*/efield*")[0]
            # get the path to the directory 
            sim2root_out  = os.path.dirname(sim2root_file)

            output  = result.stdout
            for line in output.splitlines():
                if "Output directory:" in line:
                    out_dir = line.split("Output directory:")[1].strip()
                    break
            print(out_dir)
            sim2root_out=out_dir
            print("* - * - * - * - * - * - * - * - * - *")
            #print("stop now")
            #sys.exit("stopped")
            
            # * - * - * - * - * - * - * - *
            # * produce voltage
            print("********************************")
            print("producing voltage files...")
            # Run convert_efield2voltage.py
            voltage = [
                "python3", "../../scripts/convert_efield2voltage.py", f"{sim2root_out}", "--seed=1234", "--add_jitter_ns=5", "--calibration_smearing_sigma=0.075", "--verbose=info"
            ]
                    
            subprocess.run(voltage, check=True)
            print(f"Created voltage files in {sim2root_out}.")
            print("********************************")
            
            # * - * - * - * - * - * - * - *
            # * produce ADC
            print("********************************")
            print("producing ADC files...")
            adc = [
                "python3", "../../scripts/convert_voltage2adc.py", f"{sim2root_out}"
            ]
            subprocess.run(adc, check=True)
            print(f"Created ADC files in {sim2root_out}.")
            print("********************************")
            
            # * - * - * - * - * - * - * - *
            # * produce DC2 efield
            print("********************************")
            print("producing DC2 efield files...")
            dc2 = [
                "python3", "../../scripts/convert_efield2efield.py", f"{sim2root_out}", "--add_noise_uVm=22", "--add_jitter_ns=5", "--calibration_smearing_sigma=0.075", "--target_duration_us=4.096","--target_sampling_rate_mhz=500"
            ]
                
            subprocess.run(dc2, check=True)
            print(f"Created DC2 efield files in {sim2root_out}.")
            print("********************************")
            print("* - * - * - * - * - * - * - * - * - *")
            print(f"Finished converting files in {options.directory}")
            print("********************************")
                    
    else:
        sys.exit("Please specify a directory containing Coreas simulations.")
#    sys.exit("Completed")
    
