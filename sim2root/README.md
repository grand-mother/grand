# Welcome to Sim2Root
originally developed as GrandRawRoot in https://github.com/jelenakhlr/GrandRawRoot

## authors
Lech Piotrowski, University of Warsaw - @lwpiotr (grandroot)\
Matias Tueros, Instituto de Fisica La Plata - @mtueros (Zhaires)\
Jelena Köhler, Kalrsruhe Institute of Technology  - @jelenakhlr (Coreas)

## motivation
In GRAND we want to be able to simulate air showers using both Zhaires and Coreas.
The _RawRoot_ file format is a step towards common output between CoREAS and ZHAireS. Its based on the _GRANDRoot_ file format.
The idea is for _RawRoot_ to be the starting point for the generation of _GRANDRoot_ files.

Common files are stored in the Common/ directory, Zharies specifics can be found in ZHAireSRawRoot/ and Coreas specifics in CoREASRawRoot/. 

# How to run Sim2Root
If you want to convert an air shower simulation to the _GRANDRoot_ file format, follow these steps:

*step 1:* convert air shower simulation to the _rawroot_ file format\
For CoREAS follow the instructions under *1.a)*, for ZHAireS follow the instructions under *1.b)*.

*step 2:* convert your newly created _RawRoot_ file to the _GRANDRoot_ file format


## 1.a) CoREASRawRoot/CoreasToRawROOT.py
Here we have the scripts to produce _RawRoot_ files from CoREAS simulations.

To run the script on the provided example event just go to the CoREASRawRoot/ directory and do

`python3 CoreasToRawROOT.py GP13_000004/`

Alternatively, you can specify any other directory including a full CoREAS simulation.

[disclaimer:] only tested for Corsika7

*The output file will automatically be named "Coreas_" + < EventName > + ".root"*

## 1.b) ZHAireSRawRoot/ZHAireSRawToRawROOT.py
Here we have the scripts to produce _RawRoot_ files from ZHAireS simulations.

To run the script on the provided example event just go to the `ZHAireSRawRoot/` directory and either do:

`python3 ZHAireSRawToRawROOT.py InputDirectory Mode RunID EventID OutputFilename` (only standard mode is available)

 i.e.

`python3 ZHAireSRawToRawROOT.py  ./GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618 standard 1 1618  GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618.rawroot`
`python3 ZHAireSRawToRawROOT.py  ./GP300_Xi_Sib_Proton_3.87_79.4_310.0_13790 standard 1 13790  GP300_Xi_Sib_Proton_3.87_79.4_310.0_13790.rawroot`

or alternativelly

`python3 ZHAireSRawToRawROOT.py InputDirectory`

 i.e.

`python3 ZHAireSRawToRawROOT.py GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618`

To let ZHAIRESRawToRawROOT make the choices for you. This is equivalent to running with RunID="SuitYourself", EventID="LookForIt", OutputFileName="GRANDConvention"	

Look inside of the script for more inforation

## 2) Common/sim2root.py
Inside `Common/` you can find the final converter, `sim2root.py`

As input you need to give the ROOT file containing `rawroot data TTrees`, as created with `CoreasToRawROOT` or `ZHAireSRawToRawROOT`.

i.e.

`python3 python  ../grand/sim2root/Common/sim2root.py <your path>*/*.rawroot -d 20221026 -t 180000 -e DC2Alpha`

additional options are available on command line, see sim2root --help for more information

## 3) Simulation Pipe example (in the Common directory)

The example shows how to use the two example rawroot file given in /grand/sim2root/ZHAireSRawRoot/

You can use the rawroot file of your liking. 

python  ../grand/sim2root/Common/RunSimPipe InputDirectory Extra

where "Extra" is the additional field that you can use in the directory name to make your directory cooler.

Note that you have to edit the fields inside the script to the appropiate paths if you are running in a different directory

For example, if you run here (and the two zhaires example rawroot files are there!)

python RunSimPipe.py ../ZHAireSRawRoot ZHAireS

This will run
1) rawroot 2 grandroot
python ./sim2root.py ../ZHAireSRawRoot/ -e ZHAireS
2) compute voltage
python ../../scripts/convert_efield2voltage.py sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/ --seed 1234 --target_duration_us=4.096 --add_jitter_ns 5 --calibration_smearing_sigma 0.075 --verbose=info -o sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/voltage_1618-13790_L0_0000.root
3) compute adc
python ../../scripts/convert_voltage2adc.py sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/
4) compute DC2 efield
python ../../scripts/convert_efield2efield.py sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/  --add_noise_uVm 22 --add_jitter_ns 5 --calibration_smearing_sigma 0.075 --target_duration_us 4.096 --target_sampling_rate_mhz 500

## 3) Simulation Pipe WITHOUT NOISE example (in the Common directory)

The example shows how to use the example rawroot file given in /grand/sim2root/CoREASRawRoot/

You can use the rawroot file of your liking. 

python  ../grand/sim2root/Common/RunSimPipeNoJitter.py InputDirectory Extra

where "Extra" is the additional field that you can use in the directory name to make your directory cooler.

Note that you have to edit the fields inside the script to the appropiate paths if you are running in a different directory

For example, if you run here (and the two zhaires example rawroot files are there!)

python RunSimPipeNoJitter.py ../CoREASRawRoot CoREAS-NJ

This will run
1) rawroot 2 grandroot
python ./sim2root.py ../CoREASRawRoot/ -e CoREAS-NJ
2) compute voltage
python ../../scripts/convert_efield2voltage.py sim_Dunhuang_20170401_000000_RUN1_CD_CoREAS-NJ_0000/ --seed 1234 --target_duration_us=4.096 --verbose=info --no_noise -o sim_Dunhuang_20170401_000000_RUN1_CD_CoREAS-NJ_0000/voltage_4100-4100_L0_0000.root
3) compute adc
python ../../scripts/convert_voltage2adc.py sim_Dunhuang_20170401_000000_RUN1_CD_CoREAS-NJ_0000/
4) compute DC2 efield
python ../../scripts/convert_efield2efield.py sim_Dunhuang_20170401_000000_RUN1_CD_CoREAS-NJ_0000/  --target_duration_us 4.096 --target_sampling_rate_mhz 500




# 4) Look at the results (in the Common directory)
python IllustrateSimPipe.py ./sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000

additional options are available on command line, doing IllustrateSimPipe.py -h



