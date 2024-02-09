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

`python3 ZHAireSRawToRawROOT.py ./GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618 standard 0 1  GP300_1618.root`

or alternativelly

`python3 ZHAireSRawToRawROOT.py InputDirectory`

 i.e.

`python3 ZHAireSRawToRawROOT.py GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618`

To let ZHAIRESRawToRawROOT make the choices for you. This is equivalent to running with RunID="SuitYourself", EventID="LookForIt", OutputFileName="GRANDConvention"	

## 2) Common/sim2root.py
Inside `Common/` you can find the final converter, `sim2root.py`

As input you need to give the ROOT file containing `rawroot data TTrees`, as created with `CoreasToRawROOT` or `ZHAireSRawToRawROOT`.

i.e.

`python3 python  ../grand/sim2root/Common/sim2root.py <your path>*/*.rawroot -s Xiaodushan -d 20221026 -t 180000 -e DC2Alpha



additional options are available on command line, see sim2root --help for more information

# 3) Simulation Pipe example 

The example shows how to use the two example rawroot file given in /grand/sim2root/ZHAireSRawRoot/

You can use the rawroot file of your liking. Now sim2root supports multiple events on the command line, so you just can make

python  ../grand/sim2root/Common/sim2root.py */*.rawroot -s Xiaodushan -d 20221026 -t 180000 -e DC2Alpha


python ../grand/scripts/convert_efield2voltage.py --seed 1234 --target_sampling_rate_mhz=500 --target_duration_us=4.096 ./sim_Xiaodushan_20221026_180000_RUN0_CD_DC2Alpha_0000/efield_0-1_L0_0000.root -o ./OutputFile-no_rf_chain.root --no_rf_chain --verbose=info


see the –help of convert_efiedl2voltage to see how that works.
