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

*step 1:* convert air shower simulation to the _RawRoot_ file format\
For CoREAS follow the instructions under *1.a)*, for ZHAireS follow the instructions under *1.b)*.

*step 2:* convert your newly created _RawRoot_ file to the _GRANDRoot_ file format


## 1.a) CoREASRawRoot/CoreasToRawROOT.py
Here we have the scripts to produce _RawRoot_ files from CoREAS simulations.

To run the script on the provided example event just go to the CoREASRawRoot/ directory and do

python3 CoreasToRawROOT.py 000004/

Alternatively, you can specify any other directory including a full CoREAS simulation.

[disclaimer:] only tested for Corsika7

*The output file will automatically be named "Coreas_" + < EventName > + ".root"*

## 1.b) ZHAireSRawRoot/ZHAireSRawToRawROOT.py
Here we have the scripts to produce _RawRoot_ files from ZHAireS simulations.

To run the script on the provided example event just go to the ZHAireSRawRoot/ directory and do

python3 ZHAireSRawToRawROOT.py ./GP10_192745211400_SD075V standard 0 1  GP10_192745211400_SD075V.root

## 2) Common/sim2root.py
Inside Common/ you can find the final converter, sim2root.py

As input you need to give the ROOT file containing GRANDRaw data TTrees, as created with CoreasToRawROOT or ZHAireSRawToRawROOT.
