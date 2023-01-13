# grand_simu_du.py

Calculation of DU response in volt for first event in Efield input file.

## help

```bash
# grand_simu_du.py -h
usage: grand_simu_du.py [-h] [--no_noise] -o OUT_FILE [--verbose {debug,info,warning,error,critical}]
                        [--seed SEED] [--lst LST]
                        file

Calculation of DU response in volt for first event in Efield input file.

positional arguments:
  file                  Efield input data file in GRANDROOT format.

optional arguments:
  -h, --help            show this help message and exit
  --no_noise            don't add galactic noise.
  -o OUT_FILE, --out_file OUT_FILE
                        output file in GRANDROOT format. If the file exists it is overwritten.
  --verbose {debug,info,warning,error,critical}
                        logger verbosity.
  --seed SEED           Fix the random seed to reproduce same noise, must be positive integer
  --lst LST             lst for Local Sideral Time, galactic noise is variable with LST and maximal for 18h.

```


## Example

```bash
/home/dc1# grand_simu_du.py --lst 10 --seed -10 --verbose info Coarse2_xmax_add.root  -o c2_test.root 
10:26:08.446  INFO [grand.manage_log 189] create handler for root logger: ['grand']
10:26:08.446  INFO [grand.scripts.grand_simu_du 71] 
10:26:08.446  INFO [grand.scripts.grand_simu_du 71] ===========> Begin at 2022-12-09T10:26:08Z <===========
10:26:08.446  INFO [grand.scripts.grand_simu_du 71] 
10:26:08.446  INFO [grand.scripts.grand_simu_du 71] 
10:26:08.556  INFO [grand.io.root_files 200] Events  in file Coarse2_xmax_add.root
10:26:08.758  INFO [grand.io.root_files 108] load event: 1 of run  0
10:26:08.780  INFO [grand.io.root_files 115] resize numpy array trace to (96, 3, 999)
10:26:08.798  INFO [grand.io.root_files 219] load tt_shower: 1 of run  0
10:26:08.802  INFO [grand.simu.du.model_ant_du 23] Load model of antenna GP300
10:26:08.802  INFO [grand.io.file_leff 82] Loading tabulated antenna model from /home/dc1/grand/data/model/detector/GP300Antenna_EWarm_leff.npy:/
10
...
0:26:19.907  INFO [grand.simu.master_simu 119] save result in c2_test.root
10:26:19.915 WARNING [grand.io.root_trees 263] No valid teventvoltage TTree in the file c2_test.root. Creating a new one.
10:26:20.169  INFO [grand.scripts.grand_simu_du 82] 
10:26:20.169  INFO [grand.scripts.grand_simu_du 82] 
10:26:20.169  INFO [grand.scripts.grand_simu_du 82] ===========> End at 2022-12-09T10:26:20Z <===========
10:26:20.169  INFO [grand.scripts.grand_simu_du 82] Duration (h:m:s): 0:00:11.722876
```


# grand_ioroot.py

Information and plot event/traces for ROOT file

## Help

```bash
/home/dc1# grand_ioroot.py -h
usage: grand_ioroot.py [-h] [--ttree {efield,voltage}] [-f] [--time_val] [-t TRACE] [--trace_image] [--list_du] [--list_ttree] [--dump DUMP] [-i] file

Information and plot event/traces for ROOT file

positional arguments:
  file                  path and name of ROOT file GRAND

optional arguments:
  -h, --help            show this help message and exit
  --ttree {efield,voltage}
                        Define the event TTree to read in file. Default is efield
  -f, --footprint       interactive plot (double click) of footprint, max value for each DU
  --time_val            interactive plot, value of each DU at time t defined by a slider
  -t TRACE, --trace TRACE
                        plot trace x,y,z and power spectrum of detector unit (DU)
  --trace_image         interactive image plot (double click) of norm of traces
  --list_du             list of identifier of DU
  --list_ttree          list of TTree present in file
  --dump DUMP           dump trace of DU
  -i, --info            some information about the contents of the file
```

## Example

```bash
grand_ioroot.py --ttree voltage --footprint c2_test.root
```
