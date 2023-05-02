# grand_simu_du.py

Calculation of DU response in volt for first event in Efield input file.

## help

```bash
# grand_simu_du.py -h
usage: grand_simu_du.py [-h] [--no_noise] [--no_rf_chain] -o OUT_FILE [--verbose {debug,info,warning,error,critical}]
                        [--seed SEED] [--lst LST] [--padding_factor FLOAT]
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
/home/dc1# grand_simu_du.py --lst 18 --seed 0 --verbose info test_efield.root  -o test_voltage_jpt.root 
05:42:43.526  INFO [grand.manage_log 187] create handler for root logger: ['grand']
05:42:43.527  INFO [grand.scripts.grand_simu_du 84] 
05:42:43.527  INFO [grand.scripts.grand_simu_du 84] ===========> Begin at 2023-04-07T05:42:43Z <===========
05:42:43.527  INFO [grand.scripts.grand_simu_du 84] 
05:42:43.527  INFO [grand.scripts.grand_simu_du 84] 
05:42:43.527  INFO [grand.scripts.grand_simu_du 90] seed used for random number generator is 0.
05:42:45.606  INFO [grand.simu.du.antenna_model 104] Loading GP300 antenna model ...
05:42:45.607  INFO [grand.simu.du.antenna_model 72] Loading /home/grand/data/detector/Light_GP300Antenna_EWarm_leff.npz
05:42:49.067  INFO [grand.simu.du.antenna_model 72] Loading /home/grand/data/detector/Light_GP300Antenna_SNarm_leff.npz
05:42:53.614  INFO [grand.simu.du.antenna_model 72] Loading /home/grand/data/detector/Light_GP300Antenna_Zarm_leff.npz
05:42:58.785  INFO [grand.simu.efield2voltage 79] Running on event-number: 1, run-number: 0
...
05:43:29.537  INFO [grand.simu.efield2voltage 300] save result in test_voltage_jpt.root
05:43:29.563 WARNING [grand.io.root_trees 391] No valid tvoltage TTree in the file test_voltage_jpt.root. Creating a new one.
05:43:31.142  INFO [grand.scripts.grand_simu_du 102] 
05:43:31.142  INFO [grand.scripts.grand_simu_du 102] 
05:43:31.142  INFO [grand.scripts.grand_simu_du 102] ===========> End at 2023-04-07T05:43:31Z <===========
05:43:31.142  INFO [grand.scripts.grand_simu_du 102] Duration (h:m:s): 0:00:47.615486
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