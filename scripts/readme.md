# convert_efield2voltage.py

Calculation of DU response in volt for first event in Efield input file.

## help

```bash
# convert_efield2voltage.py -h
usage: convert_efield2voltage.py [-h] [--no_noise] [--no_rf_chain] -o OUT_FILE [--verbose {debug,info,warning,error,critical}]
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
grand/scripts# convert_efield2voltage.py --lst 18 --seed 0 --verbose info ../data/test_efield.root  -o test_voltage.root 
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


# plot_noise.py

Information and plot galactic noise.

## Help

```bash
grand/scripts# plot_noise.py -h
usage: plot_noise.py [-h] [--lst {int value}] [--savefig]

Information and plot parameters of galactic noise

optional arguments:
  -h, --help            show this help message and exit
  --lst {integer value}
                        plot parameters of the galactic noise for the given LST. Default value is 18.
  --savefig             give this option if you like to save plots. Default is False.
```

## Example

```bash
plot_noise.py --lst 10 --savefig
```

# plot_rf_chain.py

Plot parameters of RF chain components for version 1.

## Help

```bash
grand/scripts# plot_noise.py -h
usage: plot_rf_chain.py [-h] [{lna, balun_after_lna, cable, vga, balun_before_adc, rf_chain}]

Information and plot parameters of various components of the RF chain (version 1).

optional arguments:
  -h, --help            show this help message and exit
  plot_option           {lna, balun_after_lna, vga, cable, balun_before_adc, rf_chain}
                        plot parameters of various components of the RF chain. plot_option allows you to choose component.
```

## Example

```bash
python3 plot_rf_chain.py lna
```

# plot_Vout_AT_Device.py

Plot Voltage output and Voltage ratios at individual RF chain elements.

## Help

```bash
grand/scripts# plot_Vout_AT_Device.py -h
usage: plot_Vout_AT_Device.py [-h] [{Vin_balun1, Vout_balun1, Vout_match_net, Vout_lna, Vout_cable_connector, Vout_VGA, Vout_tot, Vratio_Balun1, Vratio_match_net, Vratio_lna, Vratio_cable_connector, Vratio_vga, Vratio_adc}]

Information and plot parameters of various elements of the RF chain (version 2).

optional arguments:
  -h, --help            show this help message and exit
  plot_option           {Vin_balun1, Vout_balun1, Vout_match_net, Vout_lna, Vout_cable_connector, Vout_VGA, Vout_tot, Vratio_Balun1, Vratio_match_net, Vratio_lna, Vratio_cable_connector, Vratio_vga, Vratio_adc}
                        plot parameters of various components of the RF chain. plot_option allows you to choose component.
```

## Example

```bash
python3 plot_Vout_AT_Device.py Vout_lna
python3 plot_Vout_AT_Device.py Vratio_lna
```
