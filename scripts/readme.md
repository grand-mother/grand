# scripts/

Command-line tools over the library. Each takes `-h`; the help below is the
output of `-h` itself, so it is current as of the commit that wrote this file.

| Script | What it does |
|---|---|
| `convert_efield2voltage.py` | E-field to voltage at each detection unit (antenna response, galactic noise, RF chain) |
| `convert_voltage2adc.py` | Voltage to ADC counts, optionally with measured noise and the T1 trigger |
| `convert_efield2efield.py` | A "hardware-like" e-field: filtered, resampled, with noise, jitter and calibration smearing |
| `T1_trigger_offline.py` | The offline T1 trigger on an ADC file; lists the triggered entries |
| `extract_events.py` | Copies a list of events from several folders into one |
| `open_grand_file.py`, `open_grand_directory.py`, `open_grand_analysis_prompt.py` | An interactive shell with a file, a folder or an `EventList` loaded |
| `plot_noise.py`, `plot_rf_chain.py`, `plot_Vout_AT_Device.py`, `plot_tmax_vmax.py` | Plots of the noise model, the RF chain, the voltage at each device, and trace peaks |
| `extract_rf_chain.py` | Writes the combined RF-chain transfer function to `TF_RF_Chain.npy` (no options) |
| `get_version.py` | Prints the data-format version (no options; used by CI) |

The conversion scripts take the folder `sim2root.py` wrote, not a single file,
and write into it unless `-od` names another; leave out `-o` and each names
its output so the next step finds it:

```bash
python scripts/convert_efield2voltage.py <sim_folder> --seed 1234
python scripts/convert_voltage2adc.py <sim_folder>
```

# convert_efield2voltage.py

Computes the voltage at each detection unit for every event in a simulation
folder (it needs the run, shower and e-field trees there).

## authors
Ramesh ?    - @rkoirala\
Jean-Marc ? - @jean-marc\
(sorry if there are other authors im not aware of, pelase add yourself here!)
Matias Tueros, Instituto de Fisica La Plata - @mtueros (resampling and time extension)

## How this works (broadly speaking, but so its easier to understand):

1) Computes the closest magic number of frequencies  (a multiple of 2,3 or 5) needed in the fft to give enough sampling points in the irfft to get to the TARGET_DURATION_US or to comply with requested PADDING_FACTOR.\
Know that rfft is several hundred times faster if you use a multiple of 2,3 or 5, so this is very much worth it. However, this means that even if you set padding_factor=1 or dont specify a TARGET_DURATION_US (so the same duration as the source efield is used), it is still posible to have some padding added in order to have a multiple of 2,3 or 5.\
**Note that if the padding factor is less than 1, or the TARGET_DURATION_US is less than the actual duration of the efield trace, the trace will be cropped (feature not tested)**. See [scipy.fft.rfft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfft.html).
2) this magic number of frequencies is passed to the galactic noise routine (that will interpolate from its internal parameterization). 
3) this magic number of frequencies is passed to the RF chain respones routine (that will interpolate from its internal parameterization). 
4) Computes the rfft of the efield, with said number of frequencies. 
5) If requested, adds the galactic noise to the rfft.
6) If requested, convolves the rf chain to the rfft.
7) Perform the ifft to get the trace in time domain (re-sampled if requested, using fourier interpolation, See [scipy.fft.ifft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifft.html)).
8) trim the output to the originally requested TARGET_DURATION_US or PADDING FACTOR.


## CAVEATS:
0) **When the number of points in efield is more than the number of frequencies, the efield will be cropped. When it is less, efield will be zero-padded.** 
1) **rfft assumes signals are periodic**. This means efield must go towards 0 at the start and the end of the trace, if not *spectral leakage will occur*.
2) This is also true even if you zero-pad the efield. The sudden jump from nonzero to zero will create false frequency content. *If your efield trace is not going down to 0, considering applying a window function (i.e. Hanning) to the efield and renormalize its amplitude if needed to get the correct fft*. 
3) **When you downsample you will reduce the bandwidth, and aliasing could ocurr**. Formaly, the signal should be low-pass filtered before the downsampling. In our use case, we go usually from 2000Mhz Efield to 500Mhz Vrf sampling rate,  this means that bandwidth goes from 1000Mhz to 250Mhz. Our RF chain already acts as a filter (the transfer function is 0 above 250Mhz) so if you apply the RF chain, we are safe. If you are not appling the rf chain, aliasing will ocurr. 
4) **In the current state of affairs, the antenna response is non-causal.** 
5) **All this caveats can result in weird behaviours** specially in the borders of the trace (like ringing before the start of the peak, and at the start and end of the trace).

## Help

```
$ python convert_efield2voltage.py -h
usage: convert_efield2voltage.py [-h] [--no_noise] [--no_rf_chain]
                                 [--rf_chain_nut] [--rf_chain_gaa]
                                 [-o OUT_FILE] [-od OUT_DIRECTORY]
                                 [--verbose {debug,info,warning,error,critical}]
                                 [--seed SEED] [--lst LST]
                                 [--padding_factor PADDING_FACTOR]
                                 [--du_type DU_TYPE]
                                 [--target_duration_us TARGET_DURATION_US]
                                 [--target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ]
                                 [--add_jitter_ns ADD_JITTER_NS]
                                 [--calibration_smearing_sigma CALIBRATION_SMEARING_SIGMA]
                                 directory

Calculation of DU response in volt for first event in Efield input file.

positional arguments:
  directory             Simulation output data directory in GRANDROOT format.

options:
  -h, --help            show this help message and exit
  --no_noise            don't add galactic noise.
  --no_rf_chain         don't add RF chain.
  --rf_chain_nut        add RF chain in antenna nut
  --rf_chain_gaa        add RF chain for G@Auger setup
  -o OUT_FILE, --out_file OUT_FILE
                        output file in GRANDROOT format. If the file exists it
                        is overwritten.
  -od OUT_DIRECTORY, --out_directory OUT_DIRECTORY
                        output directory in GRANDROOT format. If not given, is
                        it the same as input directory
  --verbose {debug,info,warning,error,critical}
                        logger verbosity.
  --seed SEED           Fix the random seed to reproduce same galactic noise,
                        must be positive integer
  --lst LST             lst for Local Sideral Time, galactic noise is variable
                        with LST and maximal for 18h for the EW arm.
  --padding_factor PADDING_FACTOR
                        Increase size of signal with zero padding, with 1.2
                        the size is increased of 20%.
  --du_type DU_TYPE     Choose between 4 different antenna models, GP300
                        -using hfss simulations, GP300_nec -using nec
                        simulations, GP300_mat -using matlab simulations,
                        Horizon
  --target_duration_us TARGET_DURATION_US
                        Adujust (and override) padding factor in order to get
                        a signal of the given duration, in us
  --target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ
                        Not supported (the voltage file cannot record a new
                        rate, issue #229): resample with
                        convert_efield2efield.py instead
  --add_jitter_ns ADD_JITTER_NS
                        level of gaussian jitter (ns) to add to the trigger
                        times
  --calibration_smearing_sigma CALIBRATION_SMEARING_SIGMA
                        Smear the stations amplitude calibrations with a
                        gaussian centered in 1 and this input sigma
```

# convert_voltage2adc.py

```
$ python convert_voltage2adc.py -h
usage: convert_voltage2adc.py [-h] [-o OUT_FILE] [--add_noise_from NOISE_DIR]
                              [--target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ]
                              [-s SEED]
                              [-v {debug,info,warning,error,critical}]
                              [--t1_trigger] [--t1_param KEY=VALUE]
                              in_dir

Conversion of voltage at ADC input to digitized ADC counts. Includes option to
add measured noise.

positional arguments:
  in_dir                Directory holding the sim2root output:
                        voltage_*_L0_*.root and the run trees. A voltage file
                        in it may be given instead (#180).

options:
  -h, --help            show this help message and exit
  -o OUT_FILE, --out_file OUT_FILE
                        Path to output file in GrandRoot format (TADC). If the
                        file exists it is overwritten.
  --add_noise_from NOISE_DIR
                        Path to directory containing files with measured noise
                        in GrandRoot format (TADC). Default adds no noise.
  --target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ
                        Target sampling rate of the data in Mhz (not
                        implemented, currently hard coded to 500Mhz)
  -s SEED, --seed SEED  Fix the random seed for selection of measured noise
                        traces. Must be positive integer. Default yields
                        irreproducible RNG.
  -v {debug,info,warning,error,critical}, --verbose {debug,info,warning,error,critical}
                        Logger verbosity.
  --t1_trigger          Apply the offline DAQ-style T1 trigger to every DU and
                        set trigger_flag (1 = passed, 0 = not). Default off,
                        leaving the output unchanged.
  --t1_param KEY=VALUE  Override a T1 trigger parameter (repeatable), e.g.
                        --t1_param th1=120. Defaults: {'t_quiet': 512,
                        't_period': 512, 't_sepmax': 10, 'nc_min': 2,
                        'nc_max': 8, 'q_min': 0, 'q_max': 255, 'th1': 100,
                        'th2': 50, 't_pretrig': 960, 't_overlap': 64,
                        't_posttrig': 1024}. Only used with --t1_trigger.
```

# convert_efield2efield.py

```
$ python convert_efield2efield.py -h
usage: convert_efield2efield.py [-h] [--no_filter] [-o OUT_FILE]
                                [-od OUT_DIRECTORY]
                                [--verbose {debug,info,warning,error,critical}]
                                [--seed SEED] [--add_noise_uVm ADD_NOISE_UVM]
                                [--add_jitter_ns ADD_JITTER_NS]
                                [--calibration_smearing_sigma CALIBRATION_SMEARING_SIGMA]
                                [--target_duration_us TARGET_DURATION_US]
                                [--target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ]
                                directory

Calculation of Hardware-like Efield input file.

positional arguments:
  directory             Simulation output data directory in GRANDROOT format.

options:
  -h, --help            show this help message and exit
  --no_filter           remove the filter on the GRAND bandwidth. (50-200Mhz,
                        band-pass elliptic causal filter)
  -o OUT_FILE, --out_file OUT_FILE
                        output file in GRANDROOT format. If the file exists it
                        is overwritten.
  -od OUT_DIRECTORY, --out_directory OUT_DIRECTORY
                        output directory in GRANDROOT format. If not given, is
                        it the same as input directory
  --verbose {debug,info,warning,error,critical}
                        logger verbosity.
  --seed SEED           Fix the random seed to reproduce same galactic noise,
                        must be positive integer
  --add_noise_uVm ADD_NOISE_UVM
                        level of gaussian noise (uv/m) to add to the trace
                        before filtering
  --add_jitter_ns ADD_JITTER_NS
                        level of gaussian jitter (ns) to add to the trigger
                        times
  --calibration_smearing_sigma CALIBRATION_SMEARING_SIGMA
                        Smear the stations amplitude calibrations with a
                        gaussian centered in 1 and this input sigma
  --target_duration_us TARGET_DURATION_US
                        Adjust (and override) padding factor in order to get a
                        signal of the given duration, in us
  --target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ
                        Target sampling rate of the data in Mhz
```

# T1_trigger_offline.py

```
$ python T1_trigger_offline.py -h
usage: T1_trigger_offline.py [-h] [-o OUT] [--t1_param KEY=VALUE] adc_file

Offline T1 trigger on the ADC traces (TADC) of a GrandRoot file.

positional arguments:
  adc_file              ADC file (TADC) to scan

options:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     text file for the list of triggered entries (default:
                        <adc_file>.trigger.txt in the current folder)
  --t1_param KEY=VALUE  override a T1 parameter (repeatable), e.g. --t1_param
                        th1=120. Defaults: {'t_quiet': 512, 't_period': 512,
                        't_sepmax': 10, 'nc_min': 2, 'nc_max': 8, 'q_min': 0,
                        'q_max': 255, 'th1': 100, 'th2': 50, 't_pretrig': 960,
                        't_overlap': 64, 't_posttrig': 1024}
```

# extract_events.py

```
$ python extract_events.py -h
usage: extract_events.py [-h] [-c COMMENT] [-ow]
                         <source_events_list_file> <dirname>

Extract events from provided directories and store in a target directory.

positional arguments:
  <source_events_list_file>
                        A file with a list of source events to extract. The
                        format is dir_path,run_num,event_num
  <dirname>             The target directory to store the extracted events in

options:
  -h, --help            show this help message and exit
  -c COMMENT, --comment COMMENT
                        Comment stored in the metadata of every tree written
  -ow, --overwrite      Replace the GRAND files already in the target
                        directory (other files are kept)
```

# open_grand_file.py

```
$ python open_grand_file.py -h
usage: open_grand_file.py [-h] [-p] [-s] <filename>

Open a GRAND file in an IPython or Python shell.

positional arguments:
  <filename>  The GRAND ROOT filename to load

options:
  -h, --help  show this help message and exit
  -p          Use Python instead of IPython
  -s          Do not print any initial output
```

# open_grand_directory.py

```
$ python open_grand_directory.py -h
usage: open_grand_directory.py [-h] [-p] [-s] [-nv] <dirname>

Open a GRAND directory in an IPython or Python shell.

positional arguments:
  <dirname>   The GRAND ROOT directory to load

options:
  -h, --help  show this help message and exit
  -p          Use Python instead of IPython
  -s          Do not print any initial output
  -nv         Do not print verbose output
```

# open_grand_analysis_prompt.py

```
$ python open_grand_analysis_prompt.py -h
usage: open_grand_analysis_prompt.py [-h] [-p] [-s] [-nv] [-trv] <dirname>

Open a GRAND directory as an EventList in an IPython or Python shell.

positional arguments:
  <dirname>             The GRAND ROOT directory to load

options:
  -h, --help            show this help message and exit
  -p                    Use Python instead of IPython
  -s                    Do not print any initial output
  -nv                   Do not print verbose output
  -trv, --use_trawvoltage
                        Use TRawVoltage instead of TVoltage
```

# plot_noise.py

```
$ python plot_noise.py -h
usage: plot_noise.py [-h] [--savefig] [--du_type DU_TYPE] [--lst LST]

Plot function with command line arguments

options:
  -h, --help         show this help message and exit
  --savefig          Flag to save the figure
  --du_type DU_TYPE  Type of du
  --lst LST          LST info (defaults to 18)
```

# plot_rf_chain.py

```
$ python plot_rf_chain.py -h
usage: plot_rf_chain.py [-h] [--savefig] plot_option

Parser to select which noise quantity to plot. To Run: ./plot_rf_chain.py
<plot_option>. <plot_option>: matching_network, lna, balun_after_lna, vga,
cable, balun_before_adc, rf_chain example: ./plot_rf_chain.py lna --savefig

positional arguments:
  plot_option  what do you want to plot? example: lna.

options:
  -h, --help   show this help message and exit
  --savefig    don't add galactic noise.
```

# plot_Vout_AT_Device.py

```
$ python plot_Vout_AT_Device.py -h
usage: plot_Vout_AT_Device.py [-h] [--savefig] plot_option

Parser to select which quantity to plot. To Run: python3
plot_Vout_AT_Device.py <plot_option>. <plot_option>: Vin_balun1, Vout_balun1,
Vout_match_net, Vout_lna, Vout_cable_connector, Vout_VGA, Vout_tot,
Vratio_Balun1, Vratio_match_net, Vratio_lna, Vratio_cable_connector,
Vratio_vga, Vratio_adc example: python3 plot_Vout_AT_Device.py Vout_lna
--savefig

positional arguments:
  plot_option  what do you want to plot? example: Vout_lna.

options:
  -h, --help   show this help message and exit
  --savefig    don't add Voc.
```

# plot_tmax_vmax.py

Takes an e-field file and an event index as plain arguments (no `-h`):

```bash
python plot_tmax_vmax.py <efield.root> <event index>
```
