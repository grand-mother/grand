# convert_efield2voltage.py

Calculation of DU response in microVolt for first event in Efield input file.

(Efield input data file in GRANDROOT format. **Note that other files need to be present too, ie TRun, TShower and TEfield**)

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


## help

```bash
# convert_efield2voltage.py -h
usage: convert_efield2voltage.py [-h] [--no_noise] [--no_rf_chain] -o OUT_FILE [--verbose {debug,info,warning,error,critical}]
                                 [--seed SEED] [--lst LST] [--padding_factor PADDING_FACTOR]
                                 [--target_duration_us TARGET_DURATION_US]
                                 [--target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ]
                                 file

Calculation of DU response in microvolts for first event in Efield input file.

positional arguments:
  file                  Efield input data file in GRANDROOT format (note that other files need to be present too, ie TRun, TShower and TEfield).

optional arguments:
  -h, --help            show this help message and exit
  --no_noise            don't add galactic noise.
  --no_rf_chain         don't add RF chain.
  -o OUT_FILE, --out_file OUT_FILE
                        output file in GRANDROOT format. If the file exists it is overwritten.
  --verbose {debug,info,warning,error,critical}
                        logger verbosity.
  --seed SEED           Fix the random seed to reproduce same galactic noise, must be positive integer
  --lst LST             lst for Local Sideral Time, galactic noise is variable with LST and maximal for 18h for the EW arm.
  --padding_factor PADDING_FACTOR
                        Increase size of signal with zero padding, with 1.2 the size is increased of 20%.
  --target_duration_us TARGET_DURATION_US
                        Adjust (and override) padding factor in order to get a signal of the given duration, in us
  --target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ
                        Target sampling rate of the data in Mhz

```


## Example

```bash
python ./convert_efield2voltage.py --seed 1234 --target_sampling_rate_mhz=500 --target_duration_us=4.096 ./TEfield_13_L0_GP300_13790.root -o ./Prueba-no_noise.root --no_noise --verbose=info
02:21:04.335  INFO [grand.manage_log 187] create handler for root logger: ['grand']
02:21:04.335  INFO [grand.scripts.convert_efield2voltage 141] 
02:21:04.335  INFO [grand.scripts.convert_efield2voltage 141] ===========> Begin at 2024-01-10T02:21:04Z <===========
02:21:04.335  INFO [grand.scripts.convert_efield2voltage 141] 
02:21:04.335  INFO [grand.scripts.convert_efield2voltage 141] 
02:21:04.335  INFO [grand.scripts.convert_efield2voltage 144] seed used for random number generator is 1234.
02:21:04.942  INFO [grand.sim.detector.rf_chain 473] vga gain: 20 dB
02:21:04.948  INFO [grand.sim.detector.antenna_model 103] Loading GP300 antenna model ...
02:21:04.948  INFO [grand.sim.detector.antenna_model 71] Using /home/mjtueros/TrabajoTemporario/docker/grand/data/detector/Light_GP300Antenna_EWarm_leff.npz
02:21:05.050  INFO [grand.sim.detector.antenna_model 71] Using /home/mjtueros/TrabajoTemporario/docker/grand/data/detector/Light_GP300Antenna_SNarm_leff.npz
02:21:05.151  INFO [grand.sim.detector.antenna_model 71] Using /home/mjtueros/TrabajoTemporario/docker/grand/data/detector/Light_GP300Antenna_Zarm_leff.npz
02:21:05.253  INFO [grand.sim.efield2voltage 110] Running on event_number: 13790, run_number: 13
02:21:05.341  INFO [grand.geo.turtle 222] Map constructor add map /home/mjtueros/TrabajoTemporario/docker/grand/data/egm96.png in cache memory 
02:21:05.344  INFO [grand.sim.shower.gen_shower 110] Site origin [lat, long, height]: [[  40.98455811]
02:21:05.344  INFO [grand.sim.shower.gen_shower 110]  [  93.95224762]
02:21:05.344  INFO [grand.sim.shower.gen_shower 110]  [1200.        ]]
02:21:05.344  INFO [grand.sim.shower.gen_shower 112] xmax in shower coordinate: [ 43622.63 -51987.43  13927.77]
02:21:05.345  INFO [grand.sim.efield2voltage 131] shower origin in Geodetic: [  40.98456   93.95225 1200.     ]
02:21:05.351  INFO [grand.sim.efield2voltage 246] shape du_nanoseconds and t_samples =  (44, 1), (44, 1671)
padding_factor 4.902453620586475 sig_size 1671 (8192.5) fast size 8192 freqs_mhz size 4097
02:21:05.351  INFO [grand.sim.efield2voltage 162] Electric field lenght is 1671 samples at 2000.0, spanning 0.8355 us.
02:21:05.351  INFO [grand.sim.efield2voltage 163] With a padding factor of 4.902453620586475 we will take it to 8192 samples, spanning 4.096 us.
02:21:05.352  INFO [grand.sim.efield2voltage 164] However, optimal number of frequency bins to do a fast fft is 4097 giving traces of 8192 samples.
02:21:05.352  INFO [grand.sim.efield2voltage 165] With this we will obtain traces spanning 4.096 us, that we will then truncate if needed to get the requested trace duration.
02:21:05.369  INFO [grand.sim.efield2voltage 309] ==============>  Processing DU with id: 46
.
.
.
02:21:05.775  INFO [grand.sim.efield2voltage 309] ==============>  Processing DU with id: 276
02:21:05.818  INFO [grand.sim.efield2voltage 290] resampling the voltage from 2000.0 to 500.0 MHz, new trace lenght is 2048 samples
02:21:05.821  INFO [grand.sim.efield2voltage 551] save result in ./Prueba-no_noise.root

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
