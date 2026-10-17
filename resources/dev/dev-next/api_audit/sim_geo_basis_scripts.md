# API audit, step 3: `grand.sim`, `grand.geo`, `grand.basis` and the scripts

Same method as step 2 (`dataio_aoi.md`): each changed function or script was read as a diff, old and new, and classified; every behavior change came with an example, run on both versions, and the rows were reviewed against the outputs. The setup is in [#293](https://github.com/grand-mother/grand/issues/293).

## Summary

| | grand.sim | grand.geo | grand.basis | Command-line scripts |
|---|---|---|---|---|
| Changed functions or scripts | 72 | 32 | 21 | 22 |
| (1) Incompatible (A, W) | 52 | 17 | 9 | 16 |
| of which also fix a bug | 47 | 14 | 5 | 11 |
| (2) Improvements (C) | 67 | 28 | 15 | 15 |
| (3) Non-critical fixes (B2) | 45 | 12 | 5 | 14 |
| Serious fixes (B1) | 8 | 6 | 1 | 5 |
| No visible change | 2 | 3 | 4 | 0 |
| Examples run on both versions | 52 | 17 | 9 | 16 |

The changes that need a decision are in #326 to #332. #326 and #327 are bugs in dev-next.

## Codes

See `dataio_aoi.md`.

## grand.sim

| Function or script | Classes | What a caller sees on dev-next | Issue |
|---|---|---|---|
| `sim.detector.rf_chain.BalunAfterLNA.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). *Review:* The size check fires only when the data files differ from the downloaded manifest; the example faked one. | #328 |
| `sim.detector.rf_chain.BalunAfterLNA.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.BalunBeforeADC.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.BalunBeforeADC.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.Cable.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.Cable.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.GenericProcessingDU.__init__` | same | No visible change. |  |
| `sim.detector.rf_chain.GenericProcessingDU.set_out_freq_mhz` | A3, A4, B2, C1, C3 | Negative, NaN/inf, empty or 2-D frequencies raise ValueError (dev stored them); lists accepted; stored freqs_mhz always float64 (dev kept float32/int). | #328 |
| `sim.detector.rf_chain.LowNoiseAmplifier.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.LowNoiseAmplifier.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.MatchingNetwork.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.MatchingNetwork.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.RFChain.compute_for_freqs` | A3, B2, C1, C3, C4 | Negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted; impedance CSV checked against data manifest. | #328 |
| `sim.detector.rf_chain.RFChain.vout_f` | C1, C3 | Wrong voc_f shape raises ValueError instead of AssertionError; array-likes accepted; raises RuntimeError after the new release_arrays() (#284). |  |
| `sim.detector.rf_chain.RFChainNut.compute_for_freqs` | A3, B2, C1, C3, C4 | Negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted; impedance CSV checked against data manifest. | #328 |
| `sim.detector.rf_chain.RFChainNut.vout_f` | C1, C3 | Wrong voc_f shape raises ValueError instead of AssertionError; array-likes accepted; raises RuntimeError after the new release_arrays() (#284). |  |
| `sim.detector.rf_chain.RFChain_Balun1.compute_for_freqs` | A3, B2, C1, C3, C4 | Negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted; impedance CSV checked against data manifest. | #328 |
| `sim.detector.rf_chain.RFChain_Balun1.vout_f` | C1, C3 | Wrong voc_f shape raises ValueError instead of AssertionError; array-likes accepted; raises RuntimeError after the new release_arrays() (#284). |  |
| `sim.detector.rf_chain.RFChain_Cable_Connectors.compute_for_freqs` | A3, B2, C1, C3, C4 | Negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted; impedance CSV checked against data manifest. | #328 |
| `sim.detector.rf_chain.RFChain_Cable_Connectors.vout_f` | C1, C3 | Wrong voc_f shape raises ValueError instead of AssertionError; array-likes accepted; raises RuntimeError after the new release_arrays() (#284). |  |
| `sim.detector.rf_chain.RFChain_Match_net.compute_for_freqs` | A3, B2, C1, C3, C4 | Negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted; impedance CSV checked against data manifest. | #328 |
| `sim.detector.rf_chain.RFChain_Match_net.vout_f` | C1, C3 | Wrong voc_f shape raises ValueError instead of AssertionError; array-likes accepted; raises RuntimeError after the new release_arrays() (#284). |  |
| `sim.detector.rf_chain.RFChain_VGA.compute_for_freqs` | A3, B2, C1, C3, C4 | Negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted; impedance CSV checked against data manifest. | #328 |
| `sim.detector.rf_chain.RFChain_VGA.vout_f` | C1, C3 | Wrong voc_f shape raises ValueError instead of AssertionError; array-likes accepted; raises RuntimeError after the new release_arrays() (#284). |  |
| `sim.detector.rf_chain.RFChain_gaa.compute_for_freqs` | A3, B2, C1, C3, C4 | Negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted; impedance CSV checked against data manifest. | #328 |
| `sim.detector.rf_chain.RFChain_gaa.vout_f` | C1, C3 | Wrong voc_f shape raises ValueError instead of AssertionError; array-likes accepted; raises RuntimeError after the new release_arrays() (#284). |  |
| `sim.detector.rf_chain.RFChain_in_Balun1.compute_for_freqs` | A3, B2, C1, C3, C4 | Negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted; impedance CSV checked against data manifest. | #328 |
| `sim.detector.rf_chain.RFChain_in_Balun1.vout_f` | C1, C3 | Wrong voc_f shape raises ValueError instead of AssertionError; array-likes accepted; raises RuntimeError after the new release_arrays() (#284). |  |
| `sim.detector.rf_chain.Rfchain_elements.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.Rfchain_elements.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.Rfchain_elements_db.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.Rfchain_elements_db.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.Rfchain_elements_db_rad.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.Rfchain_elements_db_rad.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.Rfchain_elements_rad.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.Rfchain_elements_rad.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.VGAFilter.__init__` | A3, B2, C1 | Gain not in -5/0/5/20, or a bool, raises ValueError up front (dev: AssertionError, but gain=False worked); data file checked against the data-model manifest. | #328 |
| `sim.detector.rf_chain.VGAFilter.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.Zload.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.Zload.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.Zload_arb.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.Zload_arb.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.gaa_frontend0db.__init__` | A3, B2, C1 | Data file checked first: missing file gives FileNotFoundError naming the remedy; a size differing from the data-model manifest now raises ValueError (dev loaded it). | #328 |
| `sim.detector.rf_chain.gaa_frontend0db.compute_for_freqs` | A3, B2, C1, C3, C4 | Via set_out_freq_mhz: negative or NaN/inf frequencies raise ValueError (dev computed garbage); empty raises ValueError, not AssertionError; lists accepted. | #328 |
| `sim.detector.rf_chain.get_axis_filename` | A1, A3, B2, C1, C3, C5 | Disabled component, missing template or bad axis raise instead of printing and returning None; unknown component KeyError, not NameError; digit-string axes accepted; module globals components/csv_files removed. | #328 |
| `sim.detector.rf_chain.interpol_at_new_x` | C1, C3 | Empty a_x raises ValueError instead of AssertionError; lists accepted for a_x; scipy no longer imported. |  |
| `sim.detector.rf_chain.matmul` | C1 | Bad ABCD shapes raise ValueError naming both shapes instead of AssertionError (1-D A: ValueError, not IndexError). |  |
| `sim.detector.adc.ADC.downsample` | A4, B2, W, A3, C1 | Output length now rounded, not truncated (1003 samples 2000→500 MHz: 251, dev 250); warns on rates outside 1–1e5 MHz; array rates raise TypeError, non-positive ValueError. | #331 |
| `sim.detector.adc.ADC.process` | A3, B2, W, C1 | NaN/inf voltages now raise ValueError (dev returned -2^63 counts); saturation now logs a warning; values beyond int64 now saturate; asserts became TypeError/ValueError. | #328 |
| `sim.detector.antenna_model.AntennaModel.__init__` | A1, C1 | du_type='Horizon' removed: now ValueError listing GP300, GP300_nec, GP300_mat (dev sought Horizon .npy files absent from data/: FileNotFoundError); unknown types ValueError, not AttributeError. | #328 |
| `sim.detector.antenna_model.tabulated_antenna_model` | C1 | A missing table file now raises FileNotFoundError naming the data model and a remedy; if a data manifest is installed, a size mismatch raises ValueError. |  |
| `sim.detector.process_ant.AntennaProcessing.compute_voltage` | C1, C5 | pos or frame None now raises MissingFrameError instead of TypeError; non-finite pos/frame no longer printed to stdout, the error message carries them. |  |
| `sim.detector.process_ant.AntennaProcessing.effective_length` | A4, B1 | Sources below the antenna horizon now get zero Leff and a logged warning (dev wrapped to a wrong table row); negative azimuths no longer read one step off. | fix, #285 |
| `sim.detector.process_ant.AntennaProcessing.set_out_freq_mhz` | C1 | Non-array, empty, or not-starting-at-0 frequency arrays now raise ValueError with guidance, instead of AssertionError or IndexError. |  |
| `sim.detector.process_ant.PreComputeInterpol.init_linear_interpol` | same, C4 | No visible change. |  |
| `sim.efield2voltage.Efield2Voltage.__init__` | C3, W, A3, A4, B1, B2, C1 | New efield_level; params now default all nine keys; several efield levels warn, run tree must match; creates output_directory; bad seed, padding_factor<1, missing/inconsistent trees, unwritable output (even cwd) raise. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. | #328 |
| `sim.efield2voltage.Efield2Voltage.add` | C1 | Shape mismatch now raises ValueError naming both shapes, instead of an AssertionError that python -O skipped. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. |  |
| `sim.efield2voltage.Efield2Voltage.compute_voc_du` | A3, A4, B2, C1, C3 | Negative du_idx now raises IndexError (dev took units from the end); numpy integers accepted; calibration smearing drawn from a per-event generator seeded by seed, so different numbers. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. | #328, #331 |
| `sim.efield2voltage.Efield2Voltage.compute_voc_event` | A3, C1 | Now checks params first: unknown keys raise KeyError, non-bool flags (0/1, 'no') TypeError, lst outside 0–24 or negative values ValueError; dev ignored them or used truthiness. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. | #328 |
| `sim.efield2voltage.Efield2Voltage.compute_voltage` | A3, A5, B1, B2, C1, C3, C5 | Checks params first (as compute_voc_event); output tree kept open, written once via a hidden .partial file renamed at the end; failed runs leave no stub; append_file=False keeps all events. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. | #328 |
| `sim.efield2voltage.Efield2Voltage.compute_voltage_du` | A3, C1, C3, C4 | Negative du_idx now raises IndexError (dev used units from the end); no loaded event gives RuntimeError; numpy integers accepted; RF-chain transfer functions reused from get_event (same values). *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. | #328 |
| `sim.efield2voltage.Efield2Voltage.compute_voltage_event` | B2, C4 | Events with no units now return early instead of failing; RF-chain transfer functions reused from get_event, same values. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. | fix |
| `sim.efield2voltage.Efield2Voltage.final_resample` | A4, B1, B2, C1 | With only the nut or GAA RF chain enabled (no noise, no main chain), vout now includes that chain; dev returned V_oc. Empty events skipped; no event gives RuntimeError. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. | #328 |
| `sim.efield2voltage.Efield2Voltage.get_event` | A3, B1, B2, C1, C3, C5 | Raises for events absent from efield/shower/run trees, negative event_idx, event_idx with event_number, NaN traces, Xmax <100 m from core, bad t_bin_size, <16 samples; rf_chain.get_tf() then raises (arrays released). *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. | #328 |
| `sim.efield2voltage.Efield2Voltage.get_leff` | C1 | A unit position that is NaN or beyond ±22000 km in any coordinate now raises ValueError here; dev checked only values above +22000 km. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. |  |
| `sim.efield2voltage.Efield2Voltage.multiply` | C1 | Frequency-count mismatch now raises ValueError naming the shapes, instead of an AssertionError that python -O skipped. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. |  |
| `sim.efield2voltage.Efield2Voltage.save_voltage` | A3, A4, A5, B1, B2, C1, C2 | Refuses resampled voltages (resample_to_mhz>0) with ValueError; writes grandlib_version and the efield's analysis_level (dev: 0); single-file input without f_output works; jitter with seed 0 now seeded. *Review:* `dev`'s Efield2Voltage does not run with NumPy 2.5 (`only 0-dimensional arrays can be converted`); its side comes from reading the code. | #328 |
| `sim.efield2voltage.get_fastest_size_fft` | W, C1 | Warns (GRANDlibWarning) when f_samp_mhz lies outside 1–1e5 MHz; padding_factor below 1 or NaN raises ValueError instead of AssertionError. | #328 |
| `sim.noise.Compute_Plot_Galactic_Noise.main` | A3 | Efield mode: the B_nu and E_rms^2 y-labels became raw strings with a doubled backslash before pi, invalid mathtext; savefig of galactic_bnu should now raise ValueError. | #327 |
| `sim.noise.galaxy.galactic_noise` | A4, B1, A3, W, C1 | New per-du_type P_L tables, LST in 20-min bins (dev: int(lst)-1), per-bin amplitude ×√2 with real DC/Nyquist: different noise. f_lst outside [0,24), non-uniform grids raise; Hz grids warn. | #331, #328 |
| `sim.noise.galaxy.interpol_at_new_x` | C1, C4 | Empty sample positions now raise ValueError instead of AssertionError; interpolation unchanged (kind now passed by keyword). |  |
| `sim.shower.gen_shower.ShowerEvent.load_root` | A4, B1 | Xmax frame detection: maximum is now ground-relative even when the file stores Xmax height above sea level (filed #313). | #313 |

## grand.geo

| Function or script | Classes | What a caller sees on dev-next | Issue |
|---|---|---|---|
| `geo.coordinates.CartesianRepresentation.__new__` | C1, C5 | x, y, z arrays of different lengths raise ValueError naming each length instead of AssertionError; the TypeError for non-numeric x is reworded. |  |
| `geo.coordinates.Coordinates.__new__` | C5 | TypeError for a non-integer n now has one 'GRANDlib: <class>: ...' message instead of a tuple of message parts and the type. |  |
| `geo.coordinates.ECEF.__init__` | C5 | TypeErrors for unsupported argument or x/y/z types have reworded 'GRANDlib: ECEF: ...' messages; exception type unchanged. |  |
| `geo.coordinates.ECEF.__new__` | C5 | TypeError for an unsupported position type has a reworded 'GRANDlib: ECEF: ...' message; exception type unchanged. |  |
| `geo.coordinates.ECEF.ecef_to_grandcs` | W, C3 | New optional location sets the GRANDCS origin; without it the result is unchanged but a GRANDlibWarning says the default array origin is used. | #329 |
| `geo.coordinates.GRANDCS.__init__` | C5 | TypeErrors for unsupported argument or x types have reworded 'GRANDlib: GRANDCS: ...' messages; exception type unchanged. |  |
| `geo.coordinates.Geodetic.__init__` | A4, B2, C5 | A longitude array the caller passes is no longer modified in place (dev wrapped negative values to +360 in the caller's array); TypeError messages reworded. | #329 |
| `geo.coordinates.Geodetic.__new__` | A3, W, B2, C1, C5 | Components now checked: missing height, bool, mixed scalar/array, latitude beyond 90, longitude beyond 360 or height below -6.4e6 m raise; NaN or inf warns. | #329 |
| `geo.coordinates.Geodetic.geodetic_to_grandcs` | W, C3 | New optional location sets the GRANDCS origin; without it the result is unchanged but a GRANDlibWarning says the default array origin is used. | #329 |
| `geo.coordinates.Geodetic.geodetic_to_horizontal` | A3, B2, C3 | Was a stub returning None; now takes location and returns a Horizontal, so a call without location raises TypeError. | #329 |
| `geo.coordinates.GeodeticRepresentation.__new__` | C1, C5 | Latitude, longitude and height arrays of different lengths raise ValueError instead of AssertionError; the TypeError for non-numeric latitude is reworded. |  |
| `geo.coordinates.Horizontal.__new__` | A3, A4, B1, B2, C5 | location, basis, vector are per instance, not shared class attributes: earlier Horizontals keep their own location; Horizontal(h, location=X) uses X; zero-length vector elevation 0, not NaN; slices/copies lack location. | #329 |
| `geo.coordinates.HorizontalRepresentation.__new__` | A3, B2, C1, C5 | Elevation outside [-90, 90] degrees or a negative norm now raises ValueError (also via Horizontal); length mismatch raises ValueError instead of AssertionError. | #329 |
| `geo.coordinates.LTP.__init__` | A3, B2, C1, C5 | orientation must be exactly three letters, one per E/W, N/S, U/D ('ENUX', 'EEU' now raise); rotation matrices must be orthogonal with det +1 within 1e-6 (4-decimal matrices fail). | #329 |
| `geo.coordinates.LTP.__new__` | C5 | TypeError for an unsupported position type has a reworded 'GRANDlib: LTP: ...' message; exception type unchanged. |  |
| `geo.coordinates.LTP.ltp_to_grandcs` | A4, W, B1, C3 | Returns the position converted to GRANDCS instead of None; new optional location sets the origin, otherwise a GRANDlibWarning about the default origin is emitted. | #329 |
| `geo.coordinates.SphericalRepresentation.__new__` | A3, B2, C1, C5 | theta outside [0, 180] degrees or a negative r now raises ValueError; length mismatch raises ValueError instead of AssertionError; TypeError message reworded. | #329 |
| `geo.coordinates.geoid_undulation` | A2, A3, A4, B1, B2, C3 | Negative longitudes now wrapped to [0, 360), giving the undulation instead of NaN; geoid_undulation(lat, longitude=lon) raises TypeError; \|lat\|>90 or \|lon\|>360 raise ValueError; accepts a position. | #329 |
| `geo.geomagnet.Geomagnet.__init__` | C1, C5 | An unknown model name raises ValueError listing the available models instead of GULL's LibraryError; obstime and location TypeError messages reworded. |  |
| `geo.gull.Snapshot.__call__` | same | No visible change. |  |
| `geo.gull.Snapshot.__del__` | same, C4 | No visible change. |  |
| `geo.gull.Snapshot.__init__` | C1, C3 | filename defaults to GRAND_ROOT's IGRF13.COF (dev's '' default always failed) and may be a Path; a missing file raises FileNotFoundError instead of LibraryError. |  |
| `geo.topography.Topography.distance` | C3, C4 | New optional frame argument converts the direction from that frame to ECEF before tracing; existing calls unchanged. |  |
| `geo.topography.Topography.elevation` | A3, B2, C1, C5 | Unknown reference strings now raise ValueError (dev silently used sea level); wrong coordinates for 'LOCAL' and a non-string reference raise TypeError instead of ValueError. | #329 |
| `geo.topography.distance` | C3 | New optional frame argument, passed on to Topography.distance; existing calls unchanged. |  |
| `geo.topography.geoid_undulation` | A3, A4, B1, B2, C3 | Negative longitudes wrapped to [0, 360) (value, not NaN); positional (lat, lon) accepted; position plus latitude/longitude, or out-of-range angles, raise; no arguments give NaN. | #329 |
| `geo.turtle.Map.__del__` | same, C4 | No visible change. |  |
| `geo.turtle.Map.__init__` | A4, C1 | Map(Path).path now returns a str, not the Path given; the cache is keyed by real path; a missing file raises FileNotFoundError instead of LibraryError. | #329 |
| `geo.turtle.Map.__new__` | A4, B2 | Cache lookup uses the file's real path: str, Path, relative or symlinked spellings of one file return the same Map; a relative path after chdir loads the right file. | #329 |
| `geo.turtle.Map.elevation` | A4, B1, C1 | Multi-dimensional x, y now give an elevation array of that shape (dev flattened); NaN points give NaN with a GRANDlibWarning instead of crashing; non-numeric input raises TypeError. | #329 |
| `geo.turtle.Stack.elevation` | B1, C1 | NaN or None coordinates give NaN with a GRANDlibWarning instead of crashing the interpreter; non-numeric input raises TypeError; finite inputs unchanged. | fix |
| `geo.turtle.ecef_from_geodetic` | A3, B2, C1 | Inputs whose array dtype is not int or float raise TypeError: bool, complex, Fraction, Decimal, object arrays (dev coerced them); strings raise TypeError instead of ValueError. | #329 |

## grand.basis

| Function or script | Classes | What a caller sees on dev-next | Issue |
|---|---|---|---|
| `basis.du_network.DetectorUnitNetwork.get_max_dist_du` | C3 | Now returns the largest distance between two units in km (0.0 for fewer than two) instead of raising NotImplementedError. |  |
| `basis.du_network.DetectorUnitNetwork.get_surface` | B2, C4 | With fewer than 3 units, returns 0 on the first call instead of raising from Delaunay; area computed without deprecated 2-D np.cross, same values. | fix |
| `basis.du_network.DetectorUnitNetwork.init_pos_id` | A3, B2, C1, C3 | Malformed du_pos or du_id raise ValueError instead of AssertionError; du_pos with more than two dimensions, accepted by dev, now raises; tuple du_id now accepted. | #330 |
| `basis.du_network.DetectorUnitNetwork.plot_footprint_1d` | same | No visible change. |  |
| `basis.du_network.DetectorUnitNetwork.plot_footprint_4d` | same | No visible change. |  |
| `basis.du_network.DetectorUnitNetwork.plot_footprint_time` | C1 | Mismatched a_time and a3_values sample counts raise ValueError instead of AssertionError. |  |
| `basis.signal.find_max_with_parabola_interp` | same | No visible change. |  |
| `basis.signal.get_fastest_size_fft` | W, C1 | Sampling rates outside 1 to 1e5 MHz now emit a GRANDlibWarning; padding_fact below 1 raises ValueError instead of AssertionError. Results unchanged. | #330 |
| `basis.signal.interpol_at_new_x` | C1 | Empty a_x raises ValueError instead of AssertionError. |  |
| `basis.traces_event.Handling3dTraces.copy` | C1 | A new_traces array of the wrong shape raises ValueError instead of AssertionError. |  |
| `basis.traces_event.Handling3dTraces.get_tmax_vmax` | A4, B1, C1, C4 | With one trace, returns arrays of shape (1,) instead of 0-d with interpol='no', and no longer crashes with interpolation; invalid interpol raises ValueError instead of RuntimeError. | #330 |
| `basis.traces_event.Handling3dTraces.init_traces` | A3, A4, B2, C1, C3 | Zero, negative or non-finite rates, or a one-element rate array for several units, now raise ValueError; NumPy-scalar and integer-array rates become float64 per-unit arrays; lists accepted. | #330 |
| `basis.traces_event.Handling3dTraces.plot_all_traces_as_image` | same | No visible change. |  |
| `basis.traces_event.Handling3dTraces.reduce_nb_trace` | A3, C1 | Counts below 1 or above the trace count raise ValueError instead of AssertionError; a bool count, which dev took as 1, now raises TypeError. | #330 |
| `basis.traces_event.Handling3dTraces.remove_trace_low_signal` | C1 | norm_traces of the wrong length raises ValueError instead of AssertionError. |  |
| `basis.traces_event.Handling3dTraces.set_periodogram` | A3, C1 | Non-integer sizes such as 500.5, which dev stored, now raise TypeError; sizes below 1 raise ValueError instead of AssertionError. | #330 |
| `basis.traces_event.Handling3dTraces.set_unit_axis` | C1 | Non-string arguments raise TypeError instead of AssertionError. |  |
| `basis.traces_event.get_psd` | W, C1 | Sampling rates outside 1 to 1e5 MHz now emit a GRANDlibWarning; the PSD is unchanged. | #330 |
| `basis.type_trace.ElectricField.__post_init__` | C1 | Mismatched a_time and e_xyz sample counts raise ValueError instead of AssertionError. |  |
| `basis.type_trace.Voltage.__post_init__` | A3, B2 | Construction now raises ValueError when t is not 1-D or V's last axis length differs from len(t), e.g. V shaped (n, 3); dev accepted any shapes. | #330 |
| `manage_log.create_output_for_logger` | A4, A5, B2 | Repeated calls replace earlier calls' handlers instead of adding more: no duplicated lines, but an earlier call's file or console output stops; a reused log file is appended, not truncated. | #330 |

## Command-line scripts

| Function or script | Classes | What a caller sees on dev-next | Issue |
|---|---|---|---|
| `scripts/Compute_Vout_AT_Device_save.py` | C1, C3 | An unknown option is now an argparse usage error (exit 2) instead of a traceback; new --out_dir for the --savedata text files. |  |
| `scripts/T1_trigger_offline.py` | A2, A4, A5, B1, B2, C3 | Every DU is now tested, not only the first, so the triggered count and list can grow; the list's header gains two keys; argparse adds -o, --t1_param, rejects extra arguments. | #332 |
| `scripts/convert_efield2efield.py` | A3, A4, A5, B1, B2, C1, C5 | With -od, L1 run files now go into -od; du_count now written; existing outputs replaced; --seed 0 now seeds, identically per event; inf, empty or mismatched inputs exit. | #332 |
| `scripts/convert_efield2voltage.py` | A3, A5, W, B1, B2, C1, C3, C5 | Nonzero --target_sampling_rate_mhz now refused; an existing output is overwritten, not appended; warns when several efield levels exist; new --level; --du_type Horizon and out-of-range values rejected. | #332 |
| `scripts/convert_voltage2adc.py` | A3, A4, A5, W, B1, B2, C1, C3, C5 | Bare -o now writes into the input folder; noise drawn without replacement (same seed, different noise); noise dir without '/' now read; run file at the voltage's level used. | #332, #331 |
| `scripts/extract_events.py` | A3, A4, A5, B1, B2, C1, C5 | -ow deletes only GRAND files; -c goes into tree metadata; every level's tree is written; missing events exit 1; target '.' or containing a source refused; output lines changed. | #332 |
| `scripts/extract_rf_chain.py` | A3, A4, B2, C3, C5 | Refuses to replace an existing TF_RF_Chain.npy without --overwrite; plots only with --plot; prints one 'wrote' line instead of the arrays; new -o. | #332 |
| `scripts/get_version.py` | A4, B2 | Run outside the repository root, it now prints the real version instead of version=0.0.0. | fix |
| `scripts/open_grand_analysis_prompt.py` | B2, C4 | Directory names with quotes now open instead of breaking the session; 'Reading directory' shows the directory, not a leading option. | fix |
| `scripts/open_grand_directory.py` | B2, C4 | Directory names with quotes now open instead of breaking the session; otherwise the same. | fix |
| `scripts/open_grand_file.py` | B2, C4 | File names with quotes now open instead of breaking the session; otherwise the same. | fix |
| `scripts/transfers/bintoroot.bash` | A1, A4 | Moved to scripts/pipeline/. At the new path grand_dir no longer resolves, so register_convert.py and the granddb registration scripts are not found; conversions go unregistered. | #326 |
| `scripts/transfers/refresh_mat_views.bash` | A1 | Moved to scripts/pipeline/ and lost its executable bit (mode 644): calls by the old path, or running it directly, fail. | #326 |
| `scripts/transfers/register_convert.py` | A1, A3, B2, C5 | Moved to scripts/pipeline/. An input file not in the database now exits with status 1 and the message on stderr, instead of status 0 on stdout. | #326 |
| `scripts/transfers/register_transfers.py` | A1 | Moved to scripts/pipeline/; otherwise unchanged. | #326 |
| `scripts/transfers/run_monitoring.bash` | A1, A4 | Moved to scripts/pipeline/; now activates the grandlib_prod_6.36 conda environment instead of grandlib_test. | #326 |
| `scripts/transfers/tar_logs.bash` | A1 | Moved to scripts/pipeline/ and lost its executable bit (mode 644); ccscript.bash still runs it directly by the old path. | #326 |
| `scripts/plot_Vout_AT_Device.py` | C1, C5 | An unknown option is now an argparse usage error (exit 2) instead of a traceback; --savefig help corrected. |  |
| `scripts/plot_noise.py` | A2, A3, B2, C1 | --lst now takes whole hours 0-23 only: 18.0, 3.7 and -1 are rejected (dev truncated or wrapped them); --du_type checked by argparse. | #332 |
| `scripts/plot_rf_chain.py` | C1, C5 | An unknown option is now an argparse usage error (exit 2) instead of a traceback listing the wrong options; --savefig help corrected. |  |
| `scripts/plot_tmax_vmax.py` | A2, A3, B2, C1, C3, C5 | A non-integer event index is now rejected instead of silently 0; negative index refused; no longer prints the file name; new --savefig; extra arguments rejected. | #332 |
| `scripts/transfers/transfer_from_obs.bash` | A5, B2 | Lock file moved from /tmp/grand_transfer_lock to $XDG_RUNTIME_DIR or $HOME, so cron and login runs may not exclude each other; file names and messages are SQL-quoted. | #326 |
