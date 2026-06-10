# grand.analysis

This module contains analysis tools for the **reconstruction of events in direction and energy**.

## Submodules

- **`signals`** – Signal processing: extract peak amplitude, peak times, convert voltage to ADC.

- **`fitting`** – Direction reconstruction with a pipeline in 3 steps: PWF, SWF, and ADF.  
  This pipeline can be used on either **peak amplitudes in ADC counts** or **efield**.

- **`geom`** – Compute different geometry tools, such as the **geomagnetic angle**, the distance Xsource–antenna, and the angles (omega, eta).

- **`coords`** – Compute coordinate tools, such as conversion from **Cartesian coordinates in ground reference** to **shower plane reference**.

- **`energy_reco`** – Energy reconstruction (currently only a first proxy directly on ADC data).  
  **TODO:** implement the code to reconstruct the energy from efield (more robust).

- **`constants`** – Physical constants used in analysis + magnetic field values.

- **`physics`** – Physical tools:  
  - `atmosphere.py`: gives the effective refractive index computed as in ZHS, between Xsource and the antenna  
  - `cherenkov_angle.py`: compute the Cherenkov angle using a two-emission-point toy model around the reconstructed Xsource obtained with SWF.  
    The Cherenkov angle is the point on the ground where the light path between the two points is minimal.

- **`example`** – Example script `main.py` directly applicable to the 73 selected candidates (`Flagged_events_July_October.txt`).  
  The script processes the ROOT files (stores peak times and amplitudes), applies the reconstruction (on ADC data), and computes the energy.  
  The results of the reconstruction are stored in `recons_CR_candidates.root`.

  – In this example, a file `_gp13_65_rtksort.txt` is provided, corresponding to the RTK positions of the antennas (better reconstruction results).  

  – `display.py` script that loops over reconstructed events and displays the amplitude profile with the ADF fit and the reconstructed footprint on the ground.

- **`cramer_rao_bounds`** - Main **CRB functions** used in the reconstruction pipeline to estimate the order of magnitude of errors.

