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

- **`gnn`** (Arsène Ferrière) – **Graph neural network** reconstruction of the
  direction (`recons_GNN`) and the electromagnetic energy (`recons_energy_GNN`),
  stored in `TRecons` as `zenith_gnn`, `azimuth_gnn`, `energy_em_gnn` and their
  `std_*`. The peaks are those of `signals` (the functions of the ADF chain), on
  the channels the network was trained on (`gnn.peaks`; their index in measured
  and simulated traces differs, and is set in one place, `constants.CHANNEL_INDEX`). A script names its
  networks (`gnn.load_model("direction", model)`, `model` being `"name/version"`,
  `"name"` for its latest version, or a directory); by default, the ones
  `gnn/settings.json` (or `$GRAND_GNN_SETTINGS`) chooses. Each network is a
  directory, `<GRAND_DATA_PATH>/gnn/<name>/<version>/` when downloaded by
  `data/download_gnn_models.py`, whose `gnn_model_card.json` states its channels,
  features, trigger (antennas above 5 sigma of the noise by default,
  `gnn.features.trigger`), neighbour graph (`knn_k`) and normalisation: the graph
  is built from it. Features are the functions of `NODE_FEATURES` and
  `GRAPH_FEATURES` in `gnn/features.py`; a GNN model card names the ones it uses, in
  order. Defaults of new networks, in `gnn/defaults.json`:
  pGNN on x and y, node features (pos_x, pos_y, pos_z, time, amplitude) and the
  plane-wave direction; a stored GNN model card states all its values and does not read them.
  Needs torch and torch_geometric. Training is in a
  separate repository, which uses `gnn.features.make_graph` and `GNNScaleMDN` so
  that the networks are trained on exactly the inputs they see here.
  Run in `example/main_DOI.py` with `use_gnn = True` and the networks
  in `gnn_models` of its `config.py`: the GNN results go to the same entries
  as the PWF/SWF/ADF ones.

- **`example`** – Example script `main.py` directly applicable to the 73 selected candidates (`Flagged_events_July_October.txt`).  
  The script processes the ROOT files (stores peak times and amplitudes), applies the reconstruction (on ADC data), and computes the energy.  
  The results of the reconstruction are stored in `recons_CR_candidates.root`.

  – In this example, a file `_gp13_65_rtksort.txt` is provided, corresponding to the RTK positions of the antennas (better reconstruction results).  

  – `display.py` script that loops over reconstructed events and displays the amplitude profile with the ADF fit and the reconstructed footprint on the ground.

