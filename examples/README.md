# examples/

What each example needs, and whether it runs on a fresh checkout (#218).
"Here" means with the repository alone, after `source env/setup.sh`; the
committed simulation samples are under `sim2root/Common/`.

| Example | Runs here? | Notes |
|---|---|---|
| `aoi/event_generation.py` | yes | writes `dummy_example_events.root` (10 random events), replacing an earlier one |
| `aoi/data_play.py` | yes | after `event_generation.py`: `python data_play.py dummy_example_events.root` |
| `aoi/browse_sim2root_events_example.py` | yes, in a terminal | waits for Enter between events; give it a `sim2root/Common/sim_*` folder |
| `aoi/browse_gp13_events_example.py` | no | needs measured GP13 data |
| `aoi/*.ipynb` | yes | |
| `analysis/main_AOI.py`, `main_DOI.py` | yes | the reconstruction; see `analysis/README.md` |
| `analysis/display.py` | yes, from its folder | needs pandas; `--savefig DIR` saves instead of opening windows |
| `dataio/data_storing.py`, `data_reading.py` | yes | `data_reading.py` reads what `data_storing.py` wrote |
| `dataio/datafile_use.py` | yes | give it a ROOT file |
| `dataio/ioroot_3dtraces.py` | yes | |
| `datalib/datamanager_example.py` | no | edit `config.ini` first; needs the GRAND database and its servers |
| `eventviewer/` | yes | needs `pip install -e ".[viewer]"`; see its README |
| `geo/local_topography.py` | yes | `--download` fetches about 50 MB of topography tiles first |
| `geo/*.ipynb`, `grids.py`, `hexy.py`, `GP100_topography.py` | yes | the topography ones need the tiles of their area |
| `sim/shower_event.py` / `.ipynb` | yes | on the committed RUN1 sample |
| `sim/rf_chain_example.py` | yes | `python rf_chain_example.py galactic --lst 18` |
| `sim/antenna.ipynb`, `galactic_noise.ipynb`, `plot_VoltageAtDevice.ipynb`, `rf_chain_example.ipynb` | yes | |
| `jm_rootfiles_based/class_Handling3dTraces.ipynb` | yes | |
| `read_modify_RF_chain_elements.ipynb` | yes | |
| `old/` | no | kept for reference: outdated APIs, stubs that only print that they do not work, absolute paths of other machines |

The maintained, executed walk-throughs are the notebooks in `notebooks/`
(see the documentation's Notebooks page); these examples are older and less
checked.
