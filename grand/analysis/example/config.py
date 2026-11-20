# --------------------------------------------------
# Configuration & paths
# --------------------------------------------------

import os

# Base directory of this script
base_dir = os.path.dirname(__file__)

use_examples = True

# Also run the GNN reconstruction in main_DOI.py: its results go to the same
# entries as the PWF/SWF/ADF ones, in the *_gnn branches.  Needs torch,
# torch_geometric and the networks (data/download_gnn_models.py).
use_gnn = True

# Which network for each task.  None: the default, chosen in
# grand/analysis/gnn/settings.json.  Otherwise "name/version" or "name"
# (its latest version) in the GNN models directory, or the path of a
# network's directory (the one holding its gnn_model_card.json).  The GNN model card says
# which channels, features, trigger and graph the network needs.
gnn_models = {
    "direction": None,
    "energy": None,
}

if use_examples:
    # --------------------------
    # Example data configuration
    # --------------------------

    # Flagged events list
    flagged_txt = os.path.join(
        base_dir, "Flagged_events_July_October.txt"
    )

    # Antenna position file
    antenna_file = os.path.join(
        base_dir, "_gp13_65_rtksort.txt"
    )

    # Base directory for GP80 ROOT files
    path_rootfile = "/sps/grand/data/gp80/GrandRoot/"

    # Output directory for reconstruction
    output_dir = base_dir


    # Output ROOT file name
    output_file = "recons_CR_candidates.root"

    # Full output path
    output = os.path.join(output_dir, output_file)
