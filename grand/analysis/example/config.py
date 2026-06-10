# --------------------------------------------------
# Configuration & paths
# --------------------------------------------------

import os

# Base directory of this script
base_dir = os.path.dirname(__file__)

use_examples = True

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
