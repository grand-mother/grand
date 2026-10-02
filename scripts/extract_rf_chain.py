#!/usr/bin/env python3
"""
Extract global TF of RF chain.

Writes a (4, n_freq) complex array: the frequencies in MHz, then the transfer
function of each arm.

Nov. 2024, Colley Jean-Marc
"""

import argparse
import os

import numpy as np


def main(argv=None):
    # It took no options, ignored -h and wrote into the current folder (#246)
    parser = argparse.ArgumentParser(description="Write the combined RF-chain transfer function (30-251 MHz).")
    parser.add_argument("-o", "--out", default="TF_RF_Chain.npy", help="output .npy file (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--plot", action="store_true", help="also plot the three arms")
    args = parser.parse_args(argv)
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit("GRANDlib: extract_rf_chain: %s exists; use --overwrite" % args.out)

    import grand.sim.detector.rf_chain as grfc

    rfchain = grfc.RFChain()
    freq_MHz = np.linspace(30, 251, 251 - 30 + 1)
    rfchain.compute_for_freqs(freq_MHz)
    rfc = rfchain.get_tf()
    freq_rfc = np.zeros((4, len(freq_MHz)), dtype=rfc.dtype)
    freq_rfc[1:] = rfc
    freq_rfc[0] = freq_MHz
    np.save(args.out, freq_rfc)
    print("wrote %s %s" % (args.out, freq_rfc.shape))

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure()
        for arm in (1, 2, 3):
            plt.plot(freq_rfc[0].real, np.abs(freq_rfc[arm]), label=str(arm))
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
