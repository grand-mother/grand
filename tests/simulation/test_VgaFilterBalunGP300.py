
import numpy as np

from grand.simu.du.rf_chain import VgaFilterBalunGP300



def test_compute_for_freqs():
    vfb = VgaFilterBalunGP300()
    # out freq is in freq
    print()
    vfb.compute_for_freqs(vfb.freqs_in)
    # 
    dbs11 = vfb.data_filter[:, 1]
    degs11 = vfb.data_filter[:, 2]
    mags11 = 10 ** (dbs11 / 20)
    res11 = mags11 * np.cos(np.deg2rad(degs11))
    assert np.allclose(res11,vfb._s11_real)
    
    