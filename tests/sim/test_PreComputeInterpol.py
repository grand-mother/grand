'''
Created on 20 févr. 2023

@author: jcolley
'''

import numpy as np

from grand.simu.du.process_ant import PreComputeInterpol

F_in = np.linspace(20, 30, num=11, endpoint=True)
D_freq_out = 2.1
F_out = np.arange(20) * D_freq_out    


def test_precompute():
    '''
    freq in : ~frequency band [20, 30]
    freq out : step 2.1, [0, 2.1, 4.2, ....]
    
    idx out , freq out,   freq in , idx in
    9             18.9        out of band
    10            21          21        1
    11            23.1        23        3
    12            25.2        25        5
    13            27.3        27        7
    14            29.4        29        9
    15            30.5        out of band
    '''    
    pre = PreComputeInterpol()
    pre.init_linear_interpol(F_in, F_out)
    # index in , in band 
    assert pre.idx_first == 10
    assert pre.idx_lastp1 == 15
    # index inf for interpolation
    assert np.allclose(pre.idx_itp, np.array([1, 3, 5, 7, 9]))
    # coef lineat interpolation
    c_sup = np.array([0, 0.1, 0.2, 0.3, 0.4])
    assert np.allclose(pre.c_sup, c_sup)
    assert np.allclose(pre.c_inf, 1 - c_sup)
    print(pre)
    

def test_interpol():
    pre = PreComputeInterpol()
    pre.init_linear_interpol(F_in, F_out)
    out = pre.get_linear_interpol(F_in)
    # out must be equal to in frequency
    assert np.allclose(out, np.array([21, 23.1, 25.2, 27.3, 29.4]))
 
