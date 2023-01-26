'''
Created on Dec 12, 2022

@author: root
'''
import grand.io.root_files as grf
import matplotlib.pyplot as plt

fe=grf.FileSimuEfield('Coarse2.root')
#fe=grf.FileVoltageEvent('c2_test.root')
h_tr=fe.get_obj_handling3dtraces()
h_tr.plot_footprint_4d()
plt.show()