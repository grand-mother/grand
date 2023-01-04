from granddatalib import DataManager



dm = DataManager()
filename = 'Coarse2.root'
# filename = 'Coarse2_xmax_add.root'
filename = 'Coarse3.root'
# filename = 'Coarse4.root'
#filename = '_Filter_EfieldVSignal_LST18_XDS_Stshp_0.117_22.8_0.0_vertical_radius5325.58_100resamples.root'
# filename = '_FilterNewIntepolation_EfieldVSignal_Iron_MZS_QGSP_3.98_79.6_180.0.root'
# filename = '_Filter_EfieldVSignal_LST18_Proton_MZS_3.98_79.6_0.0.root'


#file = dm.get(filename)
print(dm.register_file(filename))





