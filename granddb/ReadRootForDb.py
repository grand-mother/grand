from granddatalib import DataManager



dm = DataManager()
filename = 'Coarse2.root'
# filename = 'Coarse2_xmax_add.root'
filename = 'Coarse3.root'
# filename = 'Coarse4.root'
#filename = '_Filter_EfieldVSignal_LST18_XDS_Stshp_0.117_22.8_0.0_vertical_radius5325.58_100resamples.root'
# filename = '_FilterNewIntepolation_EfieldVSignal_Iron_MZS_QGSP_3.98_79.6_180.0.root'
# filename = '_Filter_EfieldVSignal_LST18_Proton_MZS_3.98_79.6_0.0.root'
#filename = 'FilmJERUSALEM.29622DVD1.mp4'
#filename = '7e152cfa3ad039a7f3dc964f3a88b049c50636268e9050788e530e228a59c473.root'

print(dm.get(filename))
#file = dm.get(filename)
print(dm.register_file(filename))

#fileres = dm.SearchFileInDB(filename)
#print(fileres)
#print(f"search for {fileres[0][0]} at {fileres[0][1]}")

#print(dm.get(fileres[0][0], fileres[0][1]))


