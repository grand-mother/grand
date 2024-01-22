from granddatalib import DataManager



dm = DataManager()
#filename = 'Coarse2.root'
# filename = 'Coarse2_xmax_add.root'
#filename = 'Coarse3.root'
# filename = 'Coarse4.root'
#filename = '8818cab21736914cd7cf9306f50b5a2d4256c2f2f292549d48cfa04965ce9405.root'
#filename = '82c60c0ae8bcb50b04dd5a1fe3c2bd0755fa896f66f7d3e2db71b9ff55187e71.root'
filename = 'GRAND.TEST-RAW.20230310185213.008.root'

#filename = '_Filter_EfieldVSignal_LST18_XDS_Stshp_0.117_22.8_0.0_vertical_radius5325.58_100resamples.root'
# filename = '_FilterNewIntepolation_EfieldVSignal_Iron_MZS_QGSP_3.98_79.6_180.0.root'
# filename = '_Filter_EfieldVSignal_LST18_Proton_MZS_3.98_79.6_0.0.root'
#filename = 'FilmJERUSALEM.29622DVD1.mp4'
#filename = '7e152cfa3ad039a7f3dc964f3a88b049c50636268e9050788e530e228a59c473.root'
#filename = '82c60c0ae8bcb50b04dd5a1fe3c2bd0755fa896f66f7d3e2db71b9ff55187e71.root'
#print(dm.get(filename))
#file = dm.get(filename)

#filename = 'td002015_f0003.root'
#filename = 'td002016_f0002.root'
#filename = 'td230927_f0130.root'
#print("registering " + filename)
#file = dm.get(filename)
#print(dm.register_file(filename))


#filename = 'gr_Coreas_Run_6100.root'
#file = dm.get(filename)
#print("registering " + filename)
#print(dm.register_file(filename))
import os
rootdir="/home/fleg/DEV/GRAND/incoming/lyon/"
for filename in os.listdir(rootdir):
    if os.path.isfile(rootdir+filename):
        if filename not in ["td002011_f0005.root", "td002006_f0001.root"]:
            print("registering " + filename)
            print(dm.register_file(filename))
        else:
            print("skipping "+filename)
    else:
        print(filename+" is not file")

#for filename in ["td002015_f0008.root" , "td002001_f0001.root" , "td002003_f0001.root" , "td002006_f0013.root" , "td002007_f0004.root" , "td002007_f0018.root" , "td002011_f0005.root" , "td002015_f0003.root" , "md002000_f0004.root"]:
#    print("registering " + filename)
#    print(dm.register_file(filename))

#fileres = dm.SearchFileInDB(filename)
#print(fileres)
#print(f"search for {fileres[0][0]} at {fileres[0][1]}")

#print(dm.get(fileres[0][0], fileres[0][1]))


