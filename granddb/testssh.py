print("toto")
import granddatalib
import psycopg2
import psycopg2.extras

import time
# file = 'memo_Runner_step_radio.txt'
#file = 'LyonAiresLibraryRun.ini'
#file = 'granddatalib.py'
#file = 'readme.md'
file = 'prolongation_login.pdf'
#file = 'xrx7830.ppd'

print("\nStart")

dm = granddatalib.DataManager('config.ini')


file = 'Coarse3.root'
file = 'GRAND.TEST-RAW.20230309203415.001.root'
print("\nGet " + file)
st = time.time()
print("RESULTAT : " + str(dm.get(file)))
et = time.time()
print((et-st)*1000)



file = 'main.py'
print("\nGet in localdir incoming " + file)
st = time.time()
print("RESULTAT : " + str(dm.get(file, "localdir","./incoming",)))
et = time.time()
print((et-st)*1000)




file = 'td002015_f0003.root'
print("\nGet in localdir " + file)
st = time.time()
print("RESULTAT : " + str(dm.get(file, "localdir")))
et = time.time()
print((et-st)*1000)


file = 'LyonAiresLibraryRun.ini'
print("\nGet in CC " + file)
st = time.time()
print("RESULTAT : " + str(dm.get(file, "CC")))
et = time.time()
print((et-st)*1000)

file = 'LyonAiresLibraryRun.ini'
print("\nGet in CC /sps/trend/fleg/" + file)
st = time.time()
print("RESULTAT : " + str(dm.get(file, "CC", "/sps/trend/fleg/")))
et = time.time()
print((et-st)*1000)


file = 'Coarse3.root'
print("\nGet in WEB " + file)
st = time.time()
print("RESULTAT : " + str(dm.get(file, "WEB")))
et = time.time()
print((et-st)*1000)



