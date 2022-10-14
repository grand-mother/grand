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


dm = granddatalib.DataManager('config.ini')

file = 'toto'
print("\nGet")
st = time.time()
print("RESULTAT : " + str(dm.get(file)))
et = time.time()
print((et-st)*1000)



file = 'main.py'
print("\nGet in localdir incoming")
st = time.time()
print("RESULTAT : " + str(dm.get(file, "localdir","./incoming",)))
et = time.time()
print((et-st)*1000)




file = 'LyonAiresLibraryRun.ini'
print("\nGet in localdir")
st = time.time()
print("RESULTAT : " + str(dm.get(file, "localdir")))
et = time.time()
print((et-st)*1000)


file = 'LyonAiresLibraryRun.ini'
print("\nGet in CCIN2P3")
st = time.time()
print("RESULTAT : " + str(dm.get(file, "CCIN2P3")))
et = time.time()
print((et-st)*1000)



file = 'titi'
print("\nGet in CCIN2P3")
st = time.time()
print("RESULTAT : " + str(dm.get(file, "CCIN2P3")))
et = time.time()
print((et-st)*1000)



