import granddatalib
import psycopg2
import time

# file = 'memo_Runner_step_radio.txt'
#file = 'LyonAiresLibraryRun.ini'
#file = 'granddatalib.py'
#file = 'readme.md'
file = 'prolongation_login.pdf'
#file = 'xrx7830.ppd'

dm = granddatalib.DataManager('config.ini')
#conf.find_repo("CCA")
#for rep in dm.repositories():
#    print("repository : " + rep.name() + " paths : " + str(rep.paths()))
#    #print(rep.credentials().user())

#for dir in dm.directories():
#    print("directory : "+dir.name() + " paths : " +str(dir.paths()))

file = 'main.py'
print("\nGet in localdir incoming")
st = time.time()
print("RESULTAT : " + str(dm.get(file, "localdir","./incoming",)))
et = time.time()
print((et-st)*1000)

exit(0)

file = 'test.py'
print("\nSearch")
st = time.time()
print("RESULTAT : " + str(dm.search(file)))
et = time.time()
print((et-st)*1000)


file = 'toto'
print("\nGet in localdir")
st = time.time()
print("RESULTAT : " + str(dm.get(file, "localdir")))
et = time.time()
print((et-st)*1000)





exit(0)
file = 'toto'
print("\ngetfile2")
st = time.time()
print("RESULTAT : " + str(dm.get(file)))
et = time.time()
print((et-st)*1000)





#print("RESULTAT : " + str(conf.search(file)))


#conn = psycopg2.connect(
#    host="lpndocker01.in2p3.fr",
#    database="grand",
#    user="postgres",
#    password="postgres")