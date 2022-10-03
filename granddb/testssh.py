import granddatalib
import psycopg2

# file = 'memo_Runner_step_radio.txt'
file = 'LyonAiresLibraryRun.ini'
#file = 'xrx7830.ppd'

conf = granddatalib.DataManager('config.ini')

print("RESULTAT : " + str(conf.search(file)))


#conn = psycopg2.connect(
#    host="lpndocker01.in2p3.fr",
#    database="grand",
#    user="postgres",
#    password="postgres")