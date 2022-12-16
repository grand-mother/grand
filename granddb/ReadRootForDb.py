import rootdblib as rdb
import ast
from granddatalib import DataManager
tables = {}
paramvalue = {}

dm = DataManager()
filename = 'Coarse2_xmax_add.root'
filename = 'Coarse3.root'
#filename = '_Filter_EfieldVSignal_LST18_XDS_Stshp_0.117_22.8_0.0_vertical_radius5325.58_100resamples.root'
#filename = '_FilterNewIntepolation_EfieldVSignal_Iron_MZS_QGSP_3.98_79.6_180.0.root'
#filename = '_Filter_EfieldVSignal_LST18_Proton_MZS_3.98_79.6_0.0.root'


file = dm.get(filename)

if file is None:
    exit(1)
rfile = rdb.RootFile(file)
print("list of trees:")
[print(treename) for treename in rfile.TreeList]
print()

for treename in rfile.TreeList:
    if treename in ["teventefield", "teventshowersimdata", "teventshowerzhaires", 'teventshower', 'teventvoltage']:
       print(treename)
       table = getattr(rfile, treename + "ToDB")['table']
       tables[table] = {}
       for event, run in rfile.TreeList[treename].get_list_of_events():
           rfile.TreeList[treename].get_event(event, run)
           # print("EVENT : " + str(rfile.TreeList[treename].event_number) + " RUN "+ str(rfile.TreeList[treename].run_number))
           for param, field in getattr(rfile, treename + "ToDB").items():
               if param == "table":
                    #print("values will go to table " + field)
                    #table = field
                    None
               else:
                   tables[table][field] = str(getattr(rfile.TreeList[treename], param))
                   #print("param " + param + " with value " + str(getattr(rfile.TreeList[treename], param)) + " goes to field " + field + " in table " + table)

           var = tables[table]
           print(var)
           containerevent = dm.database().tables()[table](**var)
           #dm.database().sqlalchemysession.add(containerevent)
           #dm.database().sqlalchemysession.commit()


    if treename in ["trun", "trunefieldsimdata"]:
        print(treename)
        #print("table is " + str(getattr(rfile, treename + "ToDB")['table']))
        table = getattr(rfile, treename + "ToDB")['table']
        #tables[table] = []
        tables[table] = {}
        for run in rfile.TreeList[treename].get_list_of_runs():
            #print("RUN IS:" + str(run))
            rfile.TreeList[treename].get_run(run)
            for param, field in getattr(rfile, treename + "ToDB").items():
                if param == "table":
                    #print("values will go to table " + field)
                    #table = field
                    None
                else:
                    #print("param " + param + " with value " + str(
                    #    getattr(rfile.TreeList[treename], param)) + " goes to field " + field + " in table " + table)
                    #tables[table].append((field,getattr(rfile.TreeList[treename], param)))
                    #tables[table].append(field+"="+str(getattr(rfile.TreeList[treename], param)))
                    tables[table][field]=str(getattr(rfile.TreeList[treename], param))
            var = tables[table]
            print(var)
            var = {'run_number': '0', 'id_run_mode': '0', 'id_first_event': '3', 'first_event_time': '0', 'id_last_event': '3', 'last_event_time': '0', 'id_data_source': '0', 'id_data_generator': '0', 'id_data_generator_version': '0', 'id_site': '0', 'site_long': '0.0', 'site_lat': '0.0'}
            containerrun = dm.database().tables()[table](**var)
            #dm.database().sqlalchemysession.add(containerrun)
            #dm.database().sqlalchemysession.commit()


id_provider = 101
#query = "INSERT INTO file ( filename, description, original_name, id_provider) VALUES ('" + filename + "', 'descriptiondufichier','" + filename + "'," + str(
#    id_provider) + ") RETURNING id_file "
#record = dm.database().insert(query, values)

#query = "INSERT INTO file ( filename, description, original_name, id_provider) VALUES ( %s, %s, %s, %s) RETURNING id_file"
#values = (filename, 'descriptiondufichier', filename, str(id_provider))
#print(query)
#record = dm.database().insert2(query, values)
#print(record)

#print(dm.database().tables())
#newfile = dm.database().tables()['file'](filename=filename, description='ceci est un fichier', original_name=filename,id_provider=id_provider)
#dm.database().sqlalchemysession.add(newfile)
#dm.database().sqlalchemysession.flush()
#print(newfile.id_file)
#newfilelocation = dm.database().tables()['file_location'](id_file=newfile.id_file,id_repository=0)
#dm.database().sqlalchemysession.add(newfilelocation)
#dm.database().sqlalchemysession.commit()


#for table in tables.keys():
#    print("Values to table : " + table)
#    print(tables[table])
#    var = {'run_number': '0', 'id_run_mode': '0', 'id_first_event': '3', 'first_event_time': '0', 'id_last_event': '3', 'last_event_time': '0', 'id_data_source': '0', 'id_data_generator': '0', 'id_data_generator_version': '0', 'id_site': '0', 'site_long': '0.0', 'site_lat': '0.0'}

    #var = ','.join(str(tup[0])+'='+'"'+str(tup[1])+'"' for tup in tables[table]).strip('[]')
    #var = 'run_number="0",id_run_mode="0",id_first_event="3",first_event_time="0",id_last_event="3",last_event_time="3",id_data_source="0",id_data_generator="0",id_data_generator_version="0",id_site="0",site_long="0.0",site_lat="0.0"'
#    print(var)
    #var="run_number=0, run_mode=0, first_event=0, first_event_time=0"
    #container = dm.database().tables()['run'](run_number="0",id_run_mode="0",id_first_event="0",first_event_time="0",id_last_event="1064",last_event_time="0",id_data_source="detector",id_data_generator="GRANDlib",id_data_generator_version="0.1.0",id_site="MZS",site_long="0.0",site_lat="0.0",origin_geoid="[0. 0. 0.]")

    #works but ugly
    #container = eval("dm.database().tables()['run']("+var+")")

#    container = dm.database().tables()['run'](**var)
    #container = dm.database().tables()['run'](var)

#    dm.database().sqlalchemysession.add(container)
#    dm.database().sqlalchemysession.commit()


# ALTERNATIVE 1
# define official repo (incoming ?)
# identify provider -> create if necessary
# calculate new file name
# copy file in official repo with new name
# register file in file, file_location, provider

# ALTERNATIVE 2
# identify actual repo
# identify provider
# register file in file, file_location, provider


# print(rfile.TreeList['teventvoltage'])
# print(rfile.TreeList['trun'])
