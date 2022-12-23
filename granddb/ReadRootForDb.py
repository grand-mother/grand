import array

import granddb.granddblib
import rootdblib as rdb
import os.path
from granddatalib import DataManager

# from collections import defaultdict
provider = "Francois"

tables = {}
paramvalue = {}

dm = DataManager()
filename = 'Coarse2.root'
# filename = 'Coarse2_xmax_add.root'
filename = 'Coarse3.root'
# filename = 'Coarse4.root'
#filename = '_Filter_EfieldVSignal_LST18_XDS_Stshp_0.117_22.8_0.0_vertical_radius5325.58_100resamples.root'
# filename = '_FilterNewIntepolation_EfieldVSignal_Iron_MZS_QGSP_3.98_79.6_180.0.root'
# filename = '_Filter_EfieldVSignal_LST18_Proton_MZS_3.98_79.6_0.0.root'


file = dm.get(filename)

if file is not None:
    #register_file = False
    #read_file = False
    #idfile = None
    newfilename = dm.referer().copy(file)

    idfile, read_file = dm.database().register_file(filename, newfilename, dm.referer().id_repository, provider)

#    ## Check if file not already registered IN THIS REPO : IF YES, ABORT, IF NO REGISTER
#    file_exist = dm.database().sqlalchemysession.query(dm.database().tables()['file']).filter_by(
#        filename=os.path.basename(newfilename)).first()
#    if file_exist is not None:
#        file_exist_here = dm.database().sqlalchemysession.query(dm.database().tables()['file_location']).filter_by(
#            id_repository=dm.referer().id_repository).first()
#        if file_exist_here is None:
#            # file exists in different repo. We only need to register it in the current repo
#            register_file = True
#            idfile = file_exist.id_file
#    else:
#        # File not registered
#        register_file = True
#        read_file = True
#
#    ### Register the file
#    if register_file:
#        id_provider = dm.database().get_or_create_key('provider', 'provider', provider)
#        if read_file:
#            container = dm.database().tables()['file'](filename=os.path.basename(newfilename),
#                                                       description='ceci est un fichier',
#                                                       original_name=os.path.basename(filename), id_provider=id_provider)
#            dm.database().sqlalchemysession.add(container)
#            dm.database().sqlalchemysession.flush()
#            idfile = container.id_file
#        container = dm.database().tables()['file_location'](id_file=idfile, id_repository=dm.referer().id_repository)
#        dm.database().sqlalchemysession.add(container)

    if read_file:
        rfile = rdb.RootFile(file)

        for treename in rfile.TreeList:
            table = getattr(rfile, treename + "ToDB")['table']
            if table not in tables:
                tables[table] = {}
            # For events we iterates over event_number and run_number
            if treename in ["teventefield", "teventshowersimdata", "teventshowerzhaires", 'teventshower', 'teventvoltage']:
                for event, run in rfile.TreeList[treename].get_list_of_events():
                    if not (run, event) in tables[table]:
                        tables[table][(run, event)] = {}
                    rfile.TreeList[treename].get_event(event, run)
                    for param, field in getattr(rfile, treename + "ToDB").items():
                        if param != "table":
                            value = granddb.granddblib.casttodb(getattr(rfile.TreeList[treename], param))
                            if field.find('id_') >= 0:
                                value = dm.database().get_or_create_fk('event', field, value)
                            tables[table][(run, event)][field] = value

            # For runs we iterates over run_number
            elif treename in ["trun", "trunefieldsimdata"]:
                for run in rfile.TreeList[treename].get_list_of_runs():
                    if run not in tables[table]:
                        tables[table][run] = {}
                    rfile.TreeList[treename].get_run(run)
                    for param, field in getattr(rfile, treename + "ToDB").items():
                        if param != "table":
                            value = granddb.granddblib.casttodb(getattr(rfile.TreeList[treename], param))
                            if field.find('id_') >= 0:
                                value = dm.database().get_or_create_fk('run', field, value)
                            tables[table][run][field] = value

        # insert runs first, get id_run and update events before inserting event !
        for r in tables['run']:
            container = dm.database().tables()['run'](**tables['run'][r])
            dm.database().sqlalchemysession.add(container)
            dm.database().sqlalchemysession.flush()
            # update id_run in events
            novalidevents = []
            for e in tables['event']:
                if e[0] == int(container.run_number):
                    tables['event'][e]['id_run'] = container.id_run
                else:
                    #event has no run associated !
                    # We will not register the event and have to remove this event from the list
                    novalidevents.append(e)
            # We will not register the events with no run and have to remove them from the list !
            # But maybe better to let the program crash (thus comment the next two lines) !!!
            #for e in novalidevents:
            #    del tables['event'][e]

        for e in tables['event']:
            container = dm.database().tables()['event'](**tables['event'][e])
            dm.database().sqlalchemysession.add(container)
            dm.database().sqlalchemysession.flush()
            tables['event'][e]['id_event'] = container.id_event

        ## Add file contains
        for e in tables['event']:
            container = dm.database().tables()['file_contains'](id_file=idfile, id_run=tables['event'][e]['id_run'],
                                                                id_event=tables['event'][e]['id_event'])
            dm.database().sqlalchemysession.add(container)
            dm.database().sqlalchemysession.flush()

        # What if runs without events ? Maybe we should add it to file_contains ? But id_event is primary_key !
        # So the next lines cannot work !
        #eventsruns = list(set(tup[0] for tup in [*tables['event']]))
        #for r in tables['run']:
        #    if tables['run'][r]['id_run'] not in eventsruns:
        #        container = dm.database().tables()['file_contains'](id_file=idfile, id_run=tables['run'][r]['id_run'], id_event=None)
        #        dm.database().sqlalchemysession.add(container)
        #        dm.database().sqlalchemysession.flush()

# Finally commit all changes
dm.database().sqlalchemysession.commit()


