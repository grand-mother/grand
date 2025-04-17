import time
from datetime import datetime

import psycopg2
import psycopg2.extras
from sshtunnel import SSHTunnelForwarder
import numpy
import grand.dataio.root_trees
import re
import granddb.rootdblib as rdb
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.inspection import inspect
import grand.manage_log as mlg
import os
from sqlalchemy import func

from granddb.rootdblib import Dataset, RootFile

logger = mlg.get_logger_for_script(__name__)
mlg.create_output_for_logger("debug", log_stdout=True)


def casttodb(value):
    #print(f'{type(value)} - {value}')
    if isinstance(value, numpy.str_):
        val = repr(value)
    elif isinstance(value, numpy.bool_):
        val = int(value)
    elif isinstance(value, numpy.uint32):
        val = int(value)
    elif isinstance(value, numpy.float32):
        val = float(value)
    elif isinstance(value, numpy.ndarray):
        if value.size == 0:
            val = None
        elif value.size == 1:
            val = casttodb(value.item())
        else:
            #val = value.tolist()
            val = [casttodb(item) for item in value]
    elif isinstance(value, grand.dataio.root_trees.StdVectorList):
        val =[]
        #postgres cannot store arrays of arrays... so we split (not sure if really correct)!
        for i in value:
            if isinstance(i,numpy.ndarray) or isinstance(i, grand.dataio.root_trees.StdVectorList):
                val.append(casttodb(i))
            else:
                val.append(casttodb(i))

        #value = [i for i in value]
    elif isinstance(value, str):
        val = value.strip().strip('\t').strip('\n')
    elif isinstance(value, datetime):
        val = value
    else:
        val = value
    return val


## @brief Class to handle the Grand database.
# A simple psycopg2 connexion (dbconnection) or a sqlalchemysession (sqlalchemysession) can be used
class Database:
    _host: str
    _port: int
    _dbname: str
    _user: str
    _passwd: str
    _sshserver: str
    _sshport: int
    _tables = {}
    dbconnection = None  # psycopg2 connect
    sqlalchemysession = None  # sqlalchemy session

    # _cred : Credentials

    def __init__(self, host, port, dbname, user, passwd, sshserv="", sshport=22, cred=None):
        self._host = host
        if port == "":
            self._port = 5432
        else:
            self._port = port
        self._dbname = dbname
        self._user = user
        self._passwd = passwd
        self._sshserv = sshserv
        if sshport == "":
            self._sshport = 22
        else:
            self._sshport = sshport
        self._cred = cred

        if self._sshserv != "" and self._cred is not None:
            # TODO: Check credentials for ssh tunnel and ask for passwds
            self.server = SSHTunnelForwarder(
                (self._sshserv, self.sshport()),
                ssh_username=self._cred.user(),
                ssh_pkey=self._cred.keyfile(),
                remote_bind_address=(self._host, self._port),
                allow_agent=True
            )
            self.server.start()
            local_port = str(self.server.local_bind_port)
            self._host = "127.0.0.1"
            self._port = local_port

        # self.connect()

        engine = create_engine(
            'postgresql+psycopg2://' + self.user() + ':' + self.passwd() + '@' + self.host() + ':' + str(
                self.port()) + '/' + self._dbname)
        Base = automap_base()

        Base.prepare(engine, reflect=True)
        self.sqlalchemysession = Session(engine,autoflush=False)
        #self.sqlalchemysession.no_autoflush = True
        inspection = inspect(engine)
        for table in inspection.get_table_names():
            # for table in engine.table_names(): #this is obsolete
            self._tables[table] = getattr(Base.classes, table)

    def __del__(self):
        # self.session.flush()
        # self.session.close()
        self.dbconnection.close()
        # self.server.stop(force=True)

    def connect(self):
        self.dbconnection = psycopg2.connect(
            host=self.host(),
            database=self.dbname(),
            port=self.port(),
            user=self.user(),
            password=self.passwd())

    def disconnect(self):
        self.dbconnection.close()

    def host(self):
        return self._host

    def port(self):
        return self._port

    def dbname(self):
        return self._dbname

    def user(self):
        return self._user

    def passwd(self):
        return self._passwd

    def sshserv(self):
        return self._sshserv

    def sshport(self):
        return self._sshport

    def cred(self):
        return self._cred

    def tables(self):
        return self._tables

    def select(self, query):
        try:
            self.connect()
            cursor = self.dbconnection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query)
            record = cursor.fetchall()
            cursor.close()
        except psycopg2.DatabaseError as e:
            logger.error(f"Error {e}")
        return record

    def execute_sql(self, query):
        try:
            res = True
            self.connect()
            cursor = self.dbconnection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query)
            self.dbconnection.commit()
            cursor.close()
        except psycopg2.DatabaseError as e:
            logger.error(f"Error {e}")
            res = False
        return res

    #    def insert(self, query):
    #        record = []
    #        try:
    #            cursor = self.dbconnection.cursor(cursor_factory=psycopg2.extras.DictCursor)
    #            cursor.execute(query)
    #            print(cursor.statusmessage)
    #            self.dbconnection.commit()
    #            record.append(cursor.fetchone()[0])
    #            cursor.close()
    #        except psycopg2.DatabaseError as e:
    #            print(f'Error {e}')
    #        return record
    #
    #    def insert2(self, query, values):
    #        record = []
    #        try:
    #            cursor = self.dbconnection.cursor(cursor_factory=psycopg2.extras.DictCursor)
    #            cursor.execute(query, values)
    #            print(cursor.statusmessage)
    #            self.dbconnection.commit()
    #            record.append(cursor.fetchone()[0])
    #            cursor.close()
    #        except psycopg2.DatabaseError as e:
    #            print(f'Error {e}')
    #        return record

    ## @brief Method to get the list of the repositories defined in the database.
    # Returns a dictionary with
    # repository - character varying - name of the repo
    # path - character varying - list of paths where files can be searched for
    # server - character varying - name or IP of the server,
    # port - integer - port to access the server
    # protocol - character varying - protocol name to access the server
    # id_repository - integer - id_repository
    def get_repos(self):
        record = None
        # Intergogation using simple psycopg2 query to directly get a dict
        query = "select * from get_repos()"
        record = self.select(str(query))
        return record

    ## Search a file in DB.
    # Search first file with provided filename. If not found, search file with original_name = filename
    # returns the filename and the different locations for it
    def SearchFile(self, filename):
        result = []
        file = self.sqlalchemysession.query(self.tables()['file'], self.tables()['file_location'],
                                            self.tables()['repository']) \
            .join(self.tables()['file_location'],
                  self.tables()['file_location'].id_file == self.tables()['file'].id_file) \
            .join(self.tables()['repository'],
                  self.tables()['repository'].id_repository == self.tables()['file_location'].id_repository) \
            .filter(self.tables()['file'].filename == filename) \
            .order_by(self.tables()['repository'].id_repository) \
            .all()

        if len(file) == 0:
            file = self.sqlalchemysession.query(self.tables()['file'], self.tables()['file_location'],
                                                self.tables()['repository']) \
                .join(self.tables()['file_location'],
                      self.tables()['file_location'].id_file == self.tables()['file'].id_file) \
                .join(self.tables()['repository'],
                      self.tables()['repository'].id_repository == self.tables()['file_location'].id_repository) \
                .filter(self.tables()['file'].original_name == filename) \
                .order_by(self.tables()['repository'].id_repository) \
                .all()

        for record in file:
            logger.debug(f"file {record.file.filename} found in repository {record.repository.repository} at path {record.file_location.path}")
            result.append([record.file.filename, record.repository.repository, record.file_location.path, record.file.id_file])
        return result

    ## @brief For parameter <param> of value <value> in table <table> this function will check if the param is a foreign key and if yes it will
    # search de corresponding id in the foreign table. If found, it will return it, if not, it will add the parameter in the foreign table
    # and return the id of the newly created record.
    def get_or_create_fk(self, table, param, value):
        idfk = None
        if value is not None and value != "":
            # Check if foreign key
            if getattr(self._tables[table], param).foreign_keys:
                # Get the foreign table and id in this table
                # ugly but couldn't find another way to do it !
                fk = re.findall(r'\'(.+)\.(.+)\'', str(list(getattr(self._tables[table], param).foreign_keys)[0]))
                fktable = fk[0][0]  # foreign table
                # fkfield = fk[0][1]  # id field in foreign table
                idfk = self.get_or_create_key(fktable, fktable, value, 'autoadd')
        return idfk

    ## @brief Search in table <table> if we have a record with <value> for field <field>.
    # If yes, returns id_<table>, if not create a record and return the id_<table> for this record
    def get_or_create_key(self, table, field, value, description=""):
        idfk = None
        if value is not None and value != "":
            filt = {}
            filt[field] = str(casttodb(value))
            ret = self.sqlalchemysession.query(getattr(self._tables[table], 'id_' + table)).filter_by(**filt).all()
            if len(ret) == 0:
                filt['description'] = description
                container = self.tables()[table](**filt)
                self.sqlalchemysession.add(container)
                self.sqlalchemysession.flush()
                idfk = int(getattr(container, 'id_' + table))
            else:
                idfk = int(ret[0][0])
        return idfk

    ## @brief Function to register a repository (if necessary) in the database.
    # Returns the id_repository of the corresponding repository
    def register_repository(self, name, protocol, port, server, path, description=""):
        # Check protocol
        id_protocol = self.get_or_create_key('protocol', 'protocol', protocol, description)
        self.sqlalchemysession.flush()
        # Check if repo exists
        repo = self.sqlalchemysession.query(self.tables()['repository']
                                                   ).filter_by(repository=name).first()
        if repo is None:
            repository={ 'repository': name, 'id_protocol': id_protocol, 'port': port,
                                 'server_name': server, 'paths': path, 'description': description, }
            container = self.tables()['repository'](**repository)
            self.sqlalchemysession.add(container)
            self.sqlalchemysession.flush()
            id_repository = int(getattr(container, 'id_repository'))
        else:
            #id_repository = self.get_or_create_key('repository', 'repository', name, description)
            if set(repo.paths) == set(path):
                pass
            else:
                repo.paths = list(set(path))
            id_repository = repo.id_repository
        self.sqlalchemysession.flush()
        self.sqlalchemysession.commit()
        return id_repository


    ## @brief Function to register (if necessary) a filename into the database.
    # It will first search if the file is already known in the DB and check the repository.
    # Returns the id_file for the file and a boolean True if the file was not previously in the DB (i.e it's a new file)
    # and false if the file was already registered. This is usefull to know if the metadata of the file needs to be read
    # or not
    def register_filename(self, filename, newfilename, dataset, id_repository, provider, targetfile=None):
        register_file = False
        isnewfile = False
        idfile = None
        id_dataset = None
        if targetfile is None:
            targetfile = newfilename
        if dataset is not None:
            id_dataset = self.get_or_create_key('dataset', 'dataset_name', os.path.basename(dataset))
            filt = {}
            filt['id_dataset'] = str(casttodb(id_dataset))
            filt['id_repository'] = str(casttodb(id_repository))
            ret = self.sqlalchemysession.query(getattr(self._tables['dataset_location'], 'id_dataset')).filter_by(
                **filt).all()
            if len(ret) == 0:
                dataset_path=os.path.dirname(targetfile)
                container = self.tables()['dataset_location'](id_dataset=id_dataset, id_repository=id_repository, path=dataset_path,
                                                              description="")
                self.sqlalchemysession.add(container)
                self.sqlalchemysession.flush()

        ## Check if file not already registered IN THIS REPO : IF YES, ABORT, IF NO REGISTER
        #First see if file is registered elsewhere
        file_exist = self.sqlalchemysession.query(self.tables()['file']).filter_by(
            filename=os.path.basename(targetfile),id_dataset=id_dataset).first()
        if file_exist is not None:
            #File exists somewhere... see if in the repository we want
            file_exist_here = self.sqlalchemysession.query(self.tables()['file_location']).filter_by(
                id_repository=id_repository, id_file=file_exist.id_file).first()
            if file_exist_here is None:
                # file exists but in a different repo. We only need to register it in the current repo
                register_file = True
                idfile = file_exist.id_file
        else:
            # File not registered
            register_file = True
            isnewfile = True

        ### Register the file
        if register_file:
            id_provider = self.get_or_create_key('provider', 'provider', provider)
            if isnewfile:
                rfile = rdb.RootFile(str(filename))
                #rfile.dataset_name()
                # rfile.file().GetSize()
                container = self.tables()['file'](id_dataset=id_dataset,
                                                  filename=os.path.basename(targetfile),
                                                  description='autodesc',
                                                  original_name=os.path.basename(filename),
                                                  id_provider=id_provider,
                                                  file_size=rfile.file.GetSize()
                                                  )
                self.sqlalchemysession.add(container)
                self.sqlalchemysession.flush()
                idfile = container.id_file
            # container = self.tables()['file_location'](id_file=idfile, id_repository=id_repository, path=os.path.dirname(newfilename))
            container = self.tables()['file_location'](id_file=idfile, id_repository=id_repository, path=targetfile,
                                                       description="")
            self.sqlalchemysession.add(container)
            logger.debug(f"File name {filename} registered")
            # self.sqlalchemysession.flush()

        return idfile, isnewfile, id_dataset




    ## @brief Function to register (if necessary) the content of a file into the database.
    # It will first read the file and walk along datas to determine what has to be registered
    def register_filecontent(self, file, idfile, id_dataset):
        # We store run_number-event_number list to avoid to record them twice in event table (and produce an error due to unicity).
        # Ugly but no other efficient way to do (checking in the DB before insertion is too time consuming).
        eventlist = []
        # ttrees will be a dict of trees to add. key is the tree name and value is a dict with all values for the tree.
        ttrees = {}
        # tables = {}
        rfile = rdb.RootFile(str(file))
        # We iterate over all trees
        for treename in rfile.TreeList:
            logger.debug(f" Debug reading tree {treename}")
            treetype = treename.split('_', 1)[0]
            # We register only known and identified trees defined in rootdblib
            if hasattr(rfile, treetype + "ToDB"):
                table = getattr(rfile, treetype + "ToDB").get('table')
                # table = getattr(rfile, treetype + "ToDB")['table']
                ttrees[treename] = {}

                # Get metadata and add file_content record
                metatree = {}
                tablemeta = "file_content"
                metatree['id_file'] = idfile
                for meta, field in rfile.metaToDB.items():
                    # try/except to avoid stopping when metadata is not present in root file
                    try:
                        value = casttodb(getattr(rfile.TreeList[treename], meta))
                        if field.find('id_') >= 0:
                            value = self.get_or_create_fk(tablemeta, field, value)
                        if field == "comment":
                            field = "comments"
                        metatree[field] = value
                        metatree['tree_name'] = treename
                        #Get the number of events for events trees (set to 0 for run trees)
                        if treetype in rfile.EventTrees:
                            metatree['number_of_events'] = rfile.TreeList[treename].get_number_of_entries()
                        else:
                            metatree['number_of_events'] = 0
                    except:
                        logger.debug(f" Debug : error on meta {meta} field {field} value {value} ")
                        pass
                # Trick to use "real" tree name (instead of meta _tree_name which is not always correct)
                metatree['tree_name'] = treename
                container = self.tables()[tablemeta](**metatree)
                self.sqlalchemysession.add(container)
                # self.sqlalchemysession.flush()
                # If table not defined in rootdblib for this tree then no content to record.
                st = time.time()

                #Lets see if events exists but from other files in same dataset
                #First get the dataset
                #dataset_result = self.sqlalchemysession.query(getattr(self._tables['file'], 'id_dataset')).filter(self._tables['file'].id_file == idfile).first()
                #id_dataset = dataset_result[0]
                if id_dataset is None:
                    pass
                else:
                    # Get events corresponding to files in the same dataset
                    eventlist = (self.sqlalchemysession.query(getattr(self._tables['events'], 'run_number'),
                                                              getattr(self._tables['events'], 'event_number'))
                                 .distinct(getattr(self._tables['events'], 'run_number'),getattr(self._tables['events'], 'event_number'))
                                 .join(self._tables['file'])
                                 .filter(self._tables['file'].id_dataset == id_dataset).all())
                    #print(f'{file} {idfile} {len(eventlist)}')



                    #events.run_number, events.event_number).join(file).filter(File.id_dataset == id_dataset).all()

                if table is not None:
                    # Registering of events trees
                    if treetype in rfile.EventTrees:
                        # For events we iterates over event_number and run_number
                        for event, run in rfile.TreeList[treename].get_list_of_events():
                            # NEED TO CHECK THAT INFOS NOT ALREADY PRESENT IN DB FROM ANOTHER FILE IN SAME DATASET

                            if ((table != "events") or ([run, event] not in eventlist)):
                                if table == "events":
                                    eventlist.append([run, event])

                                if not (run, event) in ttrees[treename]:
                                    ttrees[treename][(run, event)] = {}
                                rfile.TreeList[treename].get_event(event, run)
                                for param, field in getattr(rfile, treetype + "ToDB").items():
                                    if param != "table":
                                        value = casttodb(getattr(rfile.TreeList[treename], param))
                                        # Il foreign key (i.e. starts with id_) then register value in foreign table and return the key instead of value
                                        if field.startswith('id_'):
                                            value = self.get_or_create_fk(table, field, value)
                                        ttrees[treename][(run, event)][field] = value
                                    else:
                                        # TODO: Change id_file and tree_name into arrays and add values
                                        ttrees[treename][(run, event)]['id_file'] = idfile
                                        ttrees[treename][(run, event)]['tree_name'] = treename

                                container = self.tables()[table](**ttrees[treename][(run, event)])
                                self.sqlalchemysession.add(container)

                            # if table =="events":
                            #    if [run,event] not in eventlist:
                            #        eventlist.append([run,event])
                            #        self.sqlalchemysession.add(container)

                            # else:
                            #    self.sqlalchemysession.add(container)

                            # try:
                            #    self.sqlalchemysession.add(container)
                            #    self.sqlalchemysession.flush()
                            # except :
                            #    print("error 1")

                            # self.sqlalchemysession.add(container)
                            # self.sqlalchemysession.flush()
                            # filt = {}
                            # filt["run_number"] = str(casttodb(run))
                            # filt["event_number"] = str(casttodb(event))
                            # filt["id_file"] = str(casttodb(idfile))
                            # ret = self.sqlalchemysession.query(self._tables[table]).filter_by(**filt).exists()
                            # if ret == 0 :
                            #    self.sqlalchemysession.add(container)
                            # else:
                            #    print("UPDATE ?")
                        # self.sqlalchemysession.flush()
                        # print(container.id_treename)
                        # idtree = "id_"+treename

                    # For runs we iterates over run_number
                    elif treename in rfile.RunTrees:
                        for run in rfile.TreeList[treename].get_list_of_runs():
                            if run not in ttrees[treename]:
                                ttrees[treename][run] = {}

                            rfile.TreeList[treename].get_run(run)
                            for param, field in getattr(rfile, treename + "ToDB").items():
                                if param != "table":
                                    try:
                                        value = casttodb(getattr(rfile.TreeList[treename], param))
                                        # Il foreign key (i.e. starts with id_) then register value in foreign table and return the key instead of value
                                        if field.startswith('id_'):
                                            value = self.get_or_create_fk(table, field, value)
                                        ttrees[treename][run][field] = value
                                    except:
                                        logger.warning(
                                            f"Error in getting {param} for {rfile.TreeList[treename].__class__.__name__}")
                                else:
                                    ttrees[treename][run]['id_file'] = idfile
                                    ttrees[treename][run]['tree_name'] = treename

                            container = self.tables()[table](**ttrees[treename][run])
                            self.sqlalchemysession.add(container)
                        # self.sqlalchemysession.flush()

                        # print(container.id_treename)
                        # idtree = "id_"+treename
                et = time.time()
                elapsed_time = et - st
                # print('Execution time:', elapsed_time, 'seconds')
                logger.debug(f"execution time {elapsed_time} seconds")

    ## @brief Function to register a file into the database.
    def register_file(self, orgfilename, newfilename, dataset, id_repository, provider, targetdir=None):
        idfile, read_file, id_dataset = self.register_filename(orgfilename, newfilename, dataset, id_repository, provider, targetdir)
        if read_file:
            # We read the localfile and not the remote one
            self.register_filecontent(orgfilename, idfile, id_dataset)
            # self.register_filecontent(newfilename,idfile)
        else:
            logger.info(f"file {orgfilename} already registered.")
        self.sqlalchemysession.commit()

    ## @brief Function to register a file which is already registered into the database.
    # It will first search the registered file and will remove it from the database before registering it again as a new file
    # Usefull when reprocessing or correcting a file
    def register_again_file(self, orgfilename, newfilename, dataset, id_repository, provider, targetfile=None):
        if targetfile is None:
            targetfile = newfilename
        if dataset is not None:
            id_dataset = self.get_or_create_key('dataset', 'dataset_name', os.path.basename(dataset))
        else:
            id_dataset = None
        file_exist = self.sqlalchemysession.query(self.tables()['file']).filter_by(
            filename=os.path.basename(targetfile),id_dataset=id_dataset).first()
        if file_exist is not None:
            idfile = file_exist.id_file

            removed = self.sqlalchemysession.query(func.delete_file_id(idfile)).all()
            logger.info(f"removed old files {removed}")
        idfile, read_file, id_dataset = self.register_filename(orgfilename, newfilename, dataset, id_repository, provider, targetfile)
        if read_file:
            # We read the localfile and not the remote one
            self.register_filecontent(orgfilename, idfile, id_dataset)
            # self.register_filecontent(newfilename,idfile)
        else:
            logger.info(f"file {orgfilename} already registered.")
        self.sqlalchemysession.commit()


    def register_dataset(self, directory, id_repository, provider):
        # Open the datadir
        #datadir=grand.dataio.root_trees.DataDirectory(os.path.normpath(directory))
        dataset=Dataset(os.path.normpath(directory))
        self.register_dataset_name(dataset, id_repository)
        self.register_dataset_content(dataset, id_repository,provider)
        self.sqlalchemysession.commit()


    def register_dataset_name(self, dataset, id_repository):
        #Search if dataset already exists
        ret = self.sqlalchemysession.query(getattr(self._tables['dataset'], 'id_dataset')).filter_by(dataset_name=dataset.dataset_name).all()
        if len(ret)==0:
            #Dataset does not exists. We create it
            container = self.tables()['dataset'](dataset_name=dataset.dataset_name, original_name=dataset.dataset_original_name,description=dataset.comment)
            self.sqlalchemysession.add(container)
            self.sqlalchemysession.flush()
            dataset.id_dataset = int(getattr(container, 'id_dataset'))
            logger.info(f"dataset {dataset.dataset_name} from {dataset.dataset_original_name} registered with id_dataset {dataset.id_dataset}.")
        else:
            #dataset exists : Get it
            dataset.id_dataset = int(ret[0][0])

        #Search if dataset registered in that repo
        filt = {}
        filt['id_dataset'] = str(casttodb(dataset.id_dataset))
        filt['id_repository'] = str(casttodb(id_repository))
        ret = self.sqlalchemysession.query(getattr(self._tables['dataset_location'], 'id_dataset')).filter_by(**filt).all()
        # If dataset not registered in the repository then register it
        if len(ret) == 0:
            dataset_path=os.path.normpath(dataset.full_path)
            container = self.tables()['dataset_location'](id_dataset=dataset.id_dataset,
                                                          id_repository=id_repository,
                                                          path=dataset_path,
                                                          description="")
            self.sqlalchemysession.add(container)
            self.sqlalchemysession.flush()

    def register_dataset_content(self, dataset, id_repository,provider):
        for file in dataset.get_list_of_files():
            logger.info(f"registering {file}")
            file_exist = self.sqlalchemysession.query(self.tables()['file']).filter_by(
                                                        filename=os.path.basename(file),
                                                        id_dataset=dataset.id_dataset).first()
            if file_exist is not None:
                idfile = file_exist.id_file
                removed = self.sqlalchemysession.query(func.delete_file_id(idfile)).all()
                logger.info(f"removed old files {removed}")
            idfile = self.new_register_filename(file, dataset, id_repository,provider)
            self.new_register_filecontent(file, idfile, dataset)
            #TODO: Register file content



    ## @brief Function to register (if necessary) a filename in a dataset into the database.
    # It will first search if the file is already known in the DB and check the repository.
    # Returns the id_file for the file and a boolean True if the file was not previously in the DB (i.e it's a new file)
    # and false if the file was already registered. This is usefull to know if the metadata of the file needs to be read
    # or not
    def new_register_filename(self, filename, dataset, id_repository, provider):
        id_provider = self.get_or_create_key('provider', 'provider', provider)
        size_bytes = os.path.getsize(filename)
        container = self.tables()['file'](id_dataset=dataset.id_dataset,
                                                  filename=os.path.basename(filename),
                                                  description='autodesc',
                                                  original_name=dataset.dataset_original_name,
                                                  id_provider=id_provider,
                                                  file_size=size_bytes
                                                  )
        self.sqlalchemysession.add(container)
        self.sqlalchemysession.flush()
        idfile = container.id_file
        container = self.tables()['file_location'](id_file=idfile, id_repository=id_repository, path=filename,
                                                       description="")
        self.sqlalchemysession.add(container)
        logger.debug(f"File name {filename} registered")
        return idfile




    ## @brief Function to register (if necessary) the content of a file into the database.
    # It will first read the file and walk along datas to determine what has to be registered
    def new_register_filecontent(self, file, idfile, dataset):
        # We store run_number-event_number list to avoid to record them twice in event table (and produce an error due to unicity).
        # Ugly but no other efficient way to do (checking in the DB before insertion is too time consuming).
        eventlist = []
        # ttrees will be a dict of trees to add. key is the tree name and value is a dict with all values for the tree.
        ttrees = {}
        # tables = {}
        #rfile = grand.dataio.root_trees.DataFile(str(file))
        rfile = RootFile(str(file))

        #-----
#        for treename in rfile.file.dict_of_trees.keys():
#            #treetype = rfile.file.get_tree_info(treename)["type"]
#            #print(f'tree {treename} tree type {rfile.file.get_tree_info(treename)["type"]}')
#            if hasattr(rfile, treename + "ToDB"):
#                table = getattr(rfile, treename + "ToDB").get('table')
#                #print(f'table is {table}')
#                #print(f'rfile.file.dict_of_trees is {rfile.file.dict_of_trees}')
#                #print(f'rfile.file.tree_types is {rfile.file.tree_types}')
#                #print(f'tree type is {type(rfile.file.dict_of_trees[treename])} rfile.TreeList[treename] is of type {type(rfile.TreeList[treename])}')
#                #for key, value in rfile.file.tree_types.items():
#                #    #print(f'key is {key} and value is {value}')
#                #    class_to_instantiate = getattr(grand.dataio.root_trees, key)
#                #    obj = class_to_instantiate(file)
#                #    #print(f'obj type is {type(obj)}')
#            #    class_to_instantiate = getattr(grand.dataio.root_trees, treetype)
#            #    obj = class_to_instantiate(file)
#                if treename in rfile.EventTrees:
#                    obj=rfile.get_tree(treename)
#                    for event, run in obj.get_list_of_events():
#                        print(f'run is {run} event is {event}')
#        return 0
#        #-----


        tablemeta = "file_content"

        # We iterate over all trees
        for treename in rfile.file.dict_of_trees.keys():
            logger.debug(f" Debug reading tree {treename}")
            #treetype = treename.split('_', 1)[0]
            treetype = rfile.file.get_tree_info(treename)["type"].lower()
            #treetype = treename
            metatree = {}

            # we first register meta info for all trees
            for meta, field in rfile.metaToDB.items():
                # try/except to avoid stopping when metadata is not present in root file
                try:
                    rawvalue = rfile.file.get_tree_info(treename)[meta]
                    # skip if datetime is not correct type
                    if (field == "source_datetime" or field == "creation_datetime") and type(rawvalue) is not datetime:
                        logger.debug(f'value: {rawvalue} has wrong type {type(rawvalue)} for field {field}')
                        pass
                    else:
                        value = casttodb(rawvalue)
                        metatree[field] = value

                    # If foreign key then replace value by the corresponding key
                    if field.find('id_') >= 0:
                        value = self.get_or_create_fk(tablemeta, field, rawvalue)
                        metatree[field] = value

                    #metatree['tree_name'] = treename
                except:
                    logger.debug(f" Debug : error on meta {meta} field {field} value {value} ")
                    pass

            # metatree['number_of_events'] = rfile.file.get_tree_info(treename)['evt_cnt']
            metatree['id_file'] = idfile
            # Trick to use "real" tree name (instead of meta _tree_name which is not always correct)
            metatree['tree_name'] = treename
            container = self.tables()[tablemeta](**metatree)
            self.sqlalchemysession.add(container)
            self.sqlalchemysession.flush()


            # We register the content only for known and identified trees defined in rootdblib
            if hasattr(rfile, treetype + "ToDB"):
                # Determine in which table of the database infos should go
                table = getattr(rfile, treetype + "ToDB").get('table')
                # table = getattr(rfile, treetype + "ToDB")['table']
                ttrees[treename] = {}
                id_dataset = dataset.id_dataset
                st = time.time()

                if table is not None:
                    # Registering of events trees
                    treeobject = rfile.get_tree(treename)
                    if treetype in rfile.EventTrees:
                        # For events we iterates over event_number and run_number
                        #for event, run in rfile.TreeList[treename].get_list_of_events():
                        for event, run in treeobject.get_list_of_events():
                            # NEED TO CHECK THAT INFOS NOT ALREADY PRESENT IN DB FROM ANOTHER FILE IN SAME DATASET

                            if ((table != "events") or ([run, event] not in eventlist)):
                                if table == "events":
                                    eventlist.append([run, event])

                                if not (run, event) in ttrees[treename]:
                                    ttrees[treename][(run, event)] = {}
                                #rfile.TreeList[treename].get_event(event, run)
                                treeobject.get_event(event, run)
                                for param, field in getattr(rfile, treetype + "ToDB").items():
                                    if param != "table":
                                        #value = casttodb(getattr(rfile.TreeList[treename], param))
                                        value = casttodb(getattr(treeobject, param))
                                        # Il foreign key (i.e. starts with id_) then register value in foreign table and return the key instead of value
                                        if field.startswith('id_'):
                                            value = self.get_or_create_fk(table, field, value)
                                        ttrees[treename][(run, event)][field] = value
                                    else:
                                        ttrees[treename][(run, event)]['id_file'] = idfile
                                        ttrees[treename][(run, event)]['tree_name'] = treename

                                container = self.tables()[table](**ttrees[treename][(run, event)])
                                self.sqlalchemysession.add(container)



                    # For runs we iterates over run_number
                    elif treename in rfile.RunTrees:
                        #for run in rfile.TreeList[treename].get_list_of_runs():
                        for run in treeobject.get_list_of_runs():
                            if run not in ttrees[treename]:
                                ttrees[treename][run] = {}

                            #rfile.TreeList[treename].get_run(run)
                            treeobject.get_run(run)
                            for param, field in getattr(rfile, treetype + "ToDB").items():
                                if param != "table":
                                    try:
                                        #value = casttodb(getattr(rfile.TreeList[treename], param))
                                        value = casttodb(getattr(treeobject, param))
                                        # Il foreign key (i.e. starts with id_) then register value in foreign table and return the key instead of value
                                        if field.startswith('id_'):
                                            value = self.get_or_create_fk(table, field, value)
                                        ttrees[treename][run][field] = value
                                    except:
                                        #logger.warning(
                                        #    f"Error in getting {param} for {rfile.TreeList[treename].__class__.__name__}")
                                        logger.warning(
                                            f"Error in getting {param} for {treeobject.__class__.__name__}")
                                else:
                                    ttrees[treename][run]['id_file'] = idfile
                                    ttrees[treename][run]['tree_name'] = treename

                            container = self.tables()[table](**ttrees[treename][run])
                            self.sqlalchemysession.add(container)
                        # self.sqlalchemysession.flush()
                    treeobject.stop_using()
                et = time.time()
                elapsed_time = et - st
                logger.debug(f"execution time {elapsed_time} seconds")
