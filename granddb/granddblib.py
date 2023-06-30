import psycopg2
import psycopg2.extras
from sshtunnel import SSHTunnelForwarder
import numpy
import grand.io.root_trees
import re
import granddb.rootdblib as rdb
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import func
from sqlalchemy.inspection import inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import postgresql
import grand.manage_log as mlg

logger = mlg.get_logger_for_script(__name__)
mlg.create_output_for_logger("debug", log_stdout=False)




def casttodb(value):
    if isinstance(value, numpy.uint32):
        value = int(value)
    if isinstance(value, numpy.float32):
        value = float(value)
    if isinstance(value, numpy.ndarray):
        if value.size == 0:
            value = None
        elif value.size == 1:
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, grand.io.root_trees.StdVectorList):
        value = [i for i in value]
    if isinstance(value, str):
        value = value.strip().strip('\t').strip('\n')
    return value


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
            #TODO: Check credentials for ssh tunnel and ask for passwds
            self.server = SSHTunnelForwarder(
                (self._sshserv, self.sshport()),
                ssh_username=self._cred.user(),
                ssh_pkey=self._cred.keyfile(),
                remote_bind_address=(self._host, self._port)
            )
            self.server.start()
            local_port = str(self.server.local_bind_port)
            self._host = "127.0.0.1"
            self._port = local_port

        #self.connect()

        engine = create_engine(
            'postgresql+psycopg2://' + self.user() + ':' + self.passwd() + '@' + self.host() + ':' + str(self.port()) + '/' + self._dbname)
        Base = automap_base()

        Base.prepare(engine, reflect=True)
        self.sqlalchemysession = Session(engine)
        for table in engine.table_names():
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
        file = self.sqlalchemysession.query(self.tables()['file'], self.tables()['file_location'], self.tables()['repository'])\
            .join(self.tables()['file_location'], self.tables()['file_location'].id_file==self.tables()['file'].id_file) \
            .join(self.tables()['repository'],self.tables()['repository'].id_repository==self.tables()['file_location'].id_repository) \
            .filter(self.tables()['file'].filename == filename)\
            .order_by(self.tables()['repository'].id_repository)\
            .all()

        if len(file) == 0:
            file = self.sqlalchemysession.query(self.tables()['file'], self.tables()['file_location'], self.tables()['repository'])\
                .join(self.tables()['file_location'], self.tables()['file_location'].id_file==self.tables()['file'].id_file) \
                .join(self.tables()['repository'],self.tables()['repository'].id_repository==self.tables()['file_location'].id_repository) \
                .filter(self.tables()['file'].original_name == filename)\
                .order_by(self.tables()['repository'].id_repository)\
                .all()

        for record in file:
            logger.debug(f"file {record.file.filename} found in repository {record.repository.repository}")
            result.append([record.file.filename, record.repository.repository])
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
        savepoint = self.sqlalchemysession.begin_nested()
        id_protocol = self.get_or_create_key('protocol', 'protocol', protocol, description)
        id_repository = self.get_or_create_key('repository', 'repository', name, description)
        self.sqlalchemysession.flush()
        # Check if repository access exists or not !
        repo_access = self.sqlalchemysession.query(self.tables()['repository_access']
                                                   ).filter_by(id_repository=id_repository,
                                                               id_protocol=id_protocol).first()
        if repo_access is not None:
            if set(repo_access.path) == set(path):
                pass
            else:
                repo_access.path = path
        else:
            repository_access = {'id_repository': id_repository, 'id_protocol': id_protocol, 'port': port,
                                 'server_name': server, 'path': path}
            container = self.tables()['repository_access'](**repository_access)
            self.sqlalchemysession.add(container)
            self.sqlalchemysession.flush()

        savepoint.commit()
        return id_repository

    ## @brief Function to register (if necessary) a filename into the database.
    # It will first search if the file is already known in the DB and check the repository.
    # Returns the id_file for the file and a boolean True if the file was not previously in the DB (i.e it's a new file)
    # and false if the file was already registered. This is usefull to know if the metadata of the file needs to be read
    # or not
    def register_filename(self, filename, newfilename, id_repository, provider):
        import os
        register_file = False
        isnewfile = False
        idfile = None
        ## Check if file not already registered IN THIS REPO : IF YES, ABORT, IF NO REGISTER
        file_exist = self.sqlalchemysession.query(self.tables()['file']).filter_by(
            filename=os.path.basename(newfilename)).first()
        if file_exist is not None:
            #file_exist_here = self.sqlalchemysession.query(self.tables()['file_location']).filter_by(
            #    id_repository=id_repository).first()
            file_exist_here = self.sqlalchemysession.query(self.tables()['file_location']).filter_by(
                id_repository=id_repository,path=os.path.dirname(newfilename)).first()
            if file_exist_here is None:
                # file exists in different repo. We only need to register it in the current repo
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
                container = self.tables()['file'](filename=os.path.basename(newfilename),
                                                           description='ceci est un fichier',
                                                           original_name=os.path.basename(filename),
                                                           id_provider=id_provider)
                self.sqlalchemysession.add(container)
                self.sqlalchemysession.flush()
                idfile = container.id_file
            container = self.tables()['file_location'](id_file=idfile, id_repository=id_repository, path=os.path.dirname(newfilename))
            self.sqlalchemysession.add(container)
            self.sqlalchemysession.flush()
        return idfile, isnewfile

    ## @brief Function to register (if necessary) the content of a file into the database.
    # It will first read the file and walk along datas to determine what has to be registered
    def register_filecontent(self, file, idfile):
        tables = {}
        rfile = rdb.RootFile(str(file))

        for treename in rfile.TreeList:
            table = getattr(rfile, treename + "ToDB")['table']
            if table not in tables:
                tables[table] = {}

            # For events we iterates over event_number and run_number "teventshowerzhaires"-> pb: previously in event, but now in run !
            if treename in ["teventefield", "teventshowersimdata",  "teventshower","teventvoltage",
                            "tadc","trawvoltage","tvoltage","tefield","tshower","tshowersim"]:
                for event, run in rfile.TreeList[treename].get_list_of_events():
                    if not (run, event) in tables[table]:
                        tables[table][(run, event)] = {}

                    rfile.TreeList[treename].get_event(event, run)
                    for param, field in getattr(rfile, treename + "ToDB").items():
                        if param != "table":
                            value = casttodb(getattr(rfile.TreeList[treename], param))
                            if field.find('id_') >= 0:
                                value = self.get_or_create_fk('event', field, value)

                            tables[table][(run, event)][field] = value

                            #if param == "du_id":
                            #    print("run=" + str(run) + "event=" + str(event) + param + str(getattr(rfile.TreeList[treename], param)))
                            #    tables[table][(run, event)].setdefault(field,[]).append(value)
                            #else:
                            #    tables[table][(run, event)][field] = value

            # For runs we iterates over run_number
            elif treename in ["trun", "trunefieldsimdata","trunvoltage","trunefieldsim","trunshowersim","trunnoise"]:
                for run in rfile.TreeList[treename].get_list_of_runs():
                    if run not in tables[table]:
                        tables[table][run] = {}
                    rfile.TreeList[treename].get_run(run)
                    for param, field in getattr(rfile, treename + "ToDB").items():
                        if param != "table":
                            try:
                                value = casttodb(getattr(rfile.TreeList[treename], param))
                                if field.find('id_') >= 0:
                                    value = self.get_or_create_fk('run', field, value)
                                tables[table][run][field] = value
                            except:
                                logger.warning(f"Error in getting {param} for {rfile.TreeList[treename].__class__.__name__}")

        # insert runs first, get id_run and update events before inserting event !
        for r in tables['run']:
            container = self.tables()['run'](**tables['run'][r])
            self.sqlalchemysession.add(container)
            self.sqlalchemysession.flush()
            # update id_run in events
            novalidevents = []
            for e in tables['event']:
                if e[0] == int(container.run_number):
                    tables['event'][e]['id_run'] = container.id_run
                else:
                    # event has no run associated !
                    # We will not register the event and have to remove this event from the list
                    print("no valid")
                    novalidevents.append(e)
            # We will not register the events with no run and have to remove them from the list !
            # But maybe better to let the program crash (thus comment the next two lines) !!!
            # for e in novalidevents:
            #    del tables['event'][e]

        for e in tables['event']:
            container = self.tables()['event'](**tables['event'][e])
            self.sqlalchemysession.add(container)
            self.sqlalchemysession.flush()
            tables['event'][e]['id_event'] = container.id_event

        ## Add file contains
        for e in tables['event']:
            container = self.tables()['file_contains'](id_file=idfile, id_run=tables['event'][e]['id_run'],
                                                                id_event=tables['event'][e]['id_event'])
            self.sqlalchemysession.add(container)
            self.sqlalchemysession.flush()

        # What if runs without events ? Maybe we should add it to file_contains ? But id_event is primary_key !
        # So the next lines cannot work !
        # eventsruns = list(set(tup[0] for tup in [*tables['event']]))
        # for r in tables['run']:
        #    if tables['run'][r]['id_run'] not in eventsruns:
        #        container = dm.database().tables()['file_contains'](id_file=idfile, id_run=tables['run'][r]['id_run'], id_event=None)
        #        dm.database().sqlalchemysession.add(container)
        #        dm.database().sqlalchemysession.flush()

    def register_file(self, orgfilename, newfilename, id_repository, provider):
        idfile, read_file = self.register_filename(orgfilename, newfilename, id_repository, provider)


        if read_file:
            #We read the localfile and not the remote one
            self.register_filecontent(orgfilename,idfile)
            #self.register_filecontent(newfilename,idfile)
        self.sqlalchemysession.commit()
