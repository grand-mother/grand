import psycopg2
import psycopg2.extras
from sshtunnel import SSHTunnelForwarder

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import func
from sqlalchemy.inspection import inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import postgresql

class Database:
    _host: str
    _port: int
    _dbname: str
    _user: str
    _passwd: str
    _sshserver: str
    _sshport: int
    _tables = {}
    dbconnection = None #psycopg2 connect
    sqlalchemysession = None #sqlalchemy session
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
            self.server = SSHTunnelForwarder(
                (self._sshserv, self._sshport),
                ssh_username=self._cred.user(),
                ssh_pkey=self._cred.keyfile(),
                remote_bind_address=(self._host, self._port)
            )
            self.server.start()
            local_port = str(self.server.local_bind_port)
            self._host = "127.0.0.1"
            self._port = local_port

        self.connect()

        engine = create_engine(
            'postgresql+psycopg2://' + self.user() + ':' + self.passwd() + '@' + self.host() + ':' + self.port() + '/' + self._dbname)
        Base = automap_base()

        Base.prepare(engine, reflect=True)
        self.sqlalchemysession = Session(engine)
        for table in engine.table_names():
            self._tables[table] = getattr(Base.classes, table)

    def __del__(self):
        #self.session.flush()
        #self.session.close()
        self.dbconnection.close()
        #self.server.stop(force=True)

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
            cursor = self.dbconnection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query)
            record = cursor.fetchall()
            cursor.close()
        except psycopg2.DatabaseError as e:
            print(f'Error {e}')
        return record

    def insert(self, query):
        record = []
        try:
            cursor = self.dbconnection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query)
            print(cursor.statusmessage)
            self.dbconnection.commit()
            record.append(cursor.fetchone()[0])
            cursor.close()
        except psycopg2.DatabaseError as e:
            print(f'Error {e}')
        return record

    def insert2(self, query, values):
        record = []
        try:
            cursor = self.dbconnection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query, values)
            print(cursor.statusmessage)
            self.dbconnection.commit()
            record.append(cursor.fetchone()[0])
            cursor.close()
        except psycopg2.DatabaseError as e:
            print(f'Error {e}')
        return record
    def get_repos(self):
        record = None
        # Intergogation using simple psycopg2 query
        query = "select * from get_repos()"
        record = self.select(str(query))

        # interrogation using sqlalchemy -> shitty because returns list and not dict
        # long way (sqlalchemy construction)
        # query = self.sqlalchemysession.query(self._tables['repository'].name.label("name"),
        #                        self._tables['repository_access'].path.label("path"),
        #                        self._tables['repository_access'].server_name.label("server"),
        #                        self._tables['repository_access'].port.label("port"),
        #                        self._tables['protocol'].name.label("protocol"),
        #                        self._tables['repository'],
        #                        self._tables['protocol'])\
        #    .join(self._tables['repository'])\
        #    .join(self._tables['protocol'])
        # short way using stored function
        #query = self.sqlalchemysession.query(func.get_repos())

        #record = query.all()

        return record
