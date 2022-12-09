from pathlib import Path
import scp
import paramiko
import json
from configparser import ConfigParser
import urllib.request
# import urllib.error
import psycopg2
import psycopg2.extras
import ast
from sshtunnel import SSHTunnelForwarder

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.inspection import inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import postgresql


## @brief Class for managing datas.
# It will read an inifile where localdirs and different datasources are defined.
# The first localdir will be used as an incoming directory (i.e. all files searched in distant location and found will
# be copied in this incoming directory.
# The inifile will have the following structure :
# @verbatim
# [directories]
# localdir = ["./incoming/", "/some/directory/" , "/some/other/directory ]
# [repositories]
# Repo1 = ["protocol","server",port,["/directory1/", "/other/dir/"]]
# [credentials]
# Repo1 = ["user","password","keyfile","keypasswd"]
# [database]
# localdb = ["host", port, "dbname", "user", "password", "sshtunnel_server", sshtunnel_port, "sshtunnel_credentials" ]]
# @endverbatim
#  @author Fleg
#  @date Sept 2022
class DataManager:
    _file: str
    _directories: list = []
    _repositories: list = []
    _incoming: str
    _credentials: dict = {}

    def __init__(self, file):
        configur = ConfigParser()
        # by default configparser convert all keys to lowercase... but we don't want !
        configur.optionxform = lambda option: option
        self._file = file
        configur.read(file)

        # Get credentials
        for name in configur['credentials']:
            cred = json.loads(configur.get('credentials', name))
            self._credentials[name] = Credentials(name, cred[0], cred[1], cred[2], cred[3])

        # Get localdirs (the first in the list is the incoming)
        dirlist = json.loads(configur.get('directories', 'localdir'))
        self._incoming = dirlist[0]
        self._directories.append(Datasource("localdir", "local", "localhost", "", dirlist, self.incoming()))
        # We also append localdirs to repositories... so search method will first look at local dirs before searching on remote locations
        self._repositories.append(Datasource("localdir", "local", "localhost", "", dirlist, self.incoming()))

        # Get DB infos
        for database in configur['database']:
            db = json.loads(configur.get('database', database))
            # dbase = Database(db[0], db[1], db[2], db[3], db[4], db[5], db[6], db[7])
            cred = None
            if db[7] in self._credentials.keys():
                cred = self._credentials[db[7]]
            dbase = Database(db[0], db[1], db[2], db[3], db[4], db[5], db[6], cred)
            dbrepos = dbase.get_repos()
            if not (dbrepos is None):
                for repo in dbrepos:
                    print(repo["name"] + repo["protocol"] + repo["server"] + str(repo["port"]) + str(repo["path"]))
                    # self._repositories.append(
                    #    Datasource(repo["name"], repo["protocol"], repo["server"], repo["port"],
                    #               ast.literal_eval(repo["path"]), self.incoming()))

        # Add remote repositories
        for name in configur['repositories']:
            repo = json.loads(configur.get('repositories', name))
            ds = Datasource(name, repo[0], repo[1], repo[2], repo[3],
                            self.incoming())
            if ds.name() in self._credentials.keys():
                ds.set_credentials(self._credentials[name])
            self._repositories.append(ds)

        # association of credentials to repositories
        # for repo in self._repositories:
        #    if repo.name() in self._credentials.keys():
        #        repo.set_credentials(self._credentials(repo.name()))

        # if repo.name() == name:
        #    credential = Credentials(name, cred[0], cred[1], cred[2], cred[3])
        #    repo.set_credentials(credential)

    def file(self):
        return self._file

    def incoming(self):
        return self._incoming

    def directories(self):
        return self._directories

    def repositories(self):
        return self._repositories

    # Search and get a file by its name.
    # Look first in localdirs and then in remote repositories. First match is returned.
    #    def search(self, file):
    #        res = None
    #        for rep in self.repositories():
    #            print("SEARCHING in " + rep.name())
    #            res = rep.get(file)
    #            if not (res is None):
    #                break
    #        return res

    ## Get a file from the repositories.
    # If repo or path given, then directly search there.
    # If not, search first in localdirs and then in remote repositories. First match is returned.
    def get(self, file, repository=None, path=None):
        res = None
        # First we check that file is not in localdirs
        for directory in self.directories():
            res = directory.get(file)
            if not (res is None):
                break
        # If file is not in localdir then get it from specified repository
        if res is None:
            # if repository is given we get file directly from this repo
            if not (repository is None):
                rep = self.getrepo(repository)
                if not (rep is None):
                    res = rep.get(file, path)
            # if no repo specified, we search everywhere (skip localdir because already done before)
            else:
                for rep in self.repositories():
                    if not (rep.protocol() == "local"):
                        print("SEARCH in " + rep.name())
                        res = rep.get(file)
                        if not (res is None):
                            break
        return res

    ##Get Datasource object from repository by its name
    def getrepo(self, repo):
        res = None
        for rep in self.repositories():
            if rep.name() == repo:
                res = rep
                break
        return res

    # Get file from repository. First search in localdirs if file is present.


#   def getfile(self, file, reponame):
#       res = None
#       for dire in self.directories():
#           res = dire.search(file)
#           if not (res is None):
#               print(str(res))
#               break
#       if (res is None):
#           repo = self.getrepo(reponame)
#           if not (repo is None):
#               res = repo.search(file)
#           else:
#               print("Repository not found !")
#       return res

# Get file from repository. First search in localdirs if file is present.
#    def getfile2(self, file, reponame):
#        for rep in self.repositories():
#            if ((rep.protocol() == "local") or (rep.name() == reponame)):
#                res = rep.search(file)
#                if not (res is None):
#                    break
#        return res


class Database:
    _host: str
    _port: int
    _database: str
    _user: str
    _passwd: str
    _sshserver: str
    _sshport: int
    _tables = {}

    # _cred : Credentials

    def __init__(self, host, port, database, user, passwd, sshserv="", sshport=22, cred=None):
        self._host = host
        if port == "":
            self._port = 5432
        else:
            self._port = port
        self._database = database
        self._user = user
        self._passwd = passwd
        self._sshserv = sshserv
        if sshport == "":
            self._sshport = 22
        else:
            self._sshport = sshport
        self._cred = cred

        print(self._sshserv + str(self._sshport) + self._cred.name() + self._cred.user())

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
            #self.connect()

            engine = create_engine(
                'postgresql+psycopg2://' + self.user() + ':' + self.passwd() + '@' + self.host() + ':' + self.port() + '/' + self.database())
            Base = automap_base()
            Base.prepare(engine, reflect=True)
            self.session = Session(engine)
            for table in engine.table_names():
                self._tables[table] = getattr(Base.classes, table)
    def __del__(self):
        self.session.close()
        self.server.stop()
        #self.dbconnection.close()

    def connect(self):
        self.dbconnection = psycopg2.connect(
                host=self.host(),
                database=self.database(),
                port=self.port(),
                user=self.user(),
                password=self.passwd())
    def disconnect(self):
        self.dbconnection.close()

    def host(self):
        return self._host

    def port(self):
        return self._port

    def database(self):
        return self._database

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

    def get_repos(self):
        record = None
        try:
            #cursor = self.dbconnection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            #cursor.execute(
            #     "SELECT repository.name as name, repository_access.path as path, repository_access.server_name as server, repository_access.port as port, protocol.name as protocol from repository, protocol, repository_access where repository_access.id_protocol=protocol.id_protocol and repository.id_repository=repository_access.id_repository")
            #record = cursor.fetchall()
            #cursor.close()

            query = self.session.query(self._tables['repository'].name.label("name"),
                                    self._tables['repository_access'].path.label("path"),
                                    self._tables['repository_access'].server_name.label("server"),
                                    self._tables['repository_access'].port.label("port"),
                                    self._tables['protocol'].name.label("protocol"),
                                    self._tables['repository'],
                                    self._tables['protocol'])\
                .join(self._tables['repository'])\
                .join(self._tables['protocol'])
            #record = query.all()
            #for resu in query:
            #    print(resu._asdict())

            print(query.all())
        except psycopg2.DatabaseError as e:
            print(f'Error {e}')

        return record


## @brief Class for storing credentials to access remote system.
#  @author Fleg
#  @date Sept 2022
class Credentials:
    _name: str
    _user: str
    _password: str
    _keyfile: str
    _keypasswd: str

    def __init__(self, name, user, password, keyfile, keypasswd):
        self._name = name
        self._user = user
        self._password = password
        self._keyfile = keyfile
        self._keypasswd = keypasswd

    def name(self):
        return self._name

    def user(self):
        return self._user

    def password(self):
        return self._password

    def keyfile(self):
        return self._keyfile

    def keypasswd(self):
        return self._keypasswd


## @brief Generic class defining a datasource.
# A datasource is a location where files are availables.
# This location can be reached using a protocol (local, ftp, ssh, ...) and eventually accessed with credentials.
# Files are stored in some paths (list). "incoming" is the local path where files will be copied (when found on a remote system).
#  @author Fleg
#  @date Sept 2022
class Datasource:
    # [protocol, server, port,  paths]
    _name: str = ""
    _protocol: str = ""
    _server: str
    _port: int
    _paths: list
    _credentials: Credentials
    _incoming: str

    def __init__(self, name, protocol, server, port, paths, incoming):
        self._name = name
        self._protocol = protocol
        self._server = server
        self._port = port
        self._paths = paths
        # By default no credentials
        self._credentials = Credentials(name, "", "", "", "")
        self._incoming = incoming
        if protocol == 'ssh':
            self.__class__ = DatasourceSsh
        elif protocol == 'local':
            self.__class__ = DatasourceLocal
        elif protocol == 'sftp':
            self.__class__ = DatasourceSsh
        elif protocol == 'http':
            self.__class__ = DatasourceHttp
        elif protocol == 'https':
            self.__class__ = DatasourceHttps

    def set_credentials(self, credentials):
        self._credentials = credentials

    def name(self):
        return self._name

    def protocol(self):
        return self._protocol

    def server(self):
        return self._server

    def port(self):
        return self._port

    def paths(self):
        return self._paths

    def credentials(self):
        return self._credentials

    def incoming(self):
        return self._incoming

    #    def search(self, file):
    #        print("search method for protocol " + self.protocol() + " not implemented for repository " + self.name())
    #        return None

    ## Generic get method.
    # Actual method is implemented in subclasses.
    # The path to the local copy of the file is returned for further access.
    # If no file is found then None is returned.
    def get(self, file, path=None):
        print("get method for protocol " + self.protocol() + " not implemented for repository " + self.name())
        return None


## @brief Implementation of the get method for local files.
# Files are searched in local directories. No credentials are used.
# @author Fleg
# @date Sept 2022
class DatasourceLocal(Datasource):
    # Search for file in local directories and return the path to the first corresponding file found.
    #    def search(self, file):
    #        found_file = None
    #        for path in self.paths():
    #            print("search in localdir " + path + file)
    #            my_file = Path(path + file)
    #            if my_file.is_file():
    #                print("found in localdir " + path + file)
    #                found_file = path + file
    #                break
    #            else:
    #                print("file " + file + " not found in localdir " + path)
    #        return found_file

    ## Search for file in local directories and return the path to the first corresponding file found.
    def get(self, file, path=None):
        # TODO : Check that path is in self.paths(), if not then copy in incoming ?
        found_file = None
        if not (path is None):
            my_file = Path(path + file)
            if my_file.is_file():
                found_file = path + file
            else:
                print("file " + file + " not found in localdir " + path)
        else:
            for path in self.paths():
                print("search in localdir " + path + file)
                my_file = Path(path + file)
                if my_file.is_file():
                    found_file = path + file
                    break
                else:
                    print("file " + file + " not found in localdir " + path)

        if not found_file is None:
            print("found in localdir " + path + file)
        return found_file


## @brief Implementation of the get method for ssh access.
# Files are searched on a remote system accessed by ssh.
# @author Fleg
# @date Sept 2022
class DatasourceSsh(Datasource):
    # Search for files in remote location accessed through ssh.
    # If file is found, it will be copied in the incoming local directory and the path to the local file is returned.
    # If file is not found, then None is returned.
    #    def search(self, file):
    #        localfile = None
    #        client = paramiko.SSHClient()
    #        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #        client.connect(hostname=self.server(),
    #                       port=self.port() if self.port() != "" else None,
    #                       username=self.credentials().user() if self.credentials().user() != "" else None,
    #                       key_filename=self.credentials().keyfile() if self.credentials().keyfile() != "" else None,
    #                       passphrase=self.credentials().keypasswd() if self.credentials().keypasswd() != "" else None)
    #
    #        for path in self.paths():
    #            stdin, stdout, stderr = client.exec_command('ls ' + path + file)
    #            lines = list(map(lambda s: s.strip(), stdout.readlines()))
    #
    #            if len(lines) == 1:
    #                print("file found in repository " + self.name() + " at " + lines[0].strip('\n'))
    #                print("copy to " + self.incoming() + file)
    #                scpp = scp.SCPClient(client.get_transport())
    #                scpp.get(lines[0].strip('\n'), self.incoming() + file)
    #                localfile = self.incoming() + file
    #                break
    #            else:
    #                print("file not found in repository " + self.name() + path)
    #
    #        return localfile

    ## Search for files in remote location accessed through ssh.
    # If file is found, it will be copied in the incoming local directory and the path to the local file is returned.
    # If file is not found, then None is returned.
    def get(self, file, path=None):
        localfile = None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=self.server(),
                           port=self.port() if self.port() != "" else None,
                           username=self.credentials().user() if self.credentials().user() != "" else None,
                           key_filename=self.credentials().keyfile() if self.credentials().keyfile() != "" else None,
                           passphrase=self.credentials().keypasswd() if self.credentials().keypasswd() != "" else None)
            if not (path is None):
                print("search " + path + file)
                localfile = self.get_file(client, path, file)
            else:
                for path in self.paths():
                    print("search " + path + file)
                    localfile = self.get_file(client, path, file)
                    if not (localfile is None):
                        break
        except paramiko.SSHException as e:
            print(f'Error {e}')

        return localfile

    def get_file(self, client, path, file):
        localfile = None
        stdin, stdout, stderr = client.exec_command('ls ' + path + file)
        lines = list(map(lambda s: s.strip(), stdout.readlines()))
        if len(lines) == 1:
            print("file found in repository " + self.name() + " at " + lines[0].strip('\n'))
            print("copy to " + self.incoming() + file)
            scpp = scp.SCPClient(client.get_transport())
            scpp.get(lines[0].strip('\n'), self.incoming() + file)
            localfile = self.incoming() + file
        return localfile


## @brief Implementation of the get method for http access.
# Files are searched on a remote system accessed by http.
# @author Fleg
# @date Sept 2022
class DatasourceHttp(Datasource):
    # Search for files in remote location accessed through http.
    # If file is found, it will be copied in the incoming local directory and the path to the local file is returned.
    # If file is not found, then None is returned.
    # TODO: implement authentification
    #    prot: str = "http"
    #    def search(self, file):
    #        localfile = None
    #        for path in self.paths():
    #            url = self.prot + '://' + self.server() + '/' + path + '/' + file
    #            try:
    #                urllib.request.urlretrieve(url, self.incoming() + file)
    #                print("file found in repository " + url)
    #                localfile = self.incoming() + file
    #                break
    #
    #            except urllib.error.HTTPError as e:
    #                print("error searching repository "  + url + " : "+ str(e.code) + " " + e.msg)
    #                #print(e.__dict__)
    #            except urllib.error.URLError as e:
    #                print("error searching repository "  + url + " : "+ str(e.reason))
    #            except:
    #                print("file not found in repository " + url)
    #        return localfile

    ## Search for files in remote location accessed through http.
    # If file is found, it will be copied in the incoming local directory and the path to the local file is returned.
    # If file is not found, then None is returned.
    # TODO: implement authentification
    def get(self, file, path=None):
        localfile = None
        if not (path is None):
            url = self.prot + '://' + self.server() + '/' + path + '/' + file

        for path in self.paths():
            url = self.prot + '://' + self.server() + '/' + path + '/' + file
            localfile = self.get_file(url, file)
            if not (localfile is None):
                break

        return localfile

    def get_file(self, url, file):
        localfile = None
        try:
            urllib.request.urlretrieve(url, self.incoming() + file)
            print("file found in repository " + url)
            localfile = self.incoming() + file

        except urllib.error.HTTPError as e:
            print("error searching repository " + url + " : " + str(e.code) + " " + e.msg)
            # print(e.__dict__)
        except urllib.error.URLError as e:
            print("error searching repository " + url + " : " + str(e.reason))
        except:
            print("file not found in repository " + url)
        return localfile


## @brief Implementation of the get method for https access.
# Files are searched on a remote system accessed by https.
# @author Fleg
# @date Sept 2022
class DatasourceHttps(DatasourceHttp):
    ## Search for files in remote location accessed through https.
    # If file is found, it will be copied in the incoming local directory and the path to the local file is returned.
    # If file is not found, then None is returned.
    prot: str = "https"
