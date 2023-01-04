import logging
from pathlib import Path
import scp
import paramiko
import json
from configparser import ConfigParser
import urllib.request
import os
import shutil
from granddblib import Database
from logging import getLogger

logger = getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)
#logger.addHandler(ch)

## @brief Class for managing datas.
# It will read an inifile where localdirs and different datasources are defined.
# The first localdir will be used as an incoming directory (i.e. all files searched in distant location and found will
# be copied in this incoming directory.
# The inifile will have the following structure :
# @verbatim
# [general]
# provider = "Your name"
# [directories]
# localdir = ["./incoming/", "/some/directory/" , "/some/other/directory ]
# [repositories]
# Repo1 = ["protocol","server",port,["/directory1/", "/other/dir/"]]
# [credentials]
# Repo1 = ["user","password","keyfile","keypasswd"]
# [database]
# localdb = ["host", port, "dbname", "user", "password", "sshtunnel_server", sshtunnel_port, "sshtunnel_credentials" ]]
# [registerer]
# repo2 = "/path/to/repo"
# @endverbatim
#  @author Fleg
#  @date Sept 2022
class DataManager:
    _file: str
    _directories: list = []
    _repositories: dict = {}
    _incoming: str
    _credentials: dict = {}
    _referer = None
    _database = None
    _provider = None

    def __init__(self, file="config.ini"):
        configur = ConfigParser()
        # by default configparser convert all keys to lowercase... but we don't want !
        configur.optionxform = lambda option: option
        self._file = file
        configur.read(file)

        if configur.has_section('general'):
            for name in configur['general']:
                ref = json.loads(configur.get('general', name))
                if name == "provider":
                    self._provider = ref

        if self._provider == None:
            logger.error(f"Config file ({self._file}) must have a [general] section with a provider value")
            exit(1)

        # Get credentials
        if configur.has_section('credentials'):
            for name in configur['credentials']:
                cred = json.loads(configur.get('credentials', name))
                self._credentials[name] = Credentials(name, cred[0], cred[1], cred[2], cred[3])

        if configur.has_section('directories'):
            # Get localdirs (the first in the list is the incoming)
            dirlist = json.loads(configur.get('directories', 'localdir'))
            if not dirlist[0].startswith('/'):
                logger.error(f"Incoming directory (in {self._file}) must be an absolute path.")
                exit(1)
            # Add trailing slash if needed
            dirlist = [os.path.join(path, "") for path in dirlist]
            self._incoming = dirlist[0]
            self._directories.append(Datasource("localdir", "local", "localhost", "", dirlist, self.incoming()))
            # We also append localdirs to repositories... so search method will first look at local dirs before searching on remote locations
            # self._repositories.append(Datasource("localdir", "local", "localhost", "", dirlist, self.incoming()))
            self._repositories["localdir"] = Datasource("localdir", "local", "localhost", "", dirlist, self.incoming())
        else:
            logger.error(f"Section directories is mandatory in config file {file}")
            exit(1)

        # Get DB infos
        if configur.has_section('database'):
            for database in configur['database']:
                db = json.loads(configur.get('database', database))
                cred = None
                if db[7] in self._credentials.keys():
                    cred = self._credentials[db[7]]
                self._database = Database(db[0], db[1], db[2], db[3], db[4], db[5], db[6], cred)
                dbrepos = self._database.get_repos()
                if not (dbrepos is None):
                    for repo in dbrepos:
                        dbpaths = repo["path"].strip("{}").split(",")
                        paths = dbpaths
                        # Add already existing dirs defined in conf
                        if repo["repository"] in self._repositories.keys():
                            paths = list(set(self._repositories[repo["repository"]].paths() + dbpaths))
                            # paths = self._repositories[repo["repository"]].paths() + [element for element in dbpaths if element not in self._repositories[repo["repository"]].paths()]
                        ds = Datasource(repo["repository"], repo["protocol"], repo["server"], repo["port"],
                                        paths, self.incoming(), repo["id_repository"])
                        if ds.name() in self._credentials.keys():
                            ds.set_credentials(self._credentials[ds.name()])
                        self._repositories[repo["repository"]] = ds

        # Add remote repositories
        if configur.has_section('repositories'):
            for name in configur['repositories']:
                repo = json.loads(configur.get('repositories', name))
                ds = Datasource(name, repo[0], repo[1], repo[2], repo[3],
                                self.incoming())
                if ds.name() in self._credentials.keys():
                    ds.set_credentials(self._credentials[name])
                self._repositories[name] = ds

        # Define referer
        if configur.has_section('registerer'):
            for name in configur['registerer']:
                ref = json.loads(configur.get('registerer', name))
                self._referer = self._repositories[name]
                self._referer._incoming = os.path.join(ref, "")
                self._referer.paths().append(self._referer._incoming)
        else:
            self._referer = self._repositories['localdir']

        # Need to ensure that referer repository is registred in the DB
        if self._database is not None:
            self._referer.id_repository = self.database().register_repository(self._referer.name(),
                                                                              self._referer.protocol(),
                                                                              self._referer.port(),
                                                                              self._referer.server(),
                                                                              self._referer.paths(), "")

    def provider(self):
        return self._provider
    def file(self):
        return self._file

    def incoming(self):
        return self._incoming

    def directories(self):
        return self._directories

    def repositories(self):
        return self._repositories

    def database(self):
        return self._database

    def referer(self):
        return self._referer

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
                # for rep in self.repositories():
                for name, rep in self.repositories().items():
                    if not (rep.protocol() == "local"):
                        logger.debug(f"search in repository {rep.name()}")
                        res = rep.get(file)
                        if not (res is None):
                            break
        return res

    def copy_to_incoming(self, pathfile):
        newname = self.incoming() + uniquename(pathfile)
        if os.path.join(os.path.dirname(pathfile), "") == self.incoming():
            os.rename(pathfile, newname)
        else:
            shutil.copy2(pathfile, newname)
        return newname

    ##Get Datasource object from repository by its name
    def getrepo(self, repo):
        res = None
        # for rep in self.repositories():
        for name, rep in self.repositories().items():
            if rep.name() == repo:
                res = rep
                break
        return res


    ##Function to register a file into the database. Returns the path to the file in the repository where the file was registered.
    def register_file(self,filename):
        file = self.get(filename)
        newfilename = self.referer().copy(file)
        self.database().register_file(file, newfilename, self.referer().id_repository, self.provider())
        return newfilename



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
    id_repository: int

    def __init__(self, name, protocol, server, port, paths, incoming, id_repo=None):
        self._name = name
        self._protocol = protocol
        self._server = server
        self._port = port
        self._paths = paths
        # By default no credentials
        self._credentials = Credentials(name, "", "", "", "")
        self._incoming = incoming
        self.id_repository = id_repo
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
        if self._port == '':
            self._port = None
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
        logger.warning(f"get method for protocol {self.protocol()} not implemented for repository {self.name()}")
        return None

    def copy(self, pathfile):
        logger.warning(f"copy method for protocol {self.protocol()}  not implemented for repository {self.name()}")
        return None


## @brief Implementation of the get method for local files.
# Files are searched in local directories. No credentials are used.
# @author Fleg
# @date Sept 2022
class DatasourceLocal(Datasource):
    ## Search for file in local directories and return the path to the first corresponding file found.
    def get(self, file, path=None):
        # TODO : Check that path is in self.paths(), if not then copy in incoming ?
        found_file = None
        if not (path is None):
            my_file = Path(path + file)
            if my_file.is_file():
                found_file = path + file
            else:
                logger.debug(f"file {file}  not found in localdir {path}")
        else:
            for path in self.paths():
                logger.debug(f"search in localdir {path}{file}")
                my_file = Path(path + file)
                if my_file.is_file():
                    found_file = path + file
                    break
                else:
                    logger.debug(f"file {file} not found in localdir {path}")

        if not found_file is None:
            logger.debug(f"file found in localdir {path}{file}")

        return found_file

    def copy(self, pathfile):
        newname = self.incoming() + uniquename(pathfile)
        if os.path.join(os.path.dirname(pathfile), "") == self.incoming():
            os.rename(pathfile, newname)
        else:
            shutil.copy2(pathfile, newname)
        return newname


## @brief Implementation of the get method for ssh access.
# Files are searched on a remote system accessed by ssh.
# @author Fleg
# @date Sept 2022
class DatasourceSsh(Datasource):

    def set_client(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=self.server(),
                       port=self.port() if self.port() != "" else None,
                       username=self.credentials().user() if self.credentials().user() != "" else None,
                       key_filename=self.credentials().keyfile() if self.credentials().keyfile() != "" else None,
                       passphrase=self.credentials().keypasswd() if self.credentials().keypasswd() != "" else None)
        return client

    def get(self, file, path=None):
        import getpass
        localfile = None
        client = self.set_client()
        if not (path is None):
            logger.debug(f"search {path}{file} @ {self.name()}")
            localfile = self.get_file(client, path, file)
        else:
            for path in self.paths():
                logger.debug(f"search {path}{file} @ {self.name()}")
                localfile = self.get_file(client, path, file)
                if not (localfile is None):
                    break

    ## Search for files in remote location accessed through ssh.
    # If file is found, it will be copied in the incoming local directory and the path to the local file is returned.
    # If file is not found, then None is returned.

    def get_file(self, client, path, file):
        localfile = None
        stdin, stdout, stderr = client.exec_command('ls ' + path + file)
        lines = list(map(lambda s: s.strip(), stdout.readlines()))
        if len(lines) == 1:
            logger.debug(f"file found in repository {self.name()}  @ " + lines[0].strip('\n'))
            logger.debug(f"copy to {self.incoming()}{file}")
            scpp = scp.SCPClient(client.get_transport())
            scpp.get(lines[0].strip('\n'), self.incoming() + file)
            localfile = self.incoming() + file
        return localfile

    def copy(self, pathfile):
        newname = self.incoming() + uniquename(pathfile)
        client = self.set_client()
        # search if original file exists remotely
        stdin, stdout, stderr = client.exec_command('ls ' + self.incoming() + os.path.basename(pathfile))
        lines = list(map(lambda s: s.strip(), stdout.readlines()))
        if len(lines) == 1:
            # original file exists... we rename it.
            client.exec_command('mv ' + self.incoming() + os.path.basename(pathfile) + ' ' + newname)
        else:
            # search if dest files already there
            stdin, stdout, stderr = client.exec_command('ls ' + newname)
            lines = list(map(lambda s: s.strip(), stdout.readlines()))
            if len(lines) != 1:
                # if newfile is not there we copy it
                scpp = scp.SCPClient(client.get_transport())
                scpp.put(pathfile, newname)
        return newname


## @brief Implementation of the get method for http access.
# Files are searched on a remote system accessed by http.
# @author Fleg
# @date Sept 2022
class DatasourceHttp(Datasource):
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
            logger.debug(f"file found in repository {url}")
            localfile = self.incoming() + file

        except urllib.error.HTTPError as e:
            logger.error(f"error searching repository {url} : {str(e.code)}  {e.msg}")
        except urllib.error.URLError as e:
            logger.error(f"error searching repository {url} :  {str(e.reason)}")
        #       except:
        #           print("file not found in repository " + url)
        return localfile


## @brief Implementation of the get method for https access.
# Files are searched on a remote system accessed by https.
# @author Fleg
# @date Sept 2022
class DatasourceHttps(DatasourceHttp):
    ## Search for files in remote location accessed through https.
    # Same process as http but https instead !
    prot: str = "https"


def uniquename(filename):
    return hashfile(filename) + ".root"

## @brief Function which return the hash of a file.
# This hash will be used as unique name for the file.
def hashfile(filename):
    import hashlib
    BLOCK_SIZE = 262144
    file_hash = hashlib.sha256()
    with open(filename, 'rb') as f:
        fb = f.read(BLOCK_SIZE)
        while len(fb) > 0:
            file_hash.update(fb)
            fb = f.read(BLOCK_SIZE)
    return file_hash.hexdigest()
