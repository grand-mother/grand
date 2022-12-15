from pathlib import Path
import scp
import paramiko
import json
from configparser import ConfigParser


class DataManager:
    _file: str
    _directories: list = []
    _repositories: list = []
    _incoming: str

    def __init__(self, file):
        configur = ConfigParser()
        self._file = file
        configur.read(file)
        self._directories = json.loads(configur.get('directories', 'localdir'))
        self._incoming = self._directories[0]
        self._repositories.append(Datasource("localdir", "local", "localhost", "", "", "", "", "", self._directories, self.incoming()))


        for name in configur['repositories']:
            repo = json.loads(configur.get('repositories', name))
            self._repositories.append(
                Datasource(name, repo[0], repo[1], repo[2], repo[3], repo[4], repo[5], repo[6], repo[7],
                           self.incoming()))

    def file(self):
        return self._file

    def incoming(self):
        return self._incoming

    def directories(self):
        return self._directories

    def repositories(self):
        return self._repositories

    def search(self, file):
        res = None
        for rep in self.repositories():
            res = rep.search(file)
            if not (res is None):
                print(str(res))
                break
        return res


class Credentials:
    _user: str
    _password: str
    _keyfile: str
    _keypasswd: str

    def __init__(self, user, password, keyfile, keypasswd):
        self._user = user
        self._password = password
        self._keyfile = keyfile
        self._keypasswd = keypasswd

    def user(self):
        return self._user

    def password(self):
        return self._password

    def keyfile(self):
        return self._keyfile

    def keypasswd(self):
        return self._keypasswd


class Datasource:
    # [protocol, server, port, user, password, keyfile, keypasswd, paths]
    _name: str = ""
    _protocol: str = ""
    _server: str
    _port: int
    _paths: list
    _credentials: Credentials
    _incoming: str

    def __init__(self, name, protocol, server, port, user, password, keyfile, keypasswd, paths, incoming):
        self._name = name
        self._protocol = protocol
        self._server = server
        self._port = port
        self._paths = paths
        self._credentials = Credentials(user, password, keyfile, keypasswd)
        self._incoming = incoming
        if protocol == 'ssh':
            self.__class__ = DatasourceSsh
        elif protocol == 'local':
            self.__class__ = DatasourceLocal
        elif protocol == 'sftp':
            self.__class__ = DatasourceSsh

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

    def search(self, file):
        print("protocol " + self.protocol() + " not implemented for repository " + self.name())
        return None


class DatasourceLocal(Datasource):
    def search(self, file):
        found_file = None
        for path in self.paths():
            print("search in localdir " + path + file)
            my_file = Path(path + file)
            if my_file.is_file():
                print("found in localdir " + path + file)
                found_file = path + file
                break
            else:
                print("file " + file + " not found in localdir " + path)

        return found_file


class DatasourceSsh(Datasource):
    def search(self, file):
        localfile = None
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#        client.connect(hostname=self.server(),  username=self.credentials().user())
#        client.connect(hostname=self.server(), port=self.port(), username=self.credentials().user(),
#                       key_filename=self.credentials().keyfile(), passphrase=self.credentials().keypasswd())
        client.connect(hostname=self.server(),
                       port=self.port() if self.port() != "" else None,
                       username=self.credentials().user() if self.credentials().user() != "" else None,
                       key_filename=self.credentials().keyfile() if self.credentials().keyfile() != "" else None,
                       passphrase=self.credentials().keypasswd() if self.credentials().keypasswd() != "" else None)

        for path in self.paths():
            stdin, stdout, stderr = client.exec_command('ls ' + path + file)
            lines = list(map(lambda s: s.strip(), stdout.readlines()))

            if len(lines) == 1:
                print("file found in repository " + self.name() + " at " + lines[0].strip('\n'))
                print("copy to " + self.incoming() + file)
                scpp = scp.SCPClient(client.get_transport())
                scpp.get(lines[0].strip('\n'), self.incoming() + file)
                localfile = self.incoming() + file
                break
            else:
                print("file not found in repository " + self.name() + path)

        return localfile


class Datafile:
    _name: str
    _path: str
    _source: Datasource
