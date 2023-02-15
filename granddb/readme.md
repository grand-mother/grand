This section regroups the different libraries to interact with datas in Grand.

# DataManager

Datamanager object is the main object to access datas. It first get it's configuration from an inifile (config.ini by default).
To run into docker, you need to use the grandlib/dev:1.2 version to have all the requested libraries.

## Inifile

Inifile is organized in sections. The 6 sections are [general][directories][repositories][credentials][database][registerer]

    [general]
    provider = "Your name"
    socket_timeout = 5
  
    [directories]
    localdir = ["/path/to/incoming/dir", "/another/local/directory"]
    
    [repositories]
    ; Name = [protocol,server, port, [paths]]
    CC = ["ssh","cca.in2p3.fr",22,["/path/to/datas/","/another/path/"]]
    WEB = [ "https", "github.com" , 443, ["/grand-mother/data_challenge1/raw/main/coarse_subei_traces_root/"]]
  
    [credentials]
    ; Name =  [user, keyfile]
    CC = ["login",""]
    CCIN2P3 = ["login",""]
    SSHTUNNEL = ["ssh_login",""]
  
    [database]
    ; Name = [server, port, database, login, passwd, sshtunnel_server, sshtunnel_port, sshtunnel_credentials ]
    database = ["dbserver.in2p3.fr", "" ,"dbname", "dbuser", "dbpass","ssh_tunnel.in2p3.fr", 22, "SSHTUNNEL"]

    [registerer]
    CCIN2P3 = "/sps/trend/fleg/INCOMING"
  

Directories are **local** directories where data should be. The first path in localdir will be used as an incoming folder (also see below). The incoming folder is the local folder where the files found remotely will be copied. This directory must exists and be writable. 
Repositories are **distant** places where data should be. Repositories are accessed using a protocol. 

The following protocols are supported : ssh, http, https, local.

Sections [database] and [registerer] are optional (these sections can be commented or removed if you don't want to use the database).

[credentials] section allows you to specify your login and optionally a key file to access repositories or connect database though an ssh tunnel etc...
For security reasons you will not be allowed to provide sensitive information as password in this file. If password is required (e.g. to decrypt the key file) it will be asked interactively.
For ssh protocol, it's highly encouraged to use an ssh-agent (to avoid to have to provide passwd interactively at each run)
To run an ssh-agent just do : `eval $(ssh-agent)` and `ssh-add .ssh/id_rsa`

To export your ssh agent from host to docker simply add an environment variable SSH_AUTH_SOCK=/ssh-agent to your docker
and mount the volume with `-v ${SSH_AUTH_SOCK}:/ssh-agent`

## Datamanager
When instantiated, a datamanager object will read it's configuration from the ini file. If a database is declared, it will connect to the DB to get a list of eventual other repositories.

### The get function
The get(filename) function fill perform the following actions :
- Search if a file called < filename > exists in localdirs (and subdirs). 
  - If yes, returns the path to the first file found.
  - If no, recursively search for the file in the various repositories.
    - If found in a repository, then get the first file found (using protocol for the repository) and copy it in the incoming local directory and return the path to the newly copied file.
    - If not, return None.
 
Usage example:

    import granddb.granddatalib as granddatalib
    dm = granddatalib.DataManager('config.ini')
    file="Coarse3.root"
    print(dm.get(file))

Search can be restricted/forced on only one repository by specifiying the repository name as second argument to the get function:

    print(dm.get(file, "CCIN2P3"))

In this case, only the specified repository is searched, and if the file is found it's copied into the incoming directory. If the file was already present in the incoming directory, it will be overwritten.

Search can also be restricted to a directory in a specified repository (use the desired directory path as third argument). In that case, the path has not to be declared in the config.ini file. This allows you to retreive a "one time shot" file from a specific location.

    dm.get(file, "CC", "/sps/trend/fleg/")

This is also works with localdir (but in that case the file is not copied into the incoming directory, the path to the file is simply returned)

    file = 'main.py'
    dm.get(file, "localdir")
    dm.get(file, "localdir","../venv/lib/python3.8/site-packages/pip/_internal/cli/")


### The search function

The search function (not yet properly implemented) will return the list of repositories/directories where a file can be found.
It will perform a search in the database.

### Test example
#### For linux users

To test, you can do the following : 

* Edit and configure the examples/datalib/config.ini
* Run the docker


        docker run -it -v /path/to/grand/lib:/home -v ${SSH_AUTH_SOCK}:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent --rm grandlib/dev:1.2

* Inside the docker do : 

        source env/setup.sh
        cd examples/datalib/
        python datamanager_example.py
    

* Check that the Coarse3.root has been retreived in /home/examples/datalib/incoming

        ls /home/examples/datalib/incoming/Coarse3.root

#### For mac users
Mac does'nt allow to forward agent into docker... thus you will have to start an agent directly inside your docker :
* Edit and configure the examples/datalib/config.ini
* Run the docker

        docker run -it -v /path/to/grand/lib:/home -v /path/to/.ssh:/home/.ssh --rm grandlib/dev:1.2


* Inside the docker do : 


        eval $(ssh-agent)
        ssh-add .ssh/id_rsa
        source env/setup.sh
        cd examples/datalib/
        python datamanager_example.py

* Check that the Coarse3.root has been retreived in /home/examples/datalib/incoming

        ls /home/examples/datalib/incoming/Coarse3.root
