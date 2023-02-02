Grand Database
==============

Principle
---------
A master database is set up for the Grand project (for now @CCIN2P3).
This main database will reference all the "official" files (root files) after validation.
This database is read only for users.

Collaborators can create a local database (on their own computer, a server at their lab, ...) which will synchronize with datas from the master database in Lyon. This local DB will be read+write for users. Sync is asymetric (means that records from the master DB will be added in the local DB, but records added in the local DB will not be uploaded to the master DB). 

A docker container is provided to create the local DB with tools to synchronize.


Install
------
To set up a local DB follow these steps :

* Configure the db.conf file. To get the master DB password please send a signed email to fleg@lpnhe.in2p3.fr. Also provide the IP adress from which you will run the docker (because there is an access list for the master DB and we need to authorize your machine to access the master DB).
* Build the docker with the command
.. code:: bash

   docker build -f granddb.dockerfile . -t granddb:latest


* Start the local docker with :
.. code:: bash

   ./startdbgrand.bash


* The docker also brings a small web app to browse the DB. It will be automatically started with the docker if you define its port (using the WEB_PORT in db.conf).
Synchronization is automatically done when starting the docker with startdbgrand.bash.

* I you want to synchronize without restarting the docker, simply run
.. code:: bash

   docker container exec  -it -u root granddb /app/update-db.bash
   

