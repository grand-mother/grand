Grand Database
==============
**THIS SECTION IS NOT FUCTIONAL ANYMORE. PLEASE DO NOT USE YET (will be corrected soon).**

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

* Rename the db.conf.sample into db.conf and edit the file to set up your configuration. To get the master DB password please send a signed email to fleg@lpnhe.in2p3.fr. Also provide the IP adress from which you will run the docker (because there is an access list for the master DB and we need to authorize your machine to access the master DB).
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
   

Technical details
-----------------

Database is postgresql (version 15.1.).

Separation between "official" datas (in the master DB) and user datas (in the local DB) uses sequences indexes of tables. Basically, tables have primary index to identify each record. Depending on the tables, these indexes are smallint (1-32,767), integer (1-2,147,483,647) or bigint (1-9,223,372,036,854,775,807). The choice of type depends on the number of row we can expect in the table. For example, the number of protocol to access datas will be a few at max... thus we will use smallint; on the other side, the number of events will be huge... thus for events we use bigint.
On the master DB, indexes starts at 1. On the replicated local DB indexes starts at halh of the maximal possible value for the key (i.e. 16383 for smallint, 1073741823 for integer and 4,611686018×10¹⁸ for bigint). Thus datas from the master DB and local user datas are clearly separated (primary key indexes are not in the same range).  

Sync with master DB is performed using pgsync (https://github.com/ankane/pgsync). This solution allows to synchronize only a part of the datas. Basically, it's configured to synchronize datas from the master DB (i.e. with indexes between 1 and halfmax) and leave users data (with indexes between halfmax and max) unchanged.

Web browser is pgweb (https://github.com/sosedoff/pgweb/).


