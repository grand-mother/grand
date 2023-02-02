To set up a local DB (which will be synchronized with the main master DB) follow these steps :

Configure the db.conf file. To get the master DB password please send a signed email to fleg@lpnhe.in2p3.fr. Also provide the IP adress from which you will run the docker (because there is an access list for the master DB and we need to authorize your machine to access the master DB).
Build the docker with the command: docker build -f granddb.dockerfile . -t granddb:latest
Start the local docker with :  ./startdbgrand.bash

The docker also bring a small web app to browse the DB. It will be automatically started with the docker if you define its port (using the WEB_PORT in db.conf).
