#!/bin/bash
#----- Configuration -------#
POSTGRES_PASSWORD="pgpass"
GRANDADMIN_PASSWORD="popo"
GRANDUSER_PASSWORD="pipi"
#----- Configuration -------#

docker run --name granddb \
	-e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
	-e GRANDADMIN_PASSWORD=$GRANDADMIN_PASSWORD \
	-e GRANDUSER_PASSWORD=$GRANDUSER_PASSWORD \
	-p 8081:8081 \
	-p 5432:5432 \
	-v /home/fleg/DB-postgres:/var/lib/postgresql/data \
	-v /etc/passwd:/etc/passwd:ro \
	-u `id -u`:`id -g` \
	--rm \
	--network=host \
	-d granddb:latest 

docker container exec -u root -it granddb /app/create-db.bash

