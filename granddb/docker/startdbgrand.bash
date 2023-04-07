#!/bin/bash
if [ $(stat -c %a db.conf) != 600 ]; then
  echo "file db.conf must be with permission 600. Exiting"
  exit 1
fi
source db.conf

docker run --name granddb \
	-e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
	-e GRANDADMIN_PASSWORD=$GRANDADMIN_PASSWORD \
	-e GRANDUSER_PASSWORD=$GRANDUSER_PASSWORD \
	-e MASTER_SERVER=$MASTER_SERVER \
  -e MASTER_DB=$MASTER_DB \
  -e MASTER_PORT=$MASTER_PORT \
  -e MASTER_USER=$MASTER_USER \
  -e MASTER_PASSWORD=$MASTER_PASSWORD \
	-p $WEB_PORT:8081 \
	-p $DB_PORT:5432 \
	-v $DB_PATH:/var/lib/postgresql/data \
	-v /etc/passwd:/etc/passwd:ro \
	-u `id -u`:`id -g` \
	--rm \
	-d granddb:latest

docker container exec -u root -it granddb /app/create-db.bash
if [ "$WEB_PORT" != "" ]; then
  docker container exec -u root -d granddb /app/start-web.bash
fi
