#!/bin/bash
docker run --name granddb \
	-e POSTGRES_PASSWORD=password \
	-p 5432:5432 \
	-v /home/fleg/DB-postgres:/var/lib/postgresql/data \
	-v /etc/passwd:/etc/passwd:ro \
	-u `id -u`:`id -g` \
	--rm \
	-d granddb:latest 

#sleep 10
#DB='granddb'
#export PGPASSWORD='password';
#psql  -h localhost postgres postgres -c "select 1" -d $DB &> /dev/null ||  \
#psql  -h localhost postgres postgres -c "CREATE DATABASE $DB
#    WITH 
#    OWNER = postgres
#    ENCODING = 'UTF8'
#    LC_COLLATE = 'en_US.utf8'
#    LC_CTYPE = 'en_US.utf8'
#    TABLESPACE = pg_default
#    CONNECTION LIMIT = -1
#    IS_TEMPLATE = False;"

# TODO : init sequences to 4500000000000000000
# Do a loop over sequences like 
#SELECT sequence_name
#FROM information_schema.sequences
#ORDER BY sequence_name ;
# export PGPASSWORD='password';
# psql -W -h localhost grand postgres -c "SELECT setval('public.protocol_id_protocol_seq', 4500000000000000000, true);"
