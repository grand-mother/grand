#!/bin/bash 
export PGPASSWORD='password'
#export PGPASSWORD=$POSTGRES_PASSWORD

psql  -h localhost postgres postgres -c "CREATE DATABASE granddb WITH OWNER = postgres ENCODING = 'UTF8' LC_COLLATE = 'en_US.utf8' LC_CTYPE = 'en_US.utf8' TABLESPACE = pg_default CONNECTION LIMIT = -1 IS_TEMPLATE = False;"

pg_dump -h lpndocker01.in2p3.fr -s -Fc  -U postgres  granddb >granddbdump.sql

pg_restore -h localhost -U postgres  -d granddb granddbdump.sql

rm granddbdump.sql

pgsync --defer-constraints
