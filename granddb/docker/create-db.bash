#!/bin/bash
create_user(){
  local user="$1"
  local passwd="$2"

  PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost postgres postgres -tc "SELECT 1 FROM pg_user WHERE usename = '$user'" | grep -q 1
  userexists=$?
  if [ $userexists -ne 0 ]; then
    if [ "$passwd" = "" ]; then
        PGPASSWORD=$POSTGRES_PASSWORD psql  -h localhost postgres postgres -c \
        "CREATE ROLE $user WITH LOGIN NOSUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION;"
    else
        PGPASSWORD=$POSTGRES_PASSWORD psql  -h localhost postgres postgres -c \
        "CREATE ROLE $user WITH LOGIN NOSUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD '$passwd';"
    fi
  fi
}

sed -i -e "s/GRANDADMIN_PASSWORD/$GRANDADMIN_PASSWORD/g" .pgsync.yml

# Wait for DB to be up
looper=2
until PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost postgres postgres -tc "select 1 FROM pg_user" -d postgres &> /dev/null
do
  echo "waiting for database server to be ready...$looper"
  sleep 2
  looper=$((looper + 1))
  if [ $looper -gt 10 ]; then
    echo "timout"
    exit
  fi
done

create_user "grandadmin" "$GRANDADMIN_PASSWORD"
create_user "granduser" "$GRANDUSER_PASSWORD"
create_user "grandreplicator" ""

PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost -d postgres postgres -tc "select 1" -d granddb &> /dev/null
dbexists=$?
if [ $dbexists -ne 0 ]; then
  PGPASSWORD=$POSTGRES_PASSWORD psql  -h localhost postgres postgres -c "CREATE DATABASE granddb WITH OWNER = grandadmin ENCODING = 'UTF8' LC_COLLATE = 'en_US.utf8' LC_CTYPE = 'en_US.utf8' TABLESPACE = pg_default CONNECTION LIMIT = -1 IS_TEMPLATE = False;"

  echo "Get main DB infos"
  PGPASSWORD=xxxxxxxxxxxxx pg_dump -h ccpgsqlexpe.in2p3.fr -p 6550 -s -Fc -U grandreplicator  -d granddb >granddbdump.sql
  echo "Set up database"

  PGPASSWORD=$GRANDADMIN_PASSWORD pg_restore -h localhost -U grandadmin  -d granddb granddbdump.sql
  rm granddbdump.sql
fi

echo "Run pgsync"
pgsync  --defer-constraints --preserve

seq=$(PGPASSWORD=$GRANDADMIN_PASSWORD psql -h localhost -d granddb grandadmin -tc "select sequence_name from information_schema.sequences where sequence_catalog='granddb' ORDER BY sequence_name ;")

for i in $seq
do
        cur=$(PGPASSWORD=$GRANDADMIN_PASSWORD psql -h localhost -d granddb grandadmin -tc "select last_value from $i ;")
        if [ $cur -lt 4500000000000000000 ]; then
                PGPASSWORD=$GRANDADMIN_PASSWORD psql -h localhost -d granddb grandadmin -tc "SELECT setval('$i', 4500000000000000000, true);" &> /dev/null
        fi
done
