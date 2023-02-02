#!/bin/bash
create_user(){
  local user="$1"
  local passwd="$2"

  psql -h localhost postgres postgres -tc "SELECT 1 FROM pg_user WHERE usename = '$user'" | grep -q 1
  userexists=$?
  if [ $userexists -ne 0 ]; then
    if [ "$passwd" = "" ]; then
        psql  -h localhost postgres postgres -c \
        "CREATE ROLE $user WITH LOGIN NOSUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION;"
    else
        psql  -h localhost postgres postgres -c \
        "CREATE ROLE $user WITH LOGIN NOSUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD '$passwd';"
    fi
  fi
}

sed -i -e "s/GRANDADMIN_PASSWORD/$GRANDADMIN_PASSWORD/g" .pgsync.yml
sed -i -e "s/MASTER_USER/$MASTER_USER/g" .pgsync.yml
sed -i -e "s/MASTER_PASSWORD/$MASTER_PASSWORD/g" .pgsync.yml
sed -i -e "s/MASTER_SERVER/$MASTER_SERVER/g" .pgsync.yml
sed -i -e "s/MASTER_PORT/$MASTER_PORT/g" .pgsync.yml
sed -i -e "s/MASTER_DB/$MASTER_DB/g" .pgsync.yml

echo "$MASTER_SERVER:$MASTER_PORT:$MASTER_DB:$MASTER_USER:$MASTER_PASSWORD" > /root/.pgpass
echo "localhost:5432:granddb:grandadmin:$GRANDADMIN_PASSWORD" >> /root/.pgpass
echo "localhost:5432:granddb:granduser:$GRANDUSER_PASSWORD" >> /root/.pgpass
echo "localhost:5432:*:postgres:$POSTGRES_PASSWORD" >> /root/.pgpass
chmod 600 /root/.pgpass

# Wait for DB to be up
looper=1
until psql -h localhost postgres postgres -tc "select 1 FROM pg_user" -d postgres &> /dev/null
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

psql -h localhost -d granddb -U postgres -tc "select 1"  &> /dev/null
dbexists=$?
if [ $dbexists -ne 0 ]; then
  psql  -h localhost postgres postgres -c "CREATE DATABASE granddb WITH OWNER = grandadmin ENCODING = 'UTF8' LC_COLLATE = 'en_US.utf8' LC_CTYPE = 'en_US.utf8' TABLESPACE = pg_default CONNECTION LIMIT = -1 IS_TEMPLATE = False;"

  echo "Get main DB infos"
  pg_dump -h $MASTER_SERVER -p $MASTER_PORT -s -Fc -U $MASTER_USER  -d $MASTER_DB >granddbdump.sql
  echo "Set up database"

  pg_restore -h localhost -U grandadmin  -d granddb granddbdump.sql
  rm granddbdump.sql
fi

echo "Run pgsync"
#pgsync  --defer-constraints --preserve
pgsync  --defer-constraints grand

seq=$(psql -h localhost -d granddb grandadmin -tc "select sequence_name from information_schema.sequences where sequence_catalog='granddb' ORDER BY sequence_name ;")

for i in $seq
do
        cur=$(psql -h localhost -d granddb grandadmin -tc "select last_value from $i ;")
        half=$(psql -h localhost -d granddb grandadmin -tc "select maximum_value::BIGINT / 2 from information_schema.sequences where sequence_name='$i' ;")
        if [ $cur -lt $half ]; then
                psql -h localhost -d granddb grandadmin -tc "SELECT setval('$i', $half, true);" &> /dev/null
        fi
done
