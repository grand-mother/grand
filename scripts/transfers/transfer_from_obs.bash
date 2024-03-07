#!/bin/bash
# Script to transfert data from a GRAND observatory to CCIN2P3 (or to any site)
# Fleg: 03/2024
# Copyright : Grand Observatory 2024

##### Configuration part #####
# Please adjust the following variable to your site

# Local database name (sqlite filename)
dbfile='grand_transfer.db'

# Local directory where are stored the data to be transfered (will be explored recursively)
localdatadir='/sps/grand/data/gp13/raw/2024/'

# Site name prefix in filenames
site='GP13'

# Remote server to transfer
remote_server='lpnws5131.in2p3.fr' # 'cca.in2p3.fr'

# Account on remote server
remote_account='fleg' # 'prod_grand'

# Target directory on remote server
remotedatadir='/data/tmp/'  #'/sps/grand/data/gp13/raw'

# Start date for transfer (all files older than this date will be skipped
first_transfer='20240302'

##### End of Configuration section (do not modify below) #####

# Create database if not exists
sqlite3 $dbfile "create table if not exists  gfiles (id INTEGER PRIMARY KEY AUTOINCREMENT, directory TEXT, file TEXT, date INT, success BOOLEAN, UNIQUE (directory,file));"
sqlite3 $dbfile "create table if not exists  transfer (id, tag INTEGER, date_transfer DATETIME, success BOOLEAN, comment TEXTE);"


# Define some useful stuff

# Last date of files already registered
last_transfer=$(sqlite3 $dbfile "select max(date) from gfiles;")
last_transfer=$(( last_transfer > first_transfer ? last_transfer : first_transfer ))

#tag to identify files treated in the current run
tag=$(date +'%Y%m%d%H%M%S')

# Colors
Default='\033[0m'	# Text Reset
Red='\033[0;31m'	# Red
Green='\033[0;32m'	# Green

#List of files to be inserted into the db by bunchs of 500 (larger should produce errors)
declare -A toins=([1]="")
i=1
j=0
#find all files in localdatadir corresponding to datas (i.e. name starting by site) and add them to the database if not here already
for file in $(find $localdatadir -type f -newermt $last_transfer| grep /${site}_ |sort)
do
	filename=$(basename $file)
	tmp=${filename#${site}_}
	dateobs=${tmp:0:8}
	#Add file to be registered to the list (and start new list if more than 500 to avoid request limit in insert below)
	if [ $j -ge 500 ];
	then
		i=$((i + 1))
		toins+=([$i]="")
		j=0
	fi
	toins[$i]+=",('$(dirname $file)', '$(basename $file)', ${dateobs}, false)"
	j=$((j + 1))
done

#Add all files at a time (10x faster that adding them individually)
# but iterate over various lists in case of huge number of files
for key in "${!toins[@]}"; do
	value=${toins[${key}]}
	if [ -n "$value" ];
	then
		res=$(sqlite3 $dbfile "INSERT OR IGNORE INTO gfiles (directory,file,date,success) values ${value:1}")
	fi

done

# Open a ssh connection that will be used for all transfers (avoid to reopen rsync tunnel for each file)
mkdir ~/.ssh/ctl
ssh -nNf -o ControlMaster=yes -o ControlPath="$HOME/.ssh/ctl/%L-%r@%h:%p" ${remote_account}@${remote_server}

declare -A translog=([1]="")
i=1
j=0
#select files not transfered successfully
for file in $(sqlite3 $dbfile "select directory, file, date, success, id from gfiles where success=false ORDER BY file;")
do
	#transform result into array (more easy to manipulate)
	fileinfo=(${file//|/ })
	#Transfer files (one by one to get info on each transfer)
	printf "\nSending ${fileinfo[1]} "
	trans=$(rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" --out-format="%t %b %n" -au --rsync-path="mkdir -p $remotedatadir/${fileinfo[2]:0:4}/${fileinfo[2]:4:2} && rsync" ${fileinfo[0]}/${fileinfo[1]}  ${remote_account}@${remote_server}:$remotedatadir/${fileinfo[2]:0:4}/${fileinfo[2]:4:2}/ 2>&1)

	if [ "$?" -eq "0" ]
	then
		#Transfer successful : store info to update database at the end
		translog[$i]+=";UPDATE gfiles SET success=true WHERE id=${fileinfo[4]};INSERT INTO transfer (id, tag, success,date_transfer,comment) VALUES (${fileinfo[4]},${tag}, true,datetime('now','utc'),'${trans}')"
		printf "${Green}Ok${Default}"

	else
		#Transfer failed : just log errors
		translog[$i]+=";INSERT INTO transfer (id, tag, success,date_transfer,comment) VALUES (${fileinfo[4]}, ${tag}, false,datetime('now','utc'),'${trans}')"
		printf "${Red}ERROR:${Default} \n ${trans} "
	fi

	# split info to store into db in case of large number of files
	if [ $j -ge 100 ];
        then
                i=$((i + 1))
                translog+=([$i]="")
                j=0
        fi

	j=$((j + 1))
done

printf "\n"

#update DB with all results (iterate over logs)
for key in "${!translog[@]}"; do
        value=${translog[${key}]}
        if [ -n "$value" ];
        then
                res=$(sqlite3 $dbfile "${value:1}")
        fi

done

#finally also rsync the database
#rsync -au $dbfile ${remote_account}@${remote_server}:$remotedatadir/${dbfile}_$(date +'%Y%m%d-%H%M%S')
rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" -au $dbfile ${remote_account}@${remote_server}:$remotedatadir/${tag}_${dbfile}

#close ssh connection
ssh -O exit -o ControlPath="$HOME/.ssh/ctl/%L-%r@%h:%p" ${remote_account}@${remote_server}
rm -rf ~/.ssh/ctl


