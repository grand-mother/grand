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
remote_server='cca.in2p3.fr'

# Account on remote server
remote_account='prod_grand' # 'prod_grand'

#ssh key for rsync
ssh_key_rsync="/pbs/home/p/prod_grand/.ssh/id_ed25519" # "/root/.ssh/id_ed25519-nopw"

#ssh key for exec remote scripts
ssh_key_exec="/pbs/home/p/prod_grand/.ssh/id_ed25519" # "/root/.ssh/id_ed25519-nopw"

# Target directory on remote server
remotedatadir='/sps/grand/prod_grand/tests'  #'/sps/grand/data/gp13'

# Start date for transfer (all files older than this date will be skipped
first_transfer='20240312'

# Local script to be launched before run
pre_run_script='' #'setup_network_auger.bash -init'

# Local script to be launched after run
post_run_script='' # 'setup_network_auger.bash -close'

# rsync_options : a to keep the creation time of files, z to compress if bandwidth is limited (but it's ~5 times slower). Please keep the "a" option  !
rsync_options="-a"

# treatment scripts location @CCIN2P3
ccscripts='/pbs/home/p/prod_grand/scripts/transfers/ccscript_GP13.bash'

##### End of Configuration section (do not modify below) #####

# Create database if not exists
sqlite3 $dbfile "create table if not exists  gfiles (id INTEGER PRIMARY KEY AUTOINCREMENT, directory TEXT, file TEXT, date INT, success BOOLEAN, md5sum VARCHAR(35), UNIQUE (directory,file));"
sqlite3 $dbfile "create table if not exists  transfer (id, tag INTEGER, date_transfer DATETIME, success BOOLEAN, target TEXT, comment TEXTE);"

# Define some useful stuff

#ssh options
ssh_options="-o ControlPath=\"$HOME/.ssh/ctl/%L-%r@%h:%p\""
if [ -n "$ssh_key_rsync" ]; then
  ssh_options+=" -i ${ssh_key_rsync}"
fi

# Last date of files already registered
last_transfer=$(sqlite3 $dbfile "select max(date) from gfiles;")
last_transfer=$(( last_transfer > first_transfer ? last_transfer : first_transfer ))

#tag to identify files treated in the current run
tag=$(date +'%Y%m%d%H%M%S')

# Colors
Default='\033[0m'	# Text Reset
Red='\033[0;31m'	# Red
Green='\033[0;32m'	# Green

# run pre script
if [ -n "$pre_run_script" ]
then
  pre=$($pre_run_script)
  ret=$?
  if [ "$ret" -ne "0" ]; then
    printf "Error ${ret} in pre run script : ${pre} \n"
    exit ${ret}
  fi
fi

#List of files to be inserted into the db by bunchs of 500 (larger should produce errors)
declare -A toins=([1]="")
i=1
j=0
md5="0"
#find all files in localdatadir corresponding to datas (i.e. name starting by site) and add them to the database if not here already
for file in $(find $localdatadir -type f -newermt $last_transfer| grep /${site}_ |sort)
do
  # skip opened files
  if [ !$(fuser "$file" &> /dev/null) ]; then
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
    toins[$i]+=",('$(dirname $file)', '$(basename $file)', ${dateobs}, 0, '${md5}')"
    j=$((j + 1))
  fi
done

#Add all files at a time (10x faster that adding them individually)
# but iterate over various lists in case of huge number of files
for key in "${!toins[@]}"; do
	value=${toins[${key}]}
	if [ -n "$value" ];
	then
		res=$(sqlite3 $dbfile "INSERT OR IGNORE INTO gfiles (directory,file,date,success,md5sum) values ${value:1}")
	fi
done

# Open a ssh connection that will be used for all transfers (avoid to reopen rsync tunnel for each file)
mkdir ~/.ssh/ctl >/dev/null 2>&1
ssh -nNf -o ControlMaster=yes ${ssh_options} ${remote_account}@${remote_server}
declare -A translog=([1]="")
i=1
j=0
#select files not transfered successfully
for file in $(sqlite3 $dbfile "select directory, file, date, success, id from gfiles where success=0 ORDER BY file;")
do
	#transform result into array (more easy to manipulate)
	fileinfo=(${file//|/ })
	# Ensure extension is .bin
  finalname="${fileinfo[1]%.*}.bin"
	#Transfer files (one by one to get info on each transfer)
	printf "\nSending ${fileinfo[1]} "
	trans=$(rsync -e "ssh ${ssh_options}" --out-format="%t %b md5:%C"  ${rsync_options} --rsync-path="mkdir -p $remotedatadir/raw/${fileinfo[2]:0:4}/${fileinfo[2]:4:2} && rsync" ${fileinfo[0]}/${fileinfo[1]}  ${remote_account}@${remote_server}:$remotedatadir/raw/${fileinfo[2]:0:4}/${fileinfo[2]:4:2}/${finalname} 2>&1)
	if [ "$?" -eq "0" ]
	then
	  md5=${trans#*md5:}
	  #echo $md52
		#Transfer successful : store info to update database at the end
		translog[$i]+=";UPDATE gfiles SET success=1, md5sum='${md5}' WHERE id=${fileinfo[4]};INSERT INTO transfer (id, tag, success,date_transfer,target,comment) VALUES (${fileinfo[4]},${tag}, 1,datetime('now','utc'), \"${remotedatadir}/raw/${fileinfo[2]:0:4}/${fileinfo[2]:4:2}/${finalname}\", '${trans}')"
		printf "${Green}Ok${Default}"

	else
	  md5=$(echo ${trans}|awk -F"md5:" '{print $2}')
		#Transfer failed : just log errors
		translog[$i]+=";INSERT INTO transfer (id, tag, success, date_transfer, target, comment) VALUES (${fileinfo[4]}, ${tag}, 0,datetime('now','utc'), '${remotedatadir}/raw/${fileinfo[2]:0:4}/${fileinfo[2]:4:2}/${finalname}', '${trans}')"
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
rsync -e "ssh ${ssh_options}" ${rsync_options} --rsync-path="mkdir -p $remotedatadir/database && rsync" $dbfile ${remote_account}@${remote_server}:$remotedatadir/database/${tag}_${dbfile}

#close ssh connection
ssh -O exit $ssh_options ${remote_account}@${remote_server}
rm -rf ~/.ssh/ctl

# run post script
if [ -n "$post_run_script" ]
then
  post=$($post_run_script)
  ret=$?
  if [ "$ret" -ne "0" ]; then
    printf "Error ${ret} in post run script : ${post} \n"
    exit ${ret}
  fi
fi

#Run conversion scripts @ccin2p3
if [ -n "$ccscripts" ]
then
  ssh -i ${ssh_key_exec} ${remote_account}@${remote_server} ${ccscripts} -d ${remotedatadir}/database/${tag}_${dbfile} -t ${tag}
fi