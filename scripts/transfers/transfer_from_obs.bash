#!/bin/bash
# Script to transfert data from a GRAND observatory to CCIN2P3 (or to any site)
# Fleg & Fred: 03/2024
# Copyright : Grand Observatory 2024

##### Configuration part #####
# Please adjust the following variable to your site

# Site name prefix in filenames
# 'GP13' for gp13
# gaa for gaa
site='GP13'

# Start date for transfer (all files older than this date will be skipped
first_transfer='20240314'

# skip open files (if set to false the opened files will be transfered into /tmp)
declare -A defskipopenfiles
defskipopenfiles["gaa"]=false
defskipopenfiles["GP13"]=true
skipopenfiles=${defskipopenfiles[$site]}

# Local database name (sqlite filename)
# '/home/grand/data-transfer/grand_transfer.db' for gaa
#
declare -A defdbfile
defdbfile["gaa"]='/home/grand/data-transfer/grand_transfer.db'
defdbfile["GP13"]='grand_transfer_Since202409.db'
dbfile=${defdbfile[$site]}

# Local directory where are stored the data to be transfered (will be explored recursively)
# '/home/mapx/mapx/DunhuangData/Rawdata/20dB/' for gp13
# '/home/grand/Malargue/' for gaa
declare -A deflocaldatadir
deflocaldatadir["gaa"]='/data/Malargue/'
deflocaldatadir["GP13"]='/share03/users/m/mapx/cc-in2p3/'
localdatadir=${deflocaldatadir[$site]}

#path to local rsync to use (leave blank if your default rsync --version >= 3.2.3
# define the path to rsync version >= 3.2.3 otherwise
rsyncpath=''

# Remote server to transfer
remote_server='cca.in2p3.fr'

# Account on remote server
remote_account='prod_grand'

#ssh key for rsync
# "/home/mapx/.ssh/id_ed25519" for gp13
# "/root/.ssh/id_ed25519-nopw" for gaa
declare -A defssh_key_rsync
defssh_key_rsync["gaa"]="/root/.ssh/id_ed25519-nopw"
defssh_key_rsync["GP13"]="/home/mapx/.ssh/id_ed25519-rsync"
ssh_key_rsync=${defssh_key_rsync[$site]}

#ssh key for exec remote scripts
# "/home/mapx/.ssh/id_ed25519-scrips" for gp13
# "/root/.ssh/id_ed25519-nopw-scripts" for gaa
declare -A defssh_key_exec
defssh_key_exec["gaa"]="/root/.ssh/id_ed25519-nopw-scripts"
defssh_key_exec["GP13"]="/home/mapx/.ssh/id_ed25519-scripts"
ssh_key_exec=${defssh_key_exec[$site]}

# Target directory on remote server, must be the same directory as the one defined in remote_account@remote_server:~/.ssh/authorized_keys for ssh_key_rsync
# '/sps/grand/data/gp13' for gp13
# '/sps/grand/data/gaa' for gaa
declare -A defremotedatadir
defremotedatadir["gaa"]='/sps/grand/data/gaa'
defremotedatadir["GP13"]='/sps/grand/data/gp13'
remotedatadir=${defremotedatadir[$site]}


# Local script to be launched before run
# '' for gp13
#'setup_network_auger.bash -init' for gaa
declare -A defpre_run_script
defpre_run_script["gaa"]='/root/bin/setup_network_auger.bash -init'
defpre_run_script["GP13"]=''
pre_run_script=${defpre_run_script[$site]}

# Local script to be launched after run
# '' for gp13
# '/root/bin/setup_network_auger.bash -close' for gaa
declare -A defpost_run_script
defpost_run_script["gaa"]='/root/bin/setup_network_auger.bash -close'
defpost_run_script["GP13"]=''

# rsync_options : a to keep the creation time of files, z to compress if bandwidth is limited (but it's ~5 times slower). Please keep the "a" option  !
rsync_options="-az --mkpath --chmod=go-w"

##### End of Configuration section (do not modify below) #####

# treatment scripts location @CCIN2P3
ccscripts='/pbs/home/p/prod_grand/scripts/transfers/ccscript.bash'

# Create database if not exists
sqlite3 $dbfile "create table if not exists  gfiles (id INTEGER PRIMARY KEY AUTOINCREMENT, directory TEXT, file TEXT, date INT, success BOOLEAN, md5sum VARCHAR(35), UNIQUE (directory,file));"
sqlite3 $dbfile "create table if not exists  transfer (id, tag INTEGER, date_transfer DATETIME, success BOOLEAN, target TEXT, comment TEXTE);"

# Define some useful stuff
#
if [ -n "$rsyncpath" ] && [ -f $rsyncpath ];
then
  rsync_command=$rsyncpath
else
  rsync_command=$(which rsync)
fi
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
declare -a openfiles=() ##tmp
#find all files in localdatadir corresponding to datas (i.e. name starting by site) and add them to the database if not here already
for file in $(find $localdatadir -type f -newermt $last_transfer| grep /${site}_ |sort)
do
  # skip opened files
  if  ! $(fuser "$file" &> /dev/null) ; then
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
  else
    openfiles+=("$file") ##tmp
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

# Open a ssh connection that will be used for all transfers (avoid to reopen rsync tunnel for each file) for rsync
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
	# no remotedatadir specified on rsync command line, as it's already the defined restricted directory in ccin2p3:~/.ssh/authorized_keys
  # /raw directory in the rsync thus has /sps/grand/data/gaa as root dir
  trans=$(${rsync_command} -e "ssh ${ssh_options}" --out-format="%t %b md5:%C"  ${rsync_options} ${fileinfo[0]}/${fileinfo[1]}  ${remote_account}@${remote_server}:/raw/${fileinfo[2]:0:4}/${fileinfo[2]:4:2}/${finalname} 2>&1)
  #trans=$(rsync -e "ssh ${ssh_options}" --out-format="%t %b md5:%C"  ${rsync_options} --rsync-path="mkdir -p $remotedatadir/raw/${fileinfo[2]:0:4}/${fileinfo[2]:4:2} && rsync" ${fileinfo[0]}/${fileinfo[1]}  ${remote_account}@${remote_server}:$remotedatadir/raw/${fileinfo[2]:0:4}/${fileinfo[2]:4:2}/${finalname} 2>&1)
	if [ "$?" -eq "0" ]
	then
	  md5=${trans#*md5:}
	  #echo $md52
		#Transfer successful : store info to update database at the end
		translog[$i]+=";UPDATE gfiles SET success=1, md5sum='${md5}' WHERE id=${fileinfo[4]};INSERT INTO transfer (id, tag, success,date_transfer,target,comment) VALUES (${fileinfo[4]},${tag}, 1,datetime('now','utc'), \"${remotedatadir}/raw/${fileinfo[2]:0:4}/${fileinfo[2]:4:2}/${finalname}\", '${trans}')"
		# colors only for terminal
		if [ -n "$TERM" ]; then
		  printf "${Green}Ok${Default}"
		else
		  printf "Ok"
		fi
	else
	  md5=$(echo ${trans}|awk -F"md5:" '{print $2}')
		#Transfer failed : just log errors
		translog[$i]+=";INSERT INTO transfer (id, tag, success, date_transfer, target, comment) VALUES (${fileinfo[4]}, ${tag}, 0,datetime('now','utc'), '${remotedatadir}/raw/${fileinfo[2]:0:4}/${fileinfo[2]:4:2}/${finalname}', '${trans}')"
		if [ -n "$TERM" ]; then
		  printf "${Red}ERROR:${Default} \n ${trans} "
		else
		  printf "ERROR: \n ${trans} "
		fi
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
# To avoid to transfer the whole database, we will purge the database from old files (already transfered)
# Get the list of files successfully transfered before the last transfer
list_id=$(sqlite3 $dbfile "select id from gfiles where success==1 and date<$last_transfer")
# Convert it into an array
ids=()
for id in $list_id; do ids+=($id) ; done
# Perform the deletion by groups of 500 files
CHUNK_SIZE=500
index=0
while [[ $index -lt ${#ids[@]} ]]; do
        nids=$(echo "${ids[@]:$index:$CHUNK_SIZE}" | tr ' ' ',')
        sqlite3 $dbfile "delete from transfer where id in ($nids); delete from gfiles where id in ($nids)"
        index=$((index + CHUNK_SIZE))
done
# finally create a new DB with only the last files. We set the remote db name to ${tag}_${site}_dbfile.db
dirdb=$(dirname $dbfile)
dbname="${tag}_${site}_dbfile.db"

sqlite3 $dbfile "vacuum main into '${dirdb}/${dbname}'"

# no remotedatadir specified on rsync command line, as it's already the defined restricted directory in ccin2p3:~/.ssh/authorized_keys
# /raw directory in the rsync thus has /sps/grand/data/gaa as root dir
${rsync_command} -e "ssh ${ssh_options}" ${rsync_options} ${dirdb}/${dbname} ${remote_account}@${remote_server}:/logs/${dbname}

# sync openfiles into tmp ##tmp
if [ "$skipopenfiles" = false ]; then
  for file in ${openfiles[*]}; do
	  ${rsync_command} -v -e "ssh ${ssh_options}" ${rsync_options} $file ${remote_account}@${remote_server}:/tmp/
  done
fi

#close ssh connection
ssh -O exit $ssh_options ${remote_account}@${remote_server}
rm -rf ~/.ssh/ctl

#Run conversion scripts @ccin2p3
if [ -n "$ccscripts" ]
then
  ssh -i ${ssh_key_exec} ${remote_account}@${remote_server} ${ccscripts} -s ${site} -d ${remotedatadir}/logs/${dbname} -t ${tag}
fi

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

