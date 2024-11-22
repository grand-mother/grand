#!/bin/bash -l
# Script triggered after transfering data from a GRAND observatory to CCIN2P3 (or to any site)
# It will launch the jobs to convert binary files into GrandRoot and register the results of the transfers and convertions into the database
# Fleg & Fred: 03/2024
# update Fleg : 11/2024 (option to run locally + gp80 management)
# Copyright : Grand Observatory 2024

# path to bin2root file
bin2root='/pbs/home/p/prod_grand/scripts/transfers/bintoroot.bash'
register_transfers='/pbs/home/p/prod_grand/scripts/transfers/register_transfer.bash'
refresh_mat_script='/pbs/home/p/prod_grand/scripts/transfers/refresh_mat_views.bash'
update_web_script='/sps/grand/prod_grand/monitoring_page/launch_webmonitoring_update.bash'
tar_logs_script='/pbs/home/p/prod_grand/scripts/transfers/tar_logs.bash'
# gtot options for convertion -g1 for gp13 -f2 for gaa
gtot_option="-g1 -os"

# number of files to group in same submission
nbfiles=3

#time required to run bin2root on one file
bin2rootduration=15

# Notification options q
mail_user='fleg@lpnhe.in2p3.fr'
mail_type='FAIL,TIME_LIMIT,INVALID_DEPEND'

fullscriptpath=${BASH_SOURCE[0]}
args="$*"
#Check how the script is launched (local execution is authorized and sshd exec is restricted to authorized keys)
access_type=$(ps h -o comm -p "$PPID")
if [ "${access_type}" == "sshd" ] ; then
  # manage call from remote restricted ssh command (extract opt parameters)
  # default args
  case $SSH_ORIGINAL_COMMAND in
      "$fullscriptpath "*)
          args=$(echo "${SSH_ORIGINAL_COMMAND}" | sed -e "s,^${fullscriptpath} ,,")
          ;;
      *)
          echo "Permission denied. You are not authorized to run ${fullscriptpath}. Check ssh key ?"
          exit 1
          ;;
  esac
elif [ "${access_type}" == "bash" ] ; then
        echo "Local execution authorized"
else
        echo "Permission denied."
fi



# Get tag and database file to use
while getopts ":t:d:s:" option ${args}; do
   case $option in
      t) 
         tag=${OPTARG};;
      d) 
         db=${OPTARG};;
      s)
        site=${OPTARG};;
      :)
         printf "option -${OPTARG} need an argument\n"
         exit 1;;
      ?) # Invalid option
         printf "Error: Invalid option -${OPTARG}\n"
         exit 1;;
   esac
   if [ "${OPTARG:0:1}" == "-" ]; then
           printf "option -${option} need an argument\n"
           exit 1
   fi
done

case ${site,,} in
  gp13)
    gtot_option="-g1 -rn -os";;
  gp80)
    gtot_option="-g1 -os -rn";;
  gaa)
    gtot_option="-f2 -os";;
  ?)
    gtot_option="-g1 -os";;
esac

#test dbfile exists and tag is set
if [ -z "$tag" ] || [ -z "$db" ];then 
	printf "Missing option -t or -d\n"
	exit 1
elif [ ! -f $db ];then
	printf "Database file does not exists\n"
	exit 1
fi

# Determine root_dir from database path
root_dest=${db%/logs*}/GrandRoot/
submit_dir=$(dirname "${db}")
submit_base_name=${site}_${tag}
crap_dir=${db%/logs*}/raw/crap

if [ ! -d $root_dest ];then
        	mkdir -p $root_dest >/dev/null 2>&1
fi
if [ ! -d $submit_dir ];then
                mkdir -p $submit_dir >/dev/null 2>&1
fi
if [ ! -d $crap_dir ];then
                mkdir -p $crap_dir >/dev/null 2>&1
fi
# First register raw files transfers into the DB and get the id of the registration job
outfile="${submit_dir}/${submit_base_name}-register-transfer.bash"
echo "#!/bin/bash" > $outfile
echo "$register_transfers -d $db -t $tag" >> $outfile
jregid=$(sbatch -t 0-00:20 -n 1 -J ${submit_base_name}-register-transfer -o ${submit_dir}/${submit_base_name}-register-transfer.log --mem 1G --constraint el9 --mail-user=${mail_user} --mail-type=${mail_type} ${outfile} )
jregid=$(echo $jregid |awk '{print $NF}')

# List files to be converted and group them by bunchs of nbfiles
i=0
j=0
declare -A listoffiles
for file in $(sqlite3 $db "select target from transfer,gfiles where gfiles.id=transfer.id and tag='${tag}' and transfer.success=1;")
do
  # We exclude small files (which are suposed to be crap)
  fsize=$(stat -c%s "$file")
  if [ "$fsize" -le "256" ];then
      echo "$file too small ($fsize). Moved to $crap_dir/ and skipped."
      mv "$file" "$crap_dir/"
  else
      if [ "$((i % nbfiles))" -eq "0" ]; then
        ((j++))
      fi
      #add file to the list of files to be treated
      listoffiles[$j]+=" ${file}"
      ((i++))
  fi
done

jobtime=`date -d@$(($bin2rootduration*60*$nbfiles))  -u +%H:%M`
convjobs=""
# Launch convertion of files (but after the registration has finished)
for j in  "${!listoffiles[@]}"
do
  outfile="${submit_dir}/${submit_base_name}-${j}.bash"
	logfile="${submit_dir}/${submit_base_name}-${j}.log"
	echo "#!/bin/bash" > $outfile
	echo "$bin2root -g '$gtot_option' -n $submit_base_name -d $root_dest ${listoffiles[$j]}" >> $outfile
	#submit script
	echo "submit  $outfile"
	jid=$(sbatch --dependency=afterany:${jregid} -t 0-${jobtime} -n 1 -J ${submit_base_name}-${j} -o ${submit_dir}/${submit_base_name}-${j}.log --mem 2G --constraint el9 --mail-user=${mail_user} --mail-type=${mail_type} ${outfile} )
  jid=$(echo $jid |awk '{print $NF}')
  convjobs=$convjobs":"$jid
done

if [ "$convjobs" = "" ]; then
  dep=""
else
  dep="--dependency=afterany${convjobs}"
  #finally refresh the materialized views in the database and the update of monitoring
  sbatch ${dep} -t 0-00:15 -n 1 -J refresh_mat_${tag} -o ${submit_dir}/refresh_mat_${tag}.log --mem 1G --constraint el9 --mail-user=${mail_user} --mail-type=${mail_type} ${refresh_mat_script}
  sbatch ${dep} -t 0-01:30 -n 1 -J update_webmonitoring_${tag} -o ${submit_dir}/update_webmonitoring_${tag}.log --mem 16G --constraint el9 --mail-user=${mail_user} --mail-type=${mail_type} ${update_web_script}
  sbatch -t 0-00:55 -n 1 -J tar_logs_${tag} -o ${submit_dir}/tar_logs_${tag}.log  --mem 1G --mail-user=${mail_user} --mail-type=${mail_type}  --wrap="${tar_logs_script} -s ${site,,} -d 2"
fi

