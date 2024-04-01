#!/bin/bash
# Script triggered after transfering data from a GRAND observatory to CCIN2P3 (or to any site)
# It will launch the jobs to convert binary files into GrandRoot and register the results of the transfers and convertions into the database
# Fleg & Fred: 03/2024
# Copyright : Grand Observatory 2024

# path to bin2root file
bin2root='/pbs/home/p/prod_grand/scripts/transfers/bintoroot.bash'
register_transfers='/pbs/home/p/prod_grand/scripts/transfers/register_transfer.bash'
refresh_mat_script='/pbs/home/p/prod_grand/scripts/transfers/refresh_mat_views.bash'
# gtot options for convertion -g1 for gp13 -f2 for gaa
gtot_option="-g1"

# number of files to group in same submission
nbfiles=3

# manage call from remote restricted ssh command (extract opt parameters)
# default args
fullscriptpath=${BASH_SOURCE[0]}
args="$*"
case $SSH_ORIGINAL_COMMAND in
    "$fullscriptpath "*)
        args=$(echo "${SSH_ORIGINAL_COMMAND}" | sed -e "s,^${fullscriptpath} ,,")
        ;;
    *)
        echo "Permission denied. You are not authorized to run ${fullscriptpath}. Check ssh key ?"
        exit 1
        ;;
esac

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
done

case $site in
  gp13)
    gtot_option="-g1";;
  gaa)
    gtot_option="-v2";;
  ?)
    gtot_option="-g1";;
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
submit_base_name=submit_${tag}

if [ ! -d $root_dest ];then
        	mkdir -p $root_dest >/dev/null 2>&1
fi
if [ ! -d $submit_dir ];then
                mkdir -p $submit_dir >/dev/null 2>&1
fi

# First register raw files transfers into the DB and get the id of the registration job
outfile="${submit_dir}/${submit_base_name}-register-transfer.bash"
echo "#!/bin/bash" > $outfile
echo "$register_transfers -d $db -t $tag" >> $outfile
jregid=$(sbatch -t 0-01:00 -n 1 -J ${submit_base_name}-register-transfer -o ${submit_dir}/slurm-${submit_base_name}-register-transfer --mem 8G ${outfile})
jregid=$(echo $jregid |awk '{print $NF}')

# List files to be converted and group them by bunchs of nbfiles
i=0
j=0
declare -A listoffiles
for file in $(sqlite3 $db "select target from transfer,gfiles where gfiles.id=transfer.id and tag='${tag}' and transfer.success=1;")
do
  if [ "$((i % nbfiles))" -eq "0" ]; then
    ((j++))
  fi

	#add file to the list of files to be treated
	listoffiles[$j]+=" ${file}"

  ((i++))
done

convjobs=""
# Launch convertion of files (but after the registration has finished)
for j in  "${!listoffiles[@]}"
do
  outfile="${submit_dir}/${submit_base_name}-${j}.bash"
	logfile="${submit_dir}/${submit_base_name}-${j}.log"
	echo "#!/bin/bash" > $outfile
	echo "$bin2root -g '$gtot_option' -d $root_dest ${listoffiles[$j]}" >> $outfile
	#submit script
	echo "submit  $outfile"
	jid=$(sbatch --dependency=afterany:${jregid} -t 0-01:00 -n 1 -J ${submit_base_name}-${j} -o ${submit_dir}/slurm-${submit_base_name}-${j} --mem 8G ${outfile})
  jid=$(echo $jid |awk '{print $NF}')
  convjobs=$convjobs":"$jid
done

if [ "$convjobs" -eq "" ]; then
  dep=""
else
  dep="--dependency=afterany${convjobs}"
fi
#finally refresh the materialized views in the database
sbatch ${dep} -t 0-01:00 -n 1 -J refresh_mat -o ${submit_dir}/slurm-refresh_mat --mem 1G ${refresh_mat_script}
