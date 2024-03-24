#!/bin/bash
# path to bin2root file
bin2root='/pbs/home/p/prod_grand/scripts/transfers/bintoroot.bash'


# gtot options for convertion -g1 for gp13 -f2 for gaa
gtot_option="-g1"

# number of files to group in same submission
nbfiles=3

# manage call from remote restricted ssh command (extracr opt parameters)
# default args
fullscriptpath=${BASH_SOURCE[0]}
args="$*"
case $SSH_ORIGINAL_COMMAND in
    "$fullscriptpath "*)
        args=$(echo "${SSH_ORIGINAL_COMMAND}" | sed -e "s,^${fullscriptpath} ,,")
        ;;
    *)
        echo "Permission denied."
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

for j in  "${!listoffiles[@]}"
do
  outfile="${submit_dir}/${submit_base_name}-${j}.bash"
	logfile="${submit_dir}/${submit_base_name}-${j}.log"
	echo "#!/bin/bash" > $outfile
	echo "$bin2root -g '$gtot_option' -d $root_dest ${listoffiles[$j]}" >> $outfile
	#submit script
	echo "submit  $outfile"
	sbatch -t 0-01:00 -n 1 -J ${submit_base_name}-${j} -o ${submit_dir}/slurm-${submit_base_name}-${j} --mem 8G $outfile
done


