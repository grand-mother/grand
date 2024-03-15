#!/bin/bash
# path to bin2root file
bin2root='/pbs/home/p/prod_grand/scripts/transfers/bintoroot.bash'

# gtot options for convertion
gtot_option="-g1"

# number of files to group in same submission
nbfiles=3


# Get tag and database file to use
while getopts ":t:d:" option; do
   case $option in
      t) 
         tag=${OPTARG};;
      d) 
         db=${OPTARG};;
      :) 
         printf "option -${OPTARG} need an argument\n"
	 exit 1;;	
      ?) # Invalid option
         printf "Error: Invalid option -${OPTARG}\n"
         exit 1;;
   esac
done


#test dbfile exists and tag is set
if [ -z "$tag" ] || [ -z "$db" ];then 
	printf "Missing option -t or -d\n"
	exit 1
elif [ ! -f $db ];then
	printf "Database file does not exists\n"
	exit 1
fi

# Determine root_dri from database path
root_dest=${db%/database*}/GrandRoot/
submit_dir=${db%/database*}/logs/
submit_base_name=submit_${tag}
if [ ! -d $root_dest ];then
        	mkdir -p $root_dest >/dev/null 2>&1
fi
if [ ! -d $submit_dir ];then
                mkdir -p $submit_dir >/dev/null 2>&1
fi


i=1
j=1
listoffiles=""

echo "loop"
for file in $(sqlite3 $db "select target from transfer,gfiles where gfiles.id=transfer.id and tag='${tag}' and transfer.success=1;")
do
	echo $file
	#define the submission and log files
	outfile="${submit_dir}/${submit_base_name}-${j}.bash"
	logfile="${submit_dir}/${submit_base_name}-${j}.log"
	
	#add file to the list of files to be treated
	listoffiles+=" ${file}"
	
	# When reach the number of files to treat in a run then write the submission file and submit it
	if [ "$i" -eq "${nbfiles}" ];then
		echo "#!/bin/bash" > $outfile
		# add conversion from bin to Grandroot
		echo "$bin2root -g '$gtot_option' -d $root_dest $listoffiles" >> $outfile
		
		#submit script
		sbatch -t 0-08:30 -n 1 -J ${submit_base_name}-${j} -o ${submit_dir}/slurm-${submit_base_name}-${j} --mem 8G $outfile
		# reset iterators and list of files
		i=0
                ((j++))
		listoffiles=""
	fi
	((i++))
done

