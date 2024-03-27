#!/bin/bash

# path to gtot
gtot_path='/pbs/home/p/prod_grand/softs/gtot/cmake-build-release/gtot'

# Get tag and database file to use
while getopts ":d:g:" option; do
  case $option in
    d)
      root_dest=${OPTARG};;
    g)
      gtot_options=${OPTARG};;
    :)
      printf "option -${OPTARG} need an argument\n"
      exit 1;;
    ?) # Invalid option
      printf "Error: Invalid option -${OPTARG}\n"
      exit 1;;
  esac
done

shift $(($OPTIND - 1))

cd /pbs/home/p/prod_grand/softs/grand 
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh 
conda activate /sps/grand/software/conda/grandlib_2304 
source env/setup.sh 
cd /pbs/home/p/prod_grand/scripts/transfers



for file in "$@"
do
	echo "converting ${file} to GrandRoot"
	filename=$(basename $file)
	tmp=${filename#*_}
	dateobs=${tmp:0:8}
	dest="${root_dest}/${dateobs:0:4}/${dateobs:4:2}"
	if [ ! -d $dest ];then
		mkdir -p $dest >/dev/null 2>&1
	fi
	dirlogs=${root_dest}/../logs
	logfile=${dirlogs}/bin2root-${filename%.*}
	if [ ! -d $dirlogs  ];then
		mkdir -p $dirlogs >/dev/null 2>&1
	fi
	${gtot_path} ${gtot_options} -i ${file} -o ${dest}/${filename%.*}.root >> ${logfile}
	echo $? >> ${logfile}
done

