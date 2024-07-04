#!/bin/bash

# path to gtot
gtot_path='/pbs/home/p/prod_grand/softs/gtot/cmake-build-release/gtot'
# path to script to register convertion results
register_convertion='/pbs/home/p/prod_grand/scripts/transfers/register_convert.py'
# path to script to register root file into the DB
register_root='/pbs/home/p/prod_grand/softs/grand/granddb/register_file_in_db.py'
config_file='/pbs/home/p/prod_grand/softs/grand/scripts/transfers/config-prod.ini'
sps_path='/sps/grand/'
irods_path='/grand/home/trirods/'

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


notify=0
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
  # Convert file
  ${gtot_path} ${gtot_options} -i ${file} -o ${dest}/${filename%.*}.root >> ${logfile}
  conv_status=$?
  if [ "$conv_status" -ne 0 ]; then
    notify=1
  fi
  echo $conv_status >> ${logfile}
  # Put GrandRoot file into irods
  sfile=${dest}/${filename%.*}.root
  ifile=${sfile/$sps_path/$irods_path}
  ipath=${ifile%/*}
  echo "imkdir -p $ipath" >> ${logfile}
  imkdir -p $ipath >> ${logfile} 2>&1
  echo "iput -f $sfile $ifile" >> ${logfile}
  iput -f $sfile $ifile >> ${logfile} 2>&1
  iput_status=$?
  if [ "$iput_status" -ne 0 ]; then
    notify=1
  fi
  # Register conversion result into the database
  python3 ${register_convertion} -i ${filename} -o ${filename%.*}.root -s ${conv_status} -l ${logfile}
  # Register root file into db
  python3 ${register_root} -c ${config_file} -r "CCIN2P3" ${dest}/${filename%.*}.root
done

if [ "$notify" -ne "0" ]; then
  echo "Error in files conversion : " $@ |   mail -s "Grand conversion error" fleg@lpnhe.in2p3.fr
fi
