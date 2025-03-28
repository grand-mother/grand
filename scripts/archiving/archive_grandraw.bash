#!/bin/bash
# Script to archive Grand raw files (age = 2 month ago) and push the archive into irods
# Fleg, Oct 2024

# Define useful variables
datadir="/sps/grand/data"
archive_root_dir="/sps/grand/prod_grand/archiving"
archive_root_name="doi+10.25520+in2p3.archive.grand"
irods_path='/grand/home/trirods/data/archives/'
representation="/representations/representation1/data"
# The former script to create archive needed java 8 (some used libs are not available in java versions > 8) but is now corrected
javabin='/usr/lib/jvm/jre-1.8.0-openjdk/bin/java'
#javabin='java'

delay=""

while getopts ":d:" option; do
   case $option in
      d)
         delay=${OPTARG};;
      :)
         printf "option -${OPTARG} need an argument\n"
         exit 1;;
      ?) # Invalid option
         printf "Error: Invalid option -${OPTARG}\n"
         exit 1;;
   esac

done
if [ -z "$delay" ]; then
    echo "Error: The -m option is missing. We will use the default value = 2"
    delay=2
fi

# Get the year and month for 2 month ago
read year month << DATE_COMMAND
 $(date --date="TODAY -${delay} month" "+%Y %m")
DATE_COMMAND

# Define dir to search
dir="${year}/${month}"
date="${year}_${month}"

# Loop over site data directories
#dir="2023/12"
#site="gaa"
for site in gaa gp13 gp80
do
  outfile=${archive_root_name}.${site}.${date}
	outdir="${archive_root_dir}/${site}/${outfile}"
	logfile=${outdir}-$(date "+%Y_%m_%d_%H%M%S").log
	fileslist=${archive_root_dir}/${site}/list_files_${site}.${date}
	sourcedir=${datadir}/${site}/raw/${dir}/
	parentdir=$(dirname "$sourcedir")

	flagarchived="ARCHIVED"

	#Check not yet archived
  if [ -e "${sourcedir}${flagarchived}" ]; then
    echo "${sourcedir} seems already archived... skip" |tee -a ${logfile}
    continue
  fi


  # ensure that directory exists
	mkdir -p ${archive_root_dir}/${site} > /dev/null 2>&1

  # Check that file containing the list of files to archive (${fileslist}) does not exists
  # if it exists it should mean that another process is still running, so skip
  if [ -f ${fileslist} ]; then
    echo "${fileslist} exists... skip"
    continue
  fi
  # touch file immediately to "lock" the process (find command should last)
  touch ${fileslist}
  #find ${datadir}/${site}/raw/${dir}/ -name "*.bin" >> ${fileslist}
  find ${sourcedir} -name "*.bin" >> ${fileslist}
  list=$(cat ${fileslist})

  # If no files to archive then skip
  if [ "${list}" == "" ]; then
    echo "No files in ${fileslist} ${list}...skip"
    rm ${fileslist}
    continue
  else
    echo "Archiving $month $year for $site"
    echo "Archiving $month $year for $site" >> ${logfile}
  fi
  # Create the archive
  echo "$javabin -jar createAIP.jar --configfile=config.properties.${site} --listobjects=${fileslist} -i ${outfile} >> ${logfile} 2>&1"
  $javabin -jar createAIP.jar --configfile=config.properties.${site} --listobjects=${fileslist} -i ${outfile} >> ${logfile} 2>&1
  createaip_status=$?

  if [ "$createaip_status" -eq 0 ]; then
    #link the dir to be archived
    mkdir -p ${outdir}/${representation}/${parentdir}
    ln -s ${sourcedir} ${outdir}/${representation}/${parentdir}
    echo "Archive ready to tar" >> ${logfile}
    tar -chf ${outdir}.tar ${outdir}
    tar_status=$?
    if [ "$tar_status" -eq 0 ]; then
      echo "Archive tared" >> ${logfile}
      echo "Push ${outdir}.tar to irods" >> ${logfile}
      # Push file into irods
      sfile=${outdir}.tar
      ipath="${irods_path}${site}/raw"
      ifile="${ipath}/${outfile}.tar"
      echo "imkdir -p $ipath" >> ${logfile}
      imkdir -p $ipath >> ${logfile} 2>&1
      echo "iput -f $sfile $ifile" >> ${logfile}
      iput -f $sfile $ifile >> ${logfile} 2>&1
      iput_status=$?

      if [ "$iput_status" -eq 0 ]; then
        #clean everything
        echo "remove temp dir ${outdir}" >> ${logfile}
        rm -rf ${outdir} >> ${logfile} 2>&1
        echo "remove ${outdir}.tar" >> ${logfile}
        rm ${outdir}.tar >> ${logfile} 2>&1
        echo "Raw data of ${year}/${month} from ${site} archived " >> ${logfile}
        touch ${sourcedir}${flagarchived}

        #echo "compress files" >> ${logfile}
        #while IFS= read -r line
        #do
        #  echo "gzip ${line}"
        #  gzip $line
        #done < "${fileslist}"

        rm ${fileslist}

      else
        echo "Problem transfering archive to irods" >> ${logfile}
      fi
    else
      echo "Problem taring archive" >> ${logfile}
    fi
  else
    echo "Problem creating archive" >> ${logfile}
  fi


done


