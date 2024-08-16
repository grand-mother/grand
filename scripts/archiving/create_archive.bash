#!/bin/bash
datadir="/sps/grand/data"
archive_root_name="doi+10.25520+in2p3.archive.grand"
irods_path='/grand/home/trirods/data/archives/'

usage="$(basename "$0") [-d DATE] [-s SITE] [
Archive some Grand raw files into irods :
    -s  site (gaa, gp13)
    -d  YYYY-MM to be archived
    "

while getopts "d:s:" option ${args}; do
  case $option in
    d)
      if [[ ${OPTARG} =~ ^([0-9]{4})-([0][1-9]|[1][0-2]|[1-9])$ ]]; then
        date=$(date --date="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-01" "+%Y_%m")
        dir=$(date --date="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-01" "+%Y/%m")
	    else
        echo "Date ${OPTARG} should be in format YYYY-MM"
        exit 1
	    fi
      ;;
    s)
	    if [[ ${OPTARG} =~ gp13|gaa ]] ; then
        site=${OPTARG}
	    else
		    echo "Site should be gp13 or gaa"
		    exit 1
	    fi
	    ;;
    :)
      printf "option -${OPTARG} need an argument\n"
      exit 1;;
    ?) # Invalid option
      printf "Error: Invalid option -${OPTARG}\n"
      exit 1;;
  esac
done

if [ ! "$date" ] || [ ! "$site" ]; then
  echo "arguments -d and -s must be provided"
  echo "$usage" >&2; exit 1
fi

outfile="${archive_root_name}.${site}.${date}"
logfile=archs/${site}/${outfile}--$(date "+%Y_%m_%d_%H%M%S").log

find $datadir/$site/raw/$dir/ -name "*.bin" >list_files_${site}
echo "List of files to archive :" >> ${logfile}
cat list_files_${site} >> ${logfile}

java -jar createAIP.jar --configfile=config.properties.${site} --listobjects=list_files_${site} -i ${outfile}

echo "Archive ready to tar" >> ${logfile}

tar -cvf archs/${site}/${outfile}.tar archs/${site}/${outfile}

echo "Archive tared" >> ${logfile}

echo "Push archs/${site}/${outfile}.tar to irods" >> ${logfile}
# Put file into irods
  sfile=archs/${site}/${outfile}.tar
  ipath="${irods_path}${site}/raw"
  ifile="${ipath}/${outfile}.tar"
  echo "imkdir -p $ipath" >> ${logfile}
  imkdir -p $ipath >> ${logfile} 2>&1
  echo "iput -f $sfile $ifile" >> ${logfile}
  #iput -f $sfile $ifile >> ${logfile} 2>&1
  #iput_status=$?
  #if [ "$iput_status" -ne 0 ]; then
  #  notify=1
  #fi

rm -rf archs/${site}/${outfile}
rm $sfile
echo "Month archived.">> ${logfile}
