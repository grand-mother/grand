#!/bin/bash
# Script to archive Grand raw files (age = 2 month ago) and push the archive into irods
# Fleg, Oct 2024

# Define useful variables
datadir="/sps/grand/data"
archive_root_dir="/sps/grand/prod_grand/archiving"
archive_root_name="doi+10.25520+in2p3.archive.grand"
irods_path='/grand/home/trirods/data/archives/'
# The former script to create archive needed java 8 (some used libs are not available in java versions > 8) but is now corrected
#javabin='/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.422.b05-2.el9.x86_64/jre/bin/java'
javabin='java'

# Get the year and month for 2 month ago
read year month << DATE_COMMAND
 $(date --date="TODAY -2 month" "+%Y %m")
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
	fileslist=${archive_root_dir}/${site}/list_files_${site}
  # ensure that directory exists
	mkdir -p ${archive_root_dir}/${site} > /dev/null 2>&1

  # Check that file containing the list of files to archive (${fileslist}) does not exists
  # if it exists it should mean that another process is still running, so skip
  if [ -f ${fileslist} ]; then
    break
  fi
  # touch file immediately to "lock" the process (find command should last)
  touch ${fileslist}
  find ${datadir}/${site}/raw/${dir}/ -name "*.bin" >> ${fileslist}
  list=$(cat ${fileslist})

  # If no files to archive then skip
  if [ "${list}" == "" ]; then
    echo "No files"
    rm ${fileslist}
    break
  else
    echo "Archiving $month $year for $site"
    echo "Archiving $month $year for $site" >> ${logfile}
  fi

  # Create the archive
  $javabin -jar createAIP.jar --configfile=config.properties.${site} --listobjects=${fileslist} -i ${outfile} >> ${logfile} 2>&1
  createaip_status=$?

  if [ "$createaip_status" -eq 0 ]; then
    echo "Archive ready to tar" >> ${logfile}
    tar -cf ${outdir}.tar ${outdir}
    tar_status=$?
    if [ "$tar_status" -eq 0 ]; then
      echo "remove temp dir ${outdir}" >> ${logfile}
      rm -rf ${outdir} >> ${logfile} 2>&1
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
        echo "remove ${outdir}.tar" >> ${logfile}
        rm ${outdir}.tar >> ${logfile} 2>&1
        echo "Raw data of ${year}/${month} from ${site} archived " >> ${logfile}

        echo "compress files" >> ${logfile}
        while IFS= read -r line
        do
          echo "gzip ${line}"
          gzip $line
        done < "${fileslist}"

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

exit 0






if [ "$iput_status" -eq 0 ]; then
	# tar gz the month if older than 6 months

	if [ "$diffdate" -gt 180 ];then
	        echo compress
		gzip $sfile
		gzip_status=$?
		mv ${sfile}.gz ${datadir}/${site}/raw/${dir}/${site}_bin_${date}.tgz
		#tar -czvf ${datadir}/${site}/raw/${dir}/${site}_bin_${date}.tgz -T list_files_${site}
		#tar_status=$?
	        if [ "$gzip_status" -eq 0 ];then
	                echo erase
			xargs rm <list_files_${site}
	        fi
	fi
fi
rm -rf archs/${site}/${outfile}
#rm $sfile
echo "Month archived.">> ${logfile}

