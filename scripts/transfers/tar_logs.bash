#!/bin/bash
# Script to tar all logs olders than -d month for site -s site.
data_path='/sps/grand/data'

while getopts ":d:s:" option ${args}; do
   case $option in
      d)
         monthbefore=${OPTARG};;
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

if [ -z "$site" ] || [ -z "$monthbefore" ];then
	printf "Missing option -s or -d\n"
	exit 1
fi

monthstart="$(date -d "$(date +%y-%m-1) - ${monthbefore} month")"
monthend=$(date -d "$(date +%y-%m-1) - $((${monthbefore}-1)) month")
datetag="$(date -d "$(date +%y-%m-1) - ${monthbefore} month" +%Y-%m)"
find /sps/grand/data/${site}/logs/ -type f -newermt "${monthstart}"  -and -not -newermt "${monthend}" -and -not -name '*.tgz' -and -not -name '*.tar' -and -not -name '*.gz'  |xargs tar --remove-files -uvf /sps/grand/data/${site}/logs/logs_${datetag}.tar
