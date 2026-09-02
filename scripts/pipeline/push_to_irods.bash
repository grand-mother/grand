#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
source ${SCRIPT_DIR}/pipeline_setup.bash

while getopts ":s:d:" option; do
  case $option in
    s)
      source=${OPTARG};;
    d)
      outdest=${OPTARG};;
    :)
      printf "option -${OPTARG} need an argument\n"
      exit 1;;
    *) # Invalid option
      printf "Error: Invalid option -${OPTARG}\n"
      exit 1;;
  esac
done

shift $(($OPTIND - 1))

echo $source
echo $outdest
filename=$(basename $source)

if [ -f "$source" ]; then
    echo "$source est un fichier"
    irods_option='-f '
elif [ -d "$source" ]; then
    echo "$source est un repertoire"
    irods_option='-rf '
else
    echo "$source n existe pas ou n est ni fichier ni repertoire"
    exit 1
fi

#      # Put GrandRoot file into irods
ifile=${outdest}/${filename}
ipath=${ifile%/*}
echo "imkdir -p $ipath"
imkdir -p $ipath
echo "iput $irods_option $source $ifile"
iput $irods_option $source $ifile
iput_status=$?
exit $iput_status

#if [ "$iput_status" -ne 0 ]; then
#   echo "Error ${iput_status} in iput"
#fi
#exit ${iput_status}