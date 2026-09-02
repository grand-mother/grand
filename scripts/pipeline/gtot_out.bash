#!/bin/bash

while getopts ":g:l:" option; do
  case $option in
    g)
      gtot_options=${OPTARG};;
    l)
      logfile=${OPTARG};;
    :)
      printf "option -${OPTARG} need an argument\n"
      exit 1;;
    *) # Invalid option
      printf "Error: Invalid option -${OPTARG}\n"
      exit 1;;
  esac
done

shift $(($OPTIND - 1))

file=$1
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
source ${SCRIPT_DIR}/pipeline_setup.bash
if [ -z "${logfile+x}" ] || [ -z "$logfile" ]; then
    logfile="/dev/null"
fi

echo "logfile=${logfile}" |tee -a ${logfile}
conv_status=0

source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh
conda activate ${conda_lib}
#set -o pipefail
if [ -f $file ]; then
    echo "converting ${file} to GrandRoot" >> ${logfile}
    filename=$(basename $file)
    site=$(echo "${filename%%_*}" | tr '[:upper:]' '[:lower:]')
    root_dest="${data_dir}/${site}/GrandRoot"
    tmp=${filename#*_}
    dateobs=${tmp:0:8}
    dest="${root_dest}/${dateobs:0:4}/${dateobs:4:2}"
    #gtot_options=$default_gtot_opt
    gtot_options=$(echo "$default_gtot_opt" | tr -d '"')
    if [ ! -d $dest ];then
      mkdir -p $dest >> ${logfile}  2>&1
    fi


    #Deternine if output if old sytle or directory type
    if [[ $gtot_options == *"-os"* ]]; then
      out_is_dir=false
      out_opt="-o ${dest}/${filename%.*}.root"
    else
      out_is_dir=true
      out_opt="-od ${dest}"
    fi

    #Determine if file is TR (so no conversion) or CD and gp80 so -gc option is required
    tr=$($(echo basename ${file}) |awk -F_ '{print $5}')
    case $tr in
      TR)
        cp ${file} ${dest}/${filename%.*}.root >> ${logfile}  2>&1
        conv_status=0
        ;;
      CD)
        if [ "${site}" == "gp80" ]; then
          #gtot_extra_option="-gc -os -rn -ow"
          gtot_extra_option=${gtot_options/-g1/-gc}
        else
          gtot_extra_option=${gtot_options}
        fi
        #${gtot_path}  ${gtot_extra_option} -i ${file} -o ${dest}/${filename%.*}.root
        #${gtot_path}  ${gtot_extra_option} -i ${file} ${out_opt}
        #conv_status=$?
        # We need to get the name of the directory created in case of new structure. So we extract it from the output and use tee to send the whole output both to log and stdout
        # in case of old structure, outdest is empty and in case of new structure it contains the directory path
        echo "RUN ${gtot_path}  ${gtot_extra_option} -i ${file} ${out_opt} 2>&1  |tee -a ${logfile} | grep \"Creating directory\" |  awk '{print $NF}'| awk 'NF'" >> ${logfile}
        res=$(${gtot_path}  ${gtot_extra_option} ${out_opt} -i ${file} 2>&1)
        conv_status=$?
        outdest=$(echo "${res}"  |tee -a ${logfile} |grep "Storing output in" |  awk '{print $NF}'| awk 'NF'| xargs dirname )

        #outdest=$(echo "${res}"  |tee -a ${logfile} |grep "Creating directory" |  awk '{print $NF}'| awk 'NF')
        #if [[ "${outdest}" == ""]]; then
        #  outdest=$(echo "${res}"  |tee -a ${logfile} |grep "Storing output in" |  awk '{print $NF}'| awk 'NF'| xargs dirname )
        #fi
        #outdest=$(${gtot_path}  ${gtot_extra_option} ${out_opt} -i ${file} 2>&1  |tee -a ${logfile} |grep "Creating directory" |  awk '{print $NF}'| awk 'NF')


        # Status of conv is the output of the gtot command (so the first pipe)
        #conv_status=${PIPESTATUS[0]}
        ;;
      *)
        gtot_extra_option=${gtot_options}
        #${gtot_path} ${gtot_options} -i ${file} -o ${dest}/${filename%.*}.root
        #${gtot_path} ${gtot_options} -i ${file}  ${out_opt}
        #conv_status=$?
        # We need to get the name of the directory created in case of new structure. So we extract it from the output and use tee to send the whole output both to log and stdout
        echo "RUN ${gtot_path}  ${gtot_extra_option} -i ${file} ${out_opt} 2>&1 |tee -a ${logfile} |grep \"Creating directory\" |  awk '{print \$NF}'| awk 'NF'" >> ${logfile}

        #outdest=$(${gtot_path}  ${gtot_extra_option} ${out_opt} -i ${file} 2>&1 |tee -a ${logfile}  |grep "Creating directory" |  awk '{print $NF}'| awk 'NF')
        res=$(${gtot_path}  ${gtot_extra_option} ${out_opt} -i ${file} 2>&1)
        conv_status=$?
        outdest=$(echo "${res}"  |tee -a ${logfile} |grep "Storing output in" |  awk '{print $NF}'| awk 'NF'| xargs dirname )
        #outdest=$(echo "${res}"  |tee -a ${logfile} |grep "Creating directory" |  awk '{print $NF}'| awk 'NF')
        #if [[ "${outdest}" == ""]]; then
        #  outdest=$(echo "${res}"  |tee -a ${logfile} |grep "Storing output in" |  awk '{print $NF}'| awk 'NF'| xargs dirname )
        #fi
        # Status of conv is the output of the gtot command (so the first pipe)
        #conv_status=${PIPESTATUS[0]}
        ;;
    esac
fi
if [ "$out_is_dir" != "true" ] ; then
   outdest=${dest}/${filename%.*}.root
fi

echo "outdest=${outdest}" |tee -a ${logfile}
echo "convstatus=${conv_status}" |tee -a ${logfile}
exit $conv_status


