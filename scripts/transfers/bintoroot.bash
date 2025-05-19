#!/bin/bash -l
#gtot_path='/sps/grand/prod_grand/DB_TESTS/gtot/cmake-build-release/gtot'
gtot_path='/pbs/home/p/prod_grand/softs/gtot/cmake-build-release/gtot'
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
script_dir=$(dirname $script_path)
grand_dir=${script_dir%/scripts/transfers}
register_convertion="${grand_dir}/scripts/transfers/register_convert.py"
register_root="${grand_dir}/granddb/register_file_in_db.py"
register_dir="${grand_dir}/granddb/register_dir_in_db.py"
config_file="${grand_dir}/granddb/config.ini"
sps_path='/sps/grand/'
irods_path='/grand/home/trirods/'
submit_base_name=''


# Get tag and database file to use
while getopts ":d:g:n:" option; do
  case $option in
    d)
      root_dest=${OPTARG};;
    g)
      gtot_options=${OPTARG};;
    n)
      submit_base_name=${OPTARG};;
    :)
      printf "option -${OPTARG} need an argument\n"
      exit 1;;
    *) # Invalid option
      printf "Error: Invalid option -${OPTARG}\n"
      exit 1;;
  esac
done

shift $(($OPTIND - 1))

cd ${grand_dir}/
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh
conda activate /sps/grand/software/conda/grandlib_2409
source env/setup.sh
cd ${grand_dir}/scripts/transfers
export PATH=/sps/grand/software/conda/grandlib_2409/bin/:$PATH


notify=0
for file in "$@"
do
  if [ -f $file ]; then
    echo "converting ${file} to GrandRoot"
    filename=$(basename $file)
    tmp=${filename#*_}
    dateobs=${tmp:0:8}
    dest="${root_dest}/${dateobs:0:4}/${dateobs:4:2}"
    if [ ! -d $dest ];then
      mkdir -p $dest >/dev/null 2>&1
    fi
    dirlogs=${root_dest}/../logs
    logfile=${dirlogs}/${submit_base_name}-bin2root-${filename%.*}.log
    if [ ! -d $dirlogs  ];then
      mkdir -p $dirlogs >/dev/null 2>&1
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
        cp ${file} ${dest}/${filename%.*}.root
        conv_status=0
        ;;
      CD)
        site=${filename%_*}
        site=$($(echo basename ${file}) |awk -F_ '{print $1}')
        if [ "${site,,}" == "gp80" ]; then
          #gtot_extra_option="-gc -os -rn -ow"
          gtot_extra_option=${gtot_options/-g1/-gc}
        else
          gtot_extra_option=${gtot_options}
        fi
        #${gtot_path}  ${gtot_extra_option} -i ${file} -o ${dest}/${filename%.*}.root >> ${logfile}
        #${gtot_path}  ${gtot_extra_option} -i ${file} ${out_opt} >> ${logfile}
        #conv_status=$?
        # We need to get the name of the directory created in case of new structure. So we extract it from the output and use tee to send the whole output both to log and stdout
        # in case of old structure, outdest is empty and in case of new structure it contains the directory path
        echo "RUN ${gtot_path}  ${gtot_extra_option} -i ${file} ${out_opt} |tee -a ${logfile} |grep \"Creating directory\" |  awk '{print $NF}'"
        outdest=$(${gtot_path}  ${gtot_extra_option} ${out_opt} -i ${file} |tee -a ${logfile} |grep "Creating directory" |  awk '{print $NF}')
        #outdest=$(${gtot_path}  ${gtot_extra_option} -i ${file} ${out_opt} |tee -a ${logfile} |grep "Creating directory" |  awk '{print $NF}')
        # Status of conv is the output of the gtot command (so the first pipe)
        conv_status=${PIPESTATUS[0]}
        ;;
      *)
        gtot_extra_option=${gtot_options}
        #${gtot_path} ${gtot_options} -i ${file} -o ${dest}/${filename%.*}.root >> ${logfile}
        #${gtot_path} ${gtot_options} -i ${file}  ${out_opt}>> ${logfile}
        #conv_status=$?
        # We need to get the name of the directory created in case of new structure. So we extract it from the output and use tee to send the whole output both to log and stdout
        echo "RUN ${gtot_path}  ${gtot_extra_option} -i ${file} ${out_opt} |tee -a ${logfile} |grep \"Creating directory\" |  awk '{print $NF}'"
        outdest=$(${gtot_path}  ${gtot_extra_option} ${out_opt} -i ${file}  |tee -a ${logfile} |grep "Creating directory" |  awk '{print $NF}')
        #outdest=(${gtot_path}  ${gtot_extra_option} -i ${file} ${out_opt} |tee -a ${logfile} |grep "Creating directory" |  awk '{print $NF}')
        # Status of conv is the output of the gtot command (so the first pipe)
        conv_status=${PIPESTATUS[0]}
        ;;
    esac



    if [ "$conv_status" -ne 0 ] || ([ -z "${outdest//[[:space:]]/}" ] && [ "$out_is_dir" = "true" ]) ; then
      notify=1
      echo "Error ${conv_status} in conversion."  |& tee -a ${logfile}
      outstatus=$conv_status
    else

      if [ "$out_is_dir" = "true" ] ; then
        irods_option='-rf '
        sfile=$outdest

      else
         irods_option='-f '
         sfile=${dest}/${filename%.*}.root
      fi
      # Put GrandRoot file into irods

      #sfile=${dest}/${filename%.*}.root
      ifile=${sfile/$sps_path/$irods_path}
      ipath=${ifile%/*}
      echo "imkdir -p $ipath" >> ${logfile}
      imkdir -p $ipath >> ${logfile} 2>&1
      echo "iput $irods_option $sfile $ipath" >> ${logfile}
      iput $irods_option $sfile $ipath >> ${logfile} 2>&1
      iput_status=$?


      if [ "$iput_status" -ne 0 ]; then
        notify=1
        echo "Error ${iput_status} in iput"  |& tee -a ${logfile}
        outstatus=$iput_status
      fi
      # Register conversion result into the database
      echo "Register convertion" >> ${logfile}
      echo "Run ${register_convertion} -i ${filename} -o ${filename%.*}.root -s ${conv_status} -l ${logfile}"  |& tee -a ${logfile}
      python3 ${register_convertion} -i ${filename} -o ${filename%.*}.root -s ${conv_status} -l ${logfile} >> ${logfile} 2>&1

      # Register root file into db
      if [ $tr != "TR" ]; then
          if [ "$out_is_dir" = "true" ] ; then
                  echo "register directory ${sfile} in database" >> ${logfile}
                  echo "Run python3 ${register_dir} -c ${config_file} -r "CCIN2P3" ${sfile} " |& tee -a ${logfile}
                  python3 ${register_dir} -c ${config_file} -r "CCIN2P3" ${sfile} >> ${logfile} 2>&1
          else
                  echo "register file ${sfile} in database" >> ${logfile}
                  echo "Run python3 ${register_root} -c ${config_file} -r "CCIN2P3" ${sfile} " |& tee -a ${logfile}
                  python3 ${register_root} -c ${config_file} -r "CCIN2P3" ${sfile} >> ${logfile} 2>&1
          fi
          register_status=$?
          if [ "$register_status" -ne 0 ]; then
                  notify=1
                  echo "Error ${register_status} in registration" |& tee -a ${logfile}
                  outstatus=$register_status
          fi

      fi
    fi
#    if [ $tr != "TR" ]; then
#      echo "register file ${dest}/${filename%.*}.root in database" >> ${logfile}
#      echo "Run python3 ${register_root} -c ${config_file} -r "CCIN2P3" ${dest}/${filename%.*}.root " |& tee -a ${logfile}
#      python3 ${register_root} -c ${config_file} -r "CCIN2P3" ${dest}/${filename%.*}.root >> ${logfile} 2>&1
#      register_status=$?
#      if [ "$register_status" -ne 0 ]; then
#        notify=1
#        echo "Error ${register_status} in registration" |& tee -a ${logfile}
#        outstatus=$register_status
#      fi
#    fi

  fi
done

if [ "$notify" -ne "0" ]; then
  parent_script=$(cat /proc/$PPID/comm)
  echo "Error in files conversion/registration : ${parent_script} ${0} ${*} " |   mail -s "Grand pipeline error in ${submit_base_name} " fleg@lpnhe.in2p3.fr
  exit $outstatus
fi
