#!/bin/bash -l

# path to gtot
gtot_path='/pbs/home/p/prod_grand/softs/gtot/cmake-build-release/gtot'
# path to script to register convertion results
register_convertion='/pbs/home/p/prod_grand/softs/grand/scripts/transfers/register_convert.py'
# path to script to register root file into the DB
register_root='/pbs/home/p/prod_grand/softs/grand/granddb/register_file_in_db.py'
config_file='/pbs/home/p/prod_grand/softs/grand/scripts/transfers/config-prod.ini'
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

#export PLATFORM=redhat-9-x86_64
cd /pbs/home/p/prod_grand/softs/grand
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh


#Export some env to make irods works
#export LOADEDMODULES=DataManagement/irods/4.3.1
#export TRIRODS_DATA_DIR=/grand/home/trirods/data
#export BASH_ENV=/usr/share/Modules/init/bash
#export LD_LIBRARY_PATH=/pbs/throng/grand/soft/lib/:/pbs/software/redhat-9-x86_64/irods/4.3.1/lib:/pbs/software/redhat-9-x86_64/irods/irods-externals/4.3.1/lib
#export PATH=/pbs/throng/grand/soft/miniconda3/condabin:/pbs/throng/grand/soft/bin/:/pbs/throng/grand/bin/:/opt/software/rfio-hpss/prod/bin:/usr/share/Modules/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin:/opt/ccin2p3/bin:/pbs/software/redhat-9-x86_64/irods/utils:/pbs/software/redhat-9-x86_64/irods/4.3.1/bin:.
#export _LMFILES_=/pbs/software/modulefiles/redhat-9-x86_64/DataManagement/irods/4.3.1
#export IRODS_PLUGINS_HOME=/pbs/software/redhat-9-x86_64/irods/4.3.1/lib/plugins
#export MODULEPATH=/etc/scl/modulefiles:/pbs/software/modulefiles/redhat-9-x86_64:/etc/modulefiles
conda activate /sps/grand/software/conda/grandlib_2409
source env/setup.sh
cd /pbs/home/p/prod_grand/softs/grand/scripts/transfers
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
          gtot_extra_option="-gc -os -rn"
        else
          gtot_extra_option=${gtot_options}
        fi
        ${gtot_path}  ${gtot_extra_option} -i ${file} -o ${dest}/${filename%.*}.root >> ${logfile}
        conv_status=$?
        ;;
      *)
        ${gtot_path} ${gtot_options} -i ${file} -o ${dest}/${filename%.*}.root >> ${logfile}
        conv_status=$?
        ;;
    esac

    #if [ $tr == "TR" ]; then
    #  cp ${file} ${dest}/${filename%.*}.root
    #  conv_status=0
    #else
    #  # Convert file
    #  ${gtot_path} ${gtot_options} -i ${file} -o ${dest}/${filename%.*}.root >> ${logfile}
    #  conv_status=$?
    #fi

    if [ "$conv_status" -ne 0 ]; then
      notify=1
      echo "Error ${conv_status} in conversion."  |& tee -a ${logfile}
    fi

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
      echo "Error ${iput_status} in iput"  |& tee -a ${logfile}
    fi
    # Register conversion result into the database
    echo "Register convertion" >> ${logfile}
    echo "Run ${register_convertion} -i ${filename} -o ${filename%.*}.root -s ${conv_status} -l ${logfile}"  |& tee -a ${logfile}
    python3 ${register_convertion} -i ${filename} -o ${filename%.*}.root -s ${conv_status} -l ${logfile} >> ${logfile} 2>&1
    # Register root file into db
    if [ $tr != "TR" ]; then
      echo "register file ${dest}/${filename%.*}.root in database" >> ${logfile}
      echo "Run python3 ${register_root} -c ${config_file} -r "CCIN2P3" ${dest}/${filename%.*}.root " |& tee -a ${logfile}
      python3 ${register_root} -c ${config_file} -r "CCIN2P3" ${dest}/${filename%.*}.root >> ${logfile} 2>&1
      register_status=$?
      if [ "$register_status" -ne 0 ]; then
        notify=1
        echo "Error ${register_status} in registration" |& tee -a ${logfile}
      fi
    fi
  fi
done

if [ "$notify" -ne "0" ]; then
  parent_script=$(cat /proc/$PPID/comm)
  echo "Error in files conversion/registration : ${parent_script} ${0} ${@} " |   mail -s "Grand pipeline error in $0 " fleg@lpnhe.in2p3.fr
fi
