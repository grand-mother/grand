#!/bin/bash
# Program to search for raw files not converted into root format and launch convertion and registration of these files.

datadir='/sps/grand/data'
bin2root='/pbs/home/p/prod_grand/softs/grand/scripts/transfers/bintoroot.bash'
refresh_mat_script='/pbs/home/p/prod_grand/softs/grand/scripts/transfers/refresh_mat_views.bash'
mail_user='fleg@lpnhe.in2p3.fr'
mail_type='FAIL,TIME_LIMIT,INVALID_DEPEND'
# Max of concurent jobs at same time
nbjobs=30
tag=$(date +'%Y%m%d%H%M%S')
site=$1

convjobs=""
k=0
declare -A jobsid
dep=""

case $site in
  gp80)
    gtot_option="-g1 -os -rn";;
  gp13)
    gtot_option="-g1 -os -rn";;
  gaa)
    gtot_option="-f2 -os";;
  ?)
    gtot_option="-g1 -os";;
esac
echo $site
rootbasedir=$datadir/$site/GrandRoot
rawbasedir=$datadir/$site/raw
logdir=$datadir/$site/logs
rawlist=$(find ${rawbasedir}  -type f -name "*.bin" ! -path "${rawbasedir}/crap/*")
convjobs=""
k=0
declare -A jobsid
nbjobs=30
dep=""
for rawfile in $rawlist;do
	rootfile=$(sed 's|/raw/|/GrandRoot/|g'<<< $rawfile)
	rootfile=$(sed 's|.bin|.root|g'<<< $rootfile)
	if [ ! -f "$rootfile" ]; then
		# Search for alternate name (old files)
		rootbasename1=$(basename $rootfile)
		rootbasename="${rootbasename1%.*}*"
		rootbasedir=$(dirname $rootfile)
		#echo "search for $rootbasename into $rootbasedir"
		found=$(find  $rootbasedir -name $rootbasename)
		#echo $found
		if [  -z "$found" ]; then
			echo "file $rawfile is not converted"
			echo "#!/bin/bash" > ${logdir}/b2r_${rootbasename1%.*}.bash
			echo "$bin2root -n 'convert_not_converted_${tag}_' -g '$gtot_option' -d $datadir/$site/GrandRoot $rawfile" >> ${logdir}/b2r_${rootbasename1%.*}.bash
			#if [ "$convjobs" = "" ]; then
  			#	dep=""
			#else
			#	dep="--dependency=afterany:${convjobs}"
			#fi
			if [ "$k" -gt "$nbjobs" ]; then
  				# The next jobs are launched only when a previous one ended (so dependency of the id of job j-nbjobs)
  				l=$((k - nbjobs))
  				jregid=${jobsid[${l}]}

				if [ "$dep" = "" ]; then
                                      		dep="--dependency=afterany"
                               	fi
				dep="${dep}:jregid"
			fi

			jobsid[${k}]=$(sbatch ${dep} -t 0-00:25 -n 1 -J b2r_${rootbasename1%.*} -o ${logdir}/b2r_${rootbasename1%.*}.log --mem 2G  --mail-user=fleg@lpnhe.in2p3.fr --mail-type='FAIL,TIME_LIMIT,INVALID_DEPEND' ${logdir}/b2r_${rootbasename1%.*}.bash)
			jobsid[${k}]=$(echo ${jobsid[${k}]} |awk '{print $NF}')
			#jid=$(echo $jid |awk '{print $NF}')
			#convjobs=$jid
			#convjobs=$convjobs":"${jobsid[${k}]}
			((k++))

		fi
	fi
done

# Launch updates if files were processed
if [ $k -gt 0 ]; then
	joblist=${jobsid[@]}
	joblist=${joblist// /:}
	  sbatch --dependency=afterany:${joblist} -t 0-00:15 -n 1 -J refresh_mat_${tag} -o ${logdir}/refresh_mat_${tag}.log --mem 1G  --mail-user=${mail_user} --mail-type=${mail_type} ${refresh_mat_script}

fi
