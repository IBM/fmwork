#/bin/sh
###########################################################################
### This script parses the runs and returns output in the format below: ###
###### model mode iis oos bbs tp warmup(s) setup ttft(ms) itl(ms)>#########
###########################################################################
########### How to run: ./parse.sh <result-dir> | xargs -n 9 ##############
###########################################################################

mode=$1
for i in `find $1 -name "exp.log"`; do
	# <model> <input size> <output size> <batch> <tp> <warmup> <setup> <ttft> <itl>
	error=`grep "FATAL\|ERROR" $i`
	error=`grep "SIGABRT" $i`
	if [[ "${error}" != "" ]]
	then
		continue
	fi
	# Prints model
	echo $i | awk '{split($0,a,"/"); print a[4]}'

	# Prints iis oos bbs tp
	grep "FMWORK RES" $i | awk '  {print $4, $5, $6, $7}'

	# Warmup time can be located at $13 or $15 in result string
	output="$(grep "Total warmup time" $i | awk '{print $13}' | awk '  {split($0,a,"s"); print a[1]}'| grep -Eo '[0-9].*' | tail -1)"
	if [ ! -z "$output" ];then
		echo $output
	else
		output="$(grep "Total warmup time" $i | awk '{print $15}' | awk '  {split($0,a,"s"); print a[1]}'| grep -Eo '[0-9].*' | tail -1)"
	        if [ ! -z "$output" ];then
                	echo $output
	        else
			echo "empty"
                fi
	fi

	# Prints setup time
	grep "FMWORK SETUP" $i | awk '  {print $3}'

	# Prints ttft
	grep -A1 "etim, med" $i | tail -1 | awk '{print $3}' | awk '  {split($0,a,"ms"); print a[1]}'

	# Prints itl which is avg of (oos-1) tokens
	grep " t_token" $i |  sed '1d' | awk '{print $3}' | sed 's/ms//' | awk '{ tot+=$1;cnt++ } END { print tot/cnt }'
done
