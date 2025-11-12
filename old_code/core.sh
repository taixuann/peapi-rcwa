#!/bin/bash

d=$2
t2=$3
h=$4
FF=$5
t1=$6
pos=$7
eV_min=$8
eV_max=$9
eV_step=${10}
k_max=${11}
k_step=${12}
NoX=${13}
 
if [ "$1" = "calculation" ]	
then
	if [ -d "./data" ]
	then
		echo "calcul-spectrum-k.py -d ${d} -t ${t2} --height ${h} -f ${FF} -a ${t1} --pos ${pos} --emin ${eV_min} --emax ${eV_max} --estep ${eV_step} --kmax ${k_max} --kstep ${k_step} --NoX ${NoX}"
		./calcul-spectrum-k.py "-d" "${d}" "-t" "${t2}" "--height" "${h}" "-f" "${FF}" "-a" "${t1}" "--pos" "${pos}" "--emin" "${eV_min}" "--emax" "${eV_max}" "--estep" "${eV_step}" "--kmax" "${k_max}" "--kstep" "${k_step}" "--NoX" "${NoX}"

	else
		mkdir "./data"
		echo "calcul-spectrum-k.py -d ${d} -t ${t2} --height ${h} -f ${FF} -a ${t1} --pos ${pos} --emin ${eV_min} --emax ${eV_max} --estep ${eV_step} --kmax ${k_max} --kstep ${k_step} --NoX ${NoX}"
		./calcul-spectrum-k.py "-d" "${d}" "-t" "${t2}" "--height" "${h}" "-f" "${FF}" "-a" "${t1}" "--pos" "${pos}" "--emin" "${eV_min}" "--emax" "${eV_max}" "--estep" "${eV_step}" "--kmax" "${k_max}" "--kstep" "${k_step}" "--NoX" "${NoX}"

	fi	
	if [ -d "./graphics" ]
	then
		echo "Create graphic files ${d}_${t2}_${h}_${FF}_${t1}_${pos}_${eV_max}_${eV_step}_${k_max}_${k_step}_${NoX}"
		./plot-spectrum-k.py "-d" "${d}" "-t" "${t2}" "--height" "${h}" "-f" "${FF}" "-a" "${t1}" "--pos" "${pos}" "--emin" "${eV_min}" "--emax" "${eV_max}" "--estep" "${eV_step}" "--kmax" "${k_max}" "--kstep" "${k_step}" "--NoX" "${NoX}"
	else
		mkdir "./graphics"
		echo "Create graphic files ${d}_${t2}_${h}_${FF}_${t1}_${pos}_${eV_max}_${eV_step}_${k_max}_${k_step}_${NoX}"
		./plot-spectrum-k.py "-d" "${d}" "-t" "${t2}" "--height" "${h}" "-f" "${FF}" "-a" "${t1}" "--pos" "${pos}" "--emin" "${eV_min}" "--emax" "${eV_max}" "--estep" "${eV_step}" "--kmax" "${k_max}" "--kstep" "${k_step}" "--NoX" "${NoX}"
	fi		
elif [ "$1" = "graphics" ]	
then	
	if [ -d "./graphics" ]
	then
		echo "Create graphic files ${d}_${t2}_${h}_${FF}_${t1}_${pos}_${eV_max}_${eV_step}_${k_max}_${k_step}_${NoX}"
		./plot-spectrum-k.py "-d" "${d}" "-t" "${t2}" "--height" "${h}" "-f" "${FF}" "-a" "${t1}" "--pos" "${pos}" "--emin" "${eV_min}" "--emax" "${eV_max}" "--estep" "${eV_step}" "--kmax" "${k_max}" "--kstep" "${k_step}" "--NoX" "${NoX}"
	else
		mkdir "./graphics"
		echo "Create graphic files ${d}_${t2}_${h}_${FF}_${t1}_${pos}_${eV_max}_${eV_step}_${k_max}_${k_step}_${NoX}"
		./plot-spectrum-k.py "-d" "${d}" "-t" "${t2}" "--height" "${h}" "-f" "${FF}" "-a" "${t1}" "--pos" "${pos}" "--emin" "${eV_min}" "--emax" "${eV_max}" "--estep" "${eV_step}" "--kmax" "${k_max}" "--kstep" "${k_step}" "--NoX" "${NoX}"
	fi	
else
	echo "Do nothing!"
fi
