#!/bin/bash
d=1 #um
t2=0 #um
t1=9 #um
eV_min=2 #eV
eV_max=2.8 #eV
eV_step=0.001 #eV
k_max=2 #kmax<eV_min/1.2398419
k_step=0.001
NoX=0

for t1 in 7 #um
do
	./core.sh "$1" "${d}" "${t2}" "${h}" "${FF}" "${t1}" "${pos}" "${eV_min}" "${eV_max}" "${eV_step}" "${k_max}" "${k_step}" "${NoX}"
done