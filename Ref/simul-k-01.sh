#!/bin/bash
d=0.520
t2=0.256
h=0.512
FF=0
#t1=0.01
pos=0
eV_min=1.6
eV_max=2.8
eV_step=0.002
k_max=3
k_step=0.002
NoX=0

for t1 in  0.132
do
	./core-simul-k.sh "$1" "${d}" "${t2}" "${h}" "${FF}" "${t1}" "${pos}" "${eV_min}" "${eV_max}" "${eV_step}" "${k_max}" "${k_step}" "${NoX}"
done
