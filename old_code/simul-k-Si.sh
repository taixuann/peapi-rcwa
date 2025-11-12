#!/bin/bash
d=1 #um
t2=0.3 #um (Thickness SiO2)
h=0 #um
FF=1
t1=9 #um
pos=0
eV_min=2.2 #eV (not going down 1.24eV to avoid failling arcsin angle)
eV_max=2.5 #eV
eV_step=0.001 #eV
k_max=1.6
k_step=0.001
NoX=0

for t1 in 7 #um
do
	./core-Si.sh "$1" "${d}" "${t2}" "${h}" "${FF}" "${t1}" "${pos}" "${eV_min}" "${eV_max}" "${eV_step}" "${k_max}" "${k_step}" "${NoX}"
done