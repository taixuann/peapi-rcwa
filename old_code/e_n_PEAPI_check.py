import math
import sys
import csv

def Cal(eV):
    p = math.pi
   
    f1 = 0.128 #1: main exciton resonance (lowest-energy excitonic transition - 1s)
    E1 = 2.394 #Correct with supplementary information
    y1 = 0.041 #Correct with supplementary information
    f2 = 0
    E2 = 3.25 #2: higher-energy resonance (Higher excitonic states - 2s,3s,... or interband)
    y2 = 0.45
    f3 = 0
    E3 = 5.1 #another resonance at ~5 eV (capture the deep UV absorption features of the material)
    y3 = 3.9

    e_inf = 3.76 #Correct with supplementary information

    e1 = f1*E1**2/(E1**2 - eV**2 + 1j*y1*eV) #Dielectric function of lowest excitonic transition
    e2 = f2*E2**2/(E2**2 - eV**2 + 1j*y2*eV) #Dielectric function of higher-energy transition
    e3 = f3*E3**2/(E3**2 - eV**2 + 1j*y3*eV) #Dielectrci function of another resonance

    e = e_inf + e1 + e2 + e3
    e_r = e.real
    e_i = abs(e.imag)
    n = (e**0.5).real
    k = abs((e**0.5).imag)
    #k = (e**0.5).imag

    return (e_r, e_i, n, k)

## Define linspace BEFORE using it
def linspace(start, stop, num):
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]

##Check PEAPI
eV_range = linspace(1.0, 3.7, 500) #This code will generate the list of number from 1 to 3.7 in 500 steps

n_values = [] #This code will store the result
k_values = [] #This code will store the result
e_r_values = []
e_i_values = []

##Assign the value of eV corresponding with n and k
for eV in eV_range:
    e_r, e_i, n, k = Cal(eV) #_ means ignore the value of e_r and e_i
    n_values.append(n) #Add the compute n into n_values lists
    k_values.append(k) #Add the compute n into k_values lists
    e_r_values.append(e_r)
    e_i_values.append(e_i)

##Sketching the eV with n and k
# Export data to CSV
with open("PEAPI_dielectric_values.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Energy (eV)", "n", "k", "e_r","e_i"])
    for eV, n, k, e_r, e_i in zip(eV_range, n_values, k_values,e_r_values, e_i_values):
        writer.writerow([eV, n, k,e_r,e_i])