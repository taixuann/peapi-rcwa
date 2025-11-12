import math
import sys
import csv
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


def Cal(eV):
	#eV = 1239.8419/wl		#equivalent of wl in eV
	p = np.pi
	
	e_inf = 1	#
	f = 1.12
	E0 = 12		#eV 
	h = 4.135667e-15	# Planck constan - eV.s
 
	
	e = e_inf + (f*E0**2)/(E0**2-eV**2)
	e_r = e.real
	e_i = e.imag
	n = (e**.5).real
	k = (e**.5).imag
	
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

# Plot n (refractive index) vs Energy
plt.figure(dpi=200)
plt.plot(eV_range, n_values, 'b-', label='Refractive index (n)')
plt.plot(eV_range, k_values, 'r--', label='Extinction coefficient (k)')
plt.xlabel("Photon Energy (eV)", fontweight='bold')
plt.ylabel("Optical Constants", fontweight='bold')
plt.title("SiO₂ Refractive Index and Extinction Coefficient", fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(dpi=200)
plt.plot(eV_range, e_r_values, 'g-', label='ε_r (real)')
plt.plot(eV_range, e_i_values, 'm--', label='ε_i (imag)')
plt.xlabel("Photon Energy (eV)", fontweight='bold')
plt.ylabel("Dielectric Constant", fontweight='bold')
plt.title("SiO₂ Dielectric Function", fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()