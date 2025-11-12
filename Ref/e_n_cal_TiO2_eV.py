#calculate the e_n of TiO2 experiment
import numpy as np
import matplotlib.pyplot as plt

def Cal(eV): #function to calculate the e_n of TiO2
	import numpy as np
	import sys
	wl_in = 1242/eV
	file_path = "/home/nguyet/Nguyet/Hoang/TiO2_Exp.txt"
	wl = np.loadtxt(file_path)[:,0]     #nm
	n = np.loadtxt(file_path)[:,1]
	k = np.loadtxt(file_path)[:,2]
	
	for j in range(len(wl)):
		if wl_in > wl[j] and wl_in < wl[j+1]:
			n_int = (n[j]*(wl[j+1] - wl_in) + n[j+1]*(wl_in - wl[j]))/(wl[j+1] - wl[j])
			k_int = (k[j]*(wl[j+1] - wl_in) + k[j+1]*(wl_in - wl[j]))/(wl[j+1] - wl[j])
			
		elif wl_in == wl[j]:
			n_int = n[j]
			k_int = k[j]
		else: j+=1
	e_int = (n_int + 1j*k_int)**2
	e_r = e_int.real
	e_i = e_int.imag
	
	return (e_r, e_i,n_int,k_int)



#eV = np.arange(0.8,5.8,0.1)
#n = np.empty(len(eV))

#for i in range(len(eV)):
	#
	#n[i]= Cal(eV[i])
#file_path = "/usr/simdata/Hoang/TiO2_Exp.txt"
#wl1 = np.loadtxt(file_path)[:,0]     #nm
#n1 = np.loadtxt(file_path)[:,1]
#k1 = np.loadtxt(file_path)[:,2]

#eV1 = 1242/wl1
#fig = plt.figure()
#plt.plot(eV,n,color = 'red', markersize = 0.2, linewidth = 1, label = 'n')
#plt.plot(eV1,n1, 'o', markersize = 0.2, linewidth = 1, label = 'n')

#plt.show()


			
	
