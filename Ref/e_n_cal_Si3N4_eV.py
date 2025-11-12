def Cal(eV):	#Input Wavelength in nm
	import numpy as np
	import sys
	#from scipy.special import wofz as w
	
	#eV = 1239.8419/wl		#equivalent of wl in eV
	p = np.pi
	
	n_inf = 1.794
	Eg = 2.221	
	fj = 0.031		#eV
	Lj = 0.623		#eV
	Ej = 7.34		#eV

	
	B = (fj/Lj)*(Lj**2 - (Ej - Eg)**2)
	C = 2*fj*Lj*(Ej-Eg)
	
	n = n_inf + (B*(eV - Ej) + C)/((eV - Ej)**2 + Lj**2)
	#k = np.empty(len(eV))
	
	#for i in range(len(eV)): 
	if eV > Eg:
		k = fj*(eV - Eg)**2/((eV-Ej)**2 + Lj**2)
	else:
		k = 0
		
	e_r = n**2 - k**2
	e_i = 2*n*k

	
	return (e_r, e_i, n, k)
	
	
	
	
