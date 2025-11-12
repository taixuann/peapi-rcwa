def Cal(eV):	#Input Wavelength in nm
	import numpy as np
	import sys
	#from scipy.special import wofz as w
	
	#eV = 1239.8419/wl		#equivalent of wl in eV
	p = np.pi
	

	f1 = 0 			#0.1439		
	E1 = 2.394		#eV
	y1 = 0.041		#eV/(h/2pi)
	f2 = 0.353
	E2 = 3.25		#eV
	y2 = 0.45		#eV/(h/2pi)
	f3 = 1.48
	E3 = 5.1		#eV
	y3 = 3.9		#eV/(h/2pi)
	
	e_inf = 1.86
	
	e1 = f1*E1**2/(E1**2-eV**2+1j*y1*eV)
	
	e2 = f2*E2**2/(E2**2-eV**2+1j*y2*eV)
	
	e3 = f3*E3**2/(E3**2-eV**2+1j*y3*eV)
	
	e = e_inf+e1+e2+e3
	e_r = e.real
	e_i = np.abs(e.imag)
	n = (e**.5).real
	k = (e**.5).imag
	
	return (e_r, e_i, n, k)
	
	
	
	
