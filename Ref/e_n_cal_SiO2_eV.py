def Cal(eV):	#Input Wavelength in nm
	import numpy as np
	import sys
	#from scipy.special import wofz as w
	
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
	
	
	
