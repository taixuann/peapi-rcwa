#!/usr/bin/python3
import numpy as np
import numpy.matlib
import math
import argparse
import sys, getopt
import S4 as S4
from progress.bar import IncrementalBar
import e_n_cal_SiO2_eV
import e_n_cal_TiO2_eV
import e_n_cal_PEAPI_eV
#import e_n_cal_PEAPI_NoX_eV
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('-d', type=float, required=True, help="Period of the photonic structure (d)")
parser.add_argument('-t', type=float, required=True, help="Thickness of the underlayer (t2)")
parser.add_argument('--height', type=float, required=True, help="Thickness of the periodic layer (h)")
parser.add_argument('-f', type=float, required=True, help="Filling factor (FF)")
parser.add_argument('-a', type=float, required=True, help="Thickness of the active layer (t1)")
parser.add_argument('--emin', type=float, required=True, help="Minimum radiation energy")
parser.add_argument('--emax', type=float, required=True, help="Maximum radiation energy")
parser.add_argument('--estep', type=float, required=True, help="Radiation energy resolution")
parser.add_argument('--kmax', type=float, required=True, help="Maximum inplane radiation wavenumber")
parser.add_argument('--kstep', type=float, required=True, help="Radiation inplane radiation resolution")
parser.add_argument('--pos', type=float, required=True, help="Position of active layer in the slab (from 0 to 1)")
parser.add_argument('--NoX', type=int, required=True, help="Calculation without exciton (--NoX 1)")

args = parser.parse_args()

#print(args)

d = args.d
t2 = args.t
h = args.height
FF = args.f
t1 = args.a
eV_min = args.emin
eV_max = args.emax
eV_step = args.estep
k_max = args.kmax
k_min = 0 #by default
k_step = args.kstep
pos = args.pos
calculNoX = args.NoX


# Function for S4 simulation
def S4Module(eV, e_r_PEAPI, e_i_PEAPI, e_r_SiO2, e_i_SiO2, e_r_TiO2, e_i_TiO2, n_SiO2, n_TiO2, n_PEAPI, d_PEAPI, D, h, k, N):
	
	S = S4.New(Lattice=((D,0),(0,0)),NumBasis=N)  #period of the pattern D in um

	# Definition for used material
	S.SetMaterial(Name = 'Air', Epsilon = 1 + 0*1j)
	#S.SetMaterial(Name = 'Si', Epsilon = e_r_Si + e_i_Si*1j) 		#real and imagine part of permittivity for Si
	S.SetMaterial(Name = 'SiO2', Epsilon = e_r_SiO2 + e_i_SiO2*1j) 		#real and imagine part of permittivity for SiO2
	S.SetMaterial(Name = 'TiO2', Epsilon = e_r_TiO2 + e_i_TiO2*1j) 	#real and imagine part of permittivity for TiO2
	S.SetMaterial(Name = 'Perovskite', Epsilon= e_r_PEAPI + e_i_PEAPI*1j) 	#real and imagine part of permittivity for perovskite
		

	# Create multilayers for simulation process, thinkness: um
	S.AddLayer('Above', 0, 'Air') #air above has 0 um by default
	
	S.AddLayer('AH1', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL2', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH2', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL3', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH3', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL4', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH4', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL5', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH5', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL6', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH6', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL7', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH7', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL8', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH8', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL9', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH9', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL10', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH10', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL11', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH11', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL12', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('AH12', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('AL1', h/(4*n_TiO2), 'TiO2')

	
	#S.AddLayer('ASpacer',(t2 ),'Air')
	
	S.AddLayer('PEAPI', d_PEAPI, 'Perovskite')
	
	#S.AddLayer('BSpacer',(t2 - n_PEAPI*d_PEAPI)*0.875,'Air')
	
	S.AddLayer('BL12', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH1', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL1', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH2', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL2', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH3', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL3', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH4', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL4', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH5', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL5', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH6', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL6', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH7', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL7', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH8', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL8', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH9', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL9', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH10', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL10', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH11', h/(4*n_SiO2), 'SiO2')
	S.AddLayer('BL11', h/(4*n_TiO2), 'TiO2')
	S.AddLayer('BH12', h/(4*n_SiO2), 'SiO2')
	#S.AddLayer('BL12', h/(4*n_TiO2), 'TiO2')
	
	
	S.AddLayer('Below', 0,'Air') #the substrate layer has 0 um by default

	# Define incident field/light
	angle = abs(np.arcsin(k*1.2398419/eV)*180/math.pi) 		# incident angle
	Ampl = 1  		# Intensity = 100
	S.SetExcitationPlanewave (IncidenceAngles=(angle,0), sAmplitude = Ampl+0*1j, pAmplitude = 1, Order = 0)
	S.SetFrequency(eV/1.2398419) 		# 1/Wl = 1 -> Wl = 1um; 1.1-> W = 909nm (=1/1.1), basic length: 1um

	# Get Transmission, Reflection and Absorption Coefficient
	(Tc_0, Rc_0) = S.GetPowerFlux('Above', -1)	#Before going to L1

	#Rc_0 = S.GetLayerVolumeIntegral('PEAPI', Quantity = 'U')
	#Tc_0 = 1

	return (Tc_0, Rc_0)


N = 1

(e_r_SiO2_cav, e_i_SiO2_cav, n_SiO2_cav, k_SiO2_cav) = e_n_cal_SiO2_eV.Cal(1.2398419/h)
		
(e_r_TiO2_cav, e_i_TiO2_cav, n_TiO2_cav, k_TiO2_cav) = e_n_cal_TiO2_eV.Cal(1.2398419/h)

(e_r_PEAPI_cav, e_i_PEAPI_cav, n_PEAPI_cav, k_PEAPI_cav) = e_n_cal_PEAPI_eV.Cal(1.2398419/d)

#(e_r_PEAPI_NoX_cav, e_i_PEAPI_NoX_cav, n_PEAPI_NoX_cav, k_PEAPI_NoX_cav) = e_n_cal_PEAPI_NoX_eV.Cal(1.2398419/d)

eV_range = np.arange(eV_min,eV_max+np.float(eV_step/2),eV_step) #eV
kx = np.arange(k_min,k_max+np.float(k_step/2),k_step) # kx
Rc_Above = np.zeros((eV_range.size ,kx.size))
Tc_Above = np.zeros((eV_range.size ,kx.size))

#Processing Section
eV = eV_min # in nm
j = 0
if ((t1 != 0) and (calculNoX == 1)):
	bar = IncrementalBar('Calculating',max=eV_range.size*2, suffix='%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
else:
	bar = IncrementalBar('Calculating',max=eV_range.size, suffix='%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
while  eV < (eV_max + eV_step/2):
	
	
	(e_r_SiO2, e_i_SiO2, n_SiO2, k_SiO2) = e_n_cal_SiO2_eV.Cal(eV)
	
	(e_r_TiO2, e_i_TiO2, n_TiO2, k_TiO2) = e_n_cal_TiO2_eV.Cal(eV)
	
	(e_r_PEAPI, e_i_PEAPI, n_PEAPI, k_PEAPI) = e_n_cal_PEAPI_eV.Cal(eV)

	k = 0
	wv = k_min
	while wv < (k_max + k_step/2):
		(Tc_0, Rc_0) = S4Module(eV, e_r_PEAPI, e_i_PEAPI, e_r_SiO2, e_i_SiO2, e_r_TiO2, e_i_TiO2,  n_SiO2_cav, n_TiO2_cav, n_PEAPI_cav, t1, d, h, wv, N)
		Tc_Above[j,k]  = np.abs(Tc_0.real)
		Rc_Above[j,k] = np.abs(Rc_0.real)
		wv += k_step
		k +=1	
	eV += eV_step
	bar.next()
	j += 1 

Ratio_Above = Rc_Above/Tc_Above

np.savetxt('./data/'+str(d)+'_'+str(t2)+'_'+str(h)+'_'+str(FF)+'_'+str(t1)+'_'+str(pos)+'_'+str(eV_min)+'_'+str(eV_max)+'_'+str(eV_step)+'_'+str(k_max)+'_'+str(k_step)+'_'+'k-E-refl.csv',Ratio_Above,delimiter=',')


bar.finish()


