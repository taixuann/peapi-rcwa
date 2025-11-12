#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import math
import argparse
import sys, getopt
import csv
import cv2 as cv
import matplotlib.font_manager as font_manager
from progress.bar import IncrementalBar
import e_n_cal_PEAPI_eV

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

#print('d='+str(d))
#print('t1='+str(t1))
#print('h='+str(h))
#print('FF='+str(FF))
#print('t2='+str(t2))
#print('eV_min='+str(eV_min))
#print('eV_max='+str(eV_max))
#print('eV_step='+str(eV_step))
#print('k_max='+str(k_max))
#print('k_step='+str(k_step))
#print('pos='+str(pos))
#print('NoX='+str(NoX))

eV_range = np.arange(eV_min,eV_max+np.float(eV_step/2),eV_step) #eV

ev_min = np.arange(eV_min,2.25+np.float(eV_step/2),eV_step)
ev_max = np.arange(2.25,eV_max+np.float(eV_step/2),eV_step)

kx = np.arange(k_min,k_max+np.float(k_step/2),k_step) # kx
kx_All = np.arange(-k_max,k_max+k_step/2,k_step) # kx_all
X, Y = np.meshgrid(kx_All, eV_range)	

X1, Y1 = np.meshgrid(kx_All, ev_min)

X2, Y2 = np.meshgrid(kx_All, ev_max)

bar = IncrementalBar('Calculating',max=3, suffix='%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
with open('./data/'+str(d)+'_'+str(t2)+'_'+str(h)+'_'+str(FF)+'_'+str(t1)+'_'+str(pos)+'_'+str(eV_min)+'_'+str(eV_max)+'_'+str(eV_step)+'_'+str(k_max)+'_'+str(k_step)+'_'+'k-E-refl.csv') as csvfileR:
	R = np.loadtxt(csvfileR, delimiter=',')
bar.next()

font = font_manager.FontProperties(family='DejaVu Sans', style='normal', size = 25)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 20
#plt.rcParams['text.usetex'] = True

fig = plt.figure(dpi=200)
#E_0 = 2.394 #eV
E_0 = 2.385
g = 0.085
E_C = np.empty(len(kx_All))
E_X = np.empty(len(kx_All))
E_U = np.empty(len(kx_All))
E_L = np.empty(len(kx_All))

for i in range (len(kx_All)):
    #E_C[i] = E_0 + 0.152*kx_All[i]**2
    E_C[i] = E_0 + 0.180*kx_All[i]**2
    E_X[i] = 2.394
    E_U[i] = (E_C[i] +E_X[i])/2 + (g**2 + ((E_C[i] - E_X[i])**2)/4)**0.5
    E_L[i] = (E_C[i] +E_X[i])/2 - (g**2 + ((E_C[i] - E_X[i])**2)/4)**0.5


#plt.plot(kx_All,E_C,'r--')
#plt.plot(kx_All,E_X,'r--')
#plt.plot(kx_All,E_U,'k')
#plt.plot(kx_All,E_L,'k')



#R = R/(R.max())
#R_gr = cv.Laplacian(R, cv.CV_64F, ksize=3)
R_gr = R
#R_gr = R_gr/(R_gr.max())
#R_gr = -20*cv.Laplacian(R_gr, cv.CV_64F, ksize=3)
#R_gr = R_gr/(R_gr.max())
#R_all = np.concatenate((np.fliplr(np.delete(R_gr,0,1)),R_gr),axis=1)

Y = Y.ravel()
split_index = np.searchsorted(Y, 2.25)

R1 = R[:split_index, :]
R2 = R[split_index:, :]


print("r1", R1.shape)
print("r2", R2.shape)

#print(R1)
#print(R2)

R_gr2 = R2
R_gr2 = -20*cv.Laplacian(R_gr, cv.CV_64F, ksize=3)
R_all2 = np.concatenate((np.fliplr(np.delete(R_gr2,0,1)),R_gr),axis=1)

#R_gr = R_gr/(R_gr.max())
 
R_gr1 = R1
R_all1 = np.concatenate((np.fliplr(np.delete(R_gr1,0,1)),R_gr),axis=1)

#R_all = np.vstack((R_all1, R_all2)) 


plt.pcolormesh(X1, Y1, R_all1, cmap='hot',shading='auto',vmin=0,vmax=1)
plt.xlabel(u"kxy (2\u03c0\u03bc$^{\minus}$\u00b9)",fontweight='bold')
plt.ylabel("Nang luong (eV)",fontweight='bold')
plt.axis([-k_max,k_max,2.0,2.8])
#plt.axis([-k_max,k_max,eV_min,eV_max])
plt.xticks([t for t in np.arange(-k_max,k_max+0.1,k_max/2)])
plt.yticks([t for t in np.arange(2.0,2.8+eV_step,0.1)])
cbar = plt.colorbar()
cbar.set_ticks([t for t in np.arange(0,1,0.2)])
cbar.set_ticklabels(["{:.4f}".format(t) for t in np.arange(0,1,0.2)])
plt.savefig('./graphics/'+str(d)+'_'+str(t2)+'_'+str(h)+'_'+str(FF)+'_'+str(t1)+'_'+str(pos)+'_'+str(eV_min)+'_'+str(eV_max)+'_'+str(eV_step)+'_'+str(k_max)+'_'+str(k_step)+'_'+'kx-E-refl.png',dpi=600)
plt.show()



bar.next()
bar.finish()
