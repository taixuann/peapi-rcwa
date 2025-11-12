#!/usr/bin/env python3

#---------------------------------------------------------------------------------------------------------%
#Packages
#---------------------------------------------------------------------------------------------------------%

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
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import e_n_PEAPI
import e_n_SiO2
np.float = float

#---------------------------------------------------------------------------------------------------------%
#Define parameters
#---------------------------------------------------------------------------------------------------------%

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

####debugging only (to check arguments).  Once it works, they become noisy.####

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

#---------------------------------------------------------------------------------------------------------%
#Extract value
#---------------------------------------------------------------------------------------------------------%

eV_range = np.arange(eV_min,eV_max+np.float(eV_step/2),eV_step) #eV / eV_range: all photon energies you simulated.
ev_min = np.arange(eV_min,2.394+np.float(eV_step/2),eV_step) #split energy range into two regions (below and above 2.394 eV).
ev_max = np.arange(2.394,eV_max+np.float(eV_step/2),eV_step) #split energy range into two regions (below and above 2.394 eV).

#eV range makes a list of all the photon energies you simulated from eV_min → eV_max, in steps of eV_step. Example: if you scan 1.5 eV → 3.5 eV with step 0.01, you get [1.5, 1.51, 1.52, …, 3.5]. On your y-axis of the dispersion plot (Energy axis).#
# ev_min: from eV_min up to 2.394 eV (your exciton energy). And ev_max: from 2.394 eV up to eV_max. Think of it as cutting the energy axis into “below exciton” and “above exciton” regions.#

kx = np.arange(k_min,k_max+np.float(k_step/2),k_step) # kx/ in-plane momentum (0 to k_max).
kx_All = np.arange(-k_max,k_max+k_step/2,k_step) # kx_all/symmetrized (–k to +k).

# kx makes a list of positive in-plane wavevectors: from k_min (usually 0) → k_max, in steps of k_step. Example: if k_max=1, k_step=0.1, you get [0, 0.1, 0.2, …, 1]. On your x-axis of the dispersion plot (momentum axis).#
#kx_all is same as kx, but mirrored to cover both positive and negative k: [-k_max, …, 0, …, +k_max]. This makes the dispersion plot symmetric about k=0.#

X, Y = np.meshgrid(kx_All, eV_range) #meshgrid for plotting heatmaps (kx vs. energy).
X1, Y1 = np.meshgrid(kx_All, ev_min)
X2, Y2 = np.meshgrid(kx_All, ev_max)

# X and Y create a grid of coordinates for plotting. Every point in X is a k-value, every point in Y is an energy. Without this, you can’t use plt.pcolormesh to make the heatmap.#
# X1 and Y1 are same as above, but only for the split energy ranges (ev_min, ev_max). Useful if you want to plot the “below exciton” and “above exciton” regions separately.#

#---------------------------------------------------------------------------------------------------------%
#Load reflectivity from file (R is the reflectivity as a 2D array: rows = energy, columns = kx.)
#---------------------------------------------------------------------------------------------------------%

####Extract the reflection data calculation####

bar = IncrementalBar('Calculating',max=1, suffix='%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
with open('./data/'+str(d)+'_'+str(t2)+'_'+str(h)+'_'+str(FF)+'_'+str(t1)+'_'+str(pos)+'_'+str(eV_min)+'_'+str(eV_max)+'_'+str(eV_step)+'_'+str(k_max)+'_'+str(k_step)+'_'+'k-E-refl (SiO2).csv') as csvfileR:
	R = np.loadtxt(csvfileR, delimiter=',')
bar.next()
bar.finish()

####Font setup####

font = font_manager.FontProperties(family='DejaVu Sans', style='normal', size = 25)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 20
#plt.rcParams['text.usetex'] = True

#---------------------------------------------------------------------------------------------------------%
#Drawing theoretical exciton-polariton coupling to compare with experimental data. (Ideal equaiton without damping factor)
#---------------------------------------------------------------------------------------------------------%

####Define parameters####
E_0 = 2.394 # bare cavity energy at k=0 (eV)
g = 0.230 # coupling strength (half Rabi splitting)
E_C = np.empty(len(kx_All)) # cavity parabola
E_X = np.empty(len(kx_All)) # exciton (flat)
E_U = np.empty(len(kx_All)) # upper polariton
E_L = np.empty(len(kx_All)) # lower polariton

####Equation of ideal coupled oscillator model####

for i in range (len(kx_All)):
    E_C[i] = E_0 + 0.180*kx_All[i]**2 # parabola for cavity dispersion
    E_X[i] = 2.394 ## exciton energy (flat)
    E_U[i] = (E_C[i] +E_X[i])/2 + (g**2 + ((E_C[i] - E_X[i])**2)/4)**0.5
    E_L[i] = (E_C[i] +E_X[i])/2 - (g**2 + ((E_C[i] - E_X[i])**2)/4)**0.5

####Print theoretical curves energy####
#fig = plt.figure(dpi=200)
#plt.plot(kx_All,E_C,'r--')
#plt.plot(kx_All,E_X,'r--')
#plt.plot(kx_All,E_U,'w--', linewidth=1.2)
#plt.plot(kx_All,E_L,'w--', linewidth=1.2)

#---------------------------------------------------------------------------------------------------------%
#Images processing
##want raw data → use R_gr = R.
##want to enhance visibility of branches → normalize + Laplacian.
##the Laplacian is too strong / noisy → comment it and stick to raw normalized R.
##Optionally mirror across k=0 to get full dispersion.
#---------------------------------------------------------------------------------------------------------%

####First method: Enhance lower polariton visibility####

#R = R / R.max()  # normalize 0–1

# Apply Laplacian to enhance edges (optional)
apply_laplacian = False
if apply_laplacian:
    R_gr = cv.Laplacian(R, cv.CV_64F, ksize=3)
    R_gr = R_gr / np.abs(R_gr).max()
else:
    R_gr = R
#If you set apply_laplacian=True, the code applies an edge-detection filter. This emphasizes sharp features in the reflectivity (the dark polariton lines). It’s like adjusting contrast in an image to highlight the exciton–cavity hybridization.	But it can also add noise, so it’s optional#

# Selectively enhance the lower-energy part (below exciton)
split_index = np.searchsorted(eV_range, 2.394)
R_lower = R_gr[:split_index, :]
R_upper = R_gr[split_index:, :]

# Merge again
R_gr = np.vstack((R_lower, R_upper))
R_all = np.concatenate((np.fliplr(np.delete(R_gr,0,1)),R),axis=1)

####Second method: trick to “enhance visibility” in one region, but at the cost of introducing artifacts.####

print("r1", R_lower.shape)
print("r2", R_upper.shape)
#print(R1)
#print(R2)

#This part above is splitting your reflectivity map into lower energy part (below exciton) and higher energy part (above exciton).#


#R_gr2 = R2 #Take the upper half (R2).
#R_gr2 = -20*cv.Laplacian(R2, cv.CV_64F, ksize=3) #Apply a Laplacian filter for the upper polariton branch
#R_all2 = np.concatenate((np.fliplr(np.delete(R_gr2,0,1)),R2),axis=1) #Store value in R_all2 and also the negative part through flip function

#R_gr1 = R1

#R_all1 = np.concatenate((np.fliplr(np.delete(R_gr1,0,1)),R1),axis=1)
#R_all = np.vstack((R_all1, R_all2))  #Glues the “lower energy half” and “upper energy half” back together into one big reflectivity map.#

#This part above is define the reflectivitivity map separately upper and lower branch polariton, and apply the Laplacian for the upper polariton. Finally glue all the upper and lower by R_all#


# ------------------------
# Plotting: Reflectivity
# ------------------------

plt.pcolormesh(X, Y, R_all, cmap='hot',shading='auto',vmin=0,vmax=0.1)
plt.axhline(2.394, color='cyan', linestyle='--', linewidth=1)

plt.xlabel(u"kxy (2\u03c0\u03bc$^{\minus}$\u00b9)", fontweight='bold', labelpad=15)
plt.ylabel("Energy (eV)", fontweight='bold', labelpad=15)

plt.axis([-k_max, k_max, eV_min, eV_max])
#plt.axis([-k_max, k_max, -2.0, 2.0])

plt.xticks([t for t in np.arange(-k_max, k_max+0.1, k_max/2)])
plt.yticks(np.arange(eV_min, eV_max, 0.2))

cbar = plt.colorbar(label='Reflectivity')
cbar.set_ticks([t for t in np.arange(0,1.1,0.2)])
cbar.set_ticklabels(["{:.2f}".format(t) for t in np.arange(0,1.1,0.2)])

plt.savefig('./graphics/'+str(d)+'_'+str(t2)+'_'+str(h)+'_'+str(FF)+'_'+str(t1)+'_'+str(pos)+'_'+str(eV_min)+'_'+str(eV_max)+'_'+str(eV_step)+'_'+str(k_max)+'_'+str(k_step)+'_'+'kx-E-refl (SiO2).png',dpi=600)
plt.show()

bar.next()
bar.finish()