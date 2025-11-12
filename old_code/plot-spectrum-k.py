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

#---------------------------------------------------------------------------------------------------------%
#Extract value
#---------------------------------------------------------------------------------------------------------%

eV_range = np.arange(eV_min,eV_max+np.float(eV_step/2),eV_step) #eV / eV_range: all photon energies you simulated.
ev_min = np.arange(eV_min,2.394+np.float(eV_step/2),eV_step) #split energy range into two regions (below and above 2.394 eV).
ev_max = np.arange(2.394,eV_max+np.float(eV_step/2),eV_step) #split energy range into two regions (below and above 2.394 eV).

kx = np.arange(k_min,k_max+np.float(k_step/2),k_step) # kx/ in-plane momentum (0 to k_max).
kx_All = np.arange(-k_max,k_max+k_step/2,k_step) # kx_all/symmetrized (–k to +k).

X, Y = np.meshgrid(kx_All, eV_range) #meshgrid for plotting heatmaps (kx vs. energy).
X1, Y1 = np.meshgrid(kx_All, ev_min)
X2, Y2 = np.meshgrid(kx_All, ev_max)

#---------------------------------------------------------------------------------------------------------%
#Load reflectivity from file (R is the reflectivity as a 2D array: rows = energy, columns = kx.)
#---------------------------------------------------------------------------------------------------------%

bar = IncrementalBar('Calculating',max=1, suffix='%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
with open('./data/'+str(d)+'_'+str(t2)+'_'+str(h)+'_'+str(FF)+'_'+str(t1)+'_'+str(pos)+'_'+str(eV_min)+'_'+str(eV_max)+'_'+str(eV_step)+'_'+str(k_max)+'_'+str(k_step)+'_'+'k-E-refl.csv') as csvfileR:
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
#Images processing
#---------------------------------------------------------------------------------------------------------%

#R = R / R.max()  # normalize 0–1
E_X = 2.3478

# Apply Laplacian to enhance edges (optional)
apply_laplacian = False
if apply_laplacian:
    R_gr = cv.Laplacian(R, cv.CV_64F, ksize=3)
    R_gr = R_gr / np.abs(R_gr).max()
else:
    R_gr = R
#If you set apply_laplacian=True, the code applies an edge-detection filter. This emphasizes sharp features in the reflectivity (the dark polariton lines). It’s like adjusting contrast in an image to highlight the exciton–cavity hybridization.	But it can also add noise, so it’s optional#

# Selectively enhance the lower-energy part (below exciton)
split_index = np.searchsorted(eV_range, 2.3478)
R_lower = R_gr[:split_index, :]
R_upper = R_gr[split_index:, :]

# Merge again
R_gr = np.vstack((R_lower, R_upper))
R_all = np.concatenate((np.fliplr(np.delete(R_gr,0,1)),R),axis=1)

print("r1", R_lower.shape)
print("r2", R_upper.shape)

# ------------------------
# Plotting: Reflectivity
# ------------------------

plt.pcolormesh(X, Y, R_all, cmap='hot',shading='auto',vmin=0,vmax=1)
plt.axhline(2.3478, color='cyan', linestyle='--', linewidth=1)

plt.xlabel(u"kxy (2\u03c0\u03bc$^{\minus}$\u00b9)", fontweight='bold', labelpad=15)
plt.ylabel("Energy (eV)", fontweight='bold', labelpad=15)

plt.axis([-k_max, k_max, eV_min, eV_max])
#plt.axis([-k_max, k_max, -2.0, 2.0])

plt.xticks([t for t in np.arange(-k_max, k_max+0.1, k_max/2)])
plt.yticks(np.arange(eV_min, eV_max, 0.2))

cbar = plt.colorbar(label='Reflectivity')
cbar.set_ticks([t for t in np.arange(0,1.1,0.2)])
cbar.set_ticklabels(["{:.2f}".format(t) for t in np.arange(0,1.1,0.2)])

plt.savefig('./graphics/'+str(d)+'_'+str(t2)+'_'+str(h)+'_'+str(FF)+'_'+str(t1)+'_'+str(pos)+'_'+str(eV_min)+'_'+str(eV_max)+'_'+str(eV_step)+'_'+str(k_max)+'_'+str(k_step)+'_'+'kx-E-refl.png',dpi=600)
plt.show()

bar.next()
bar.finish()

# ------------------------
# Plotting: Reflectivity at k = 0 and extracting dispersion
# ------------------------
#1) Plot reflectivity trace at k|| = 0 and mark dips
# Determine center column index corresponding to k|| = 0
center_idx = R_all.shape[1] // 2 # This line means that take 1 column in R_all with the integer with 2 midpoint
E_axis = eV_range  # energy axis for rows
R_k0 = R_all[:, center_idx] # This one retrieves the data from R_all to R_k0 to assign R at k// = 0

# Find minima on the k=0 trace (invert reflectivity to find dips)
prominence_rel = 0.02  # This defines how “strong” a dip must be to be detected. 0.02 means: a dip must stand out by at least 2% of the full intensity range to be counted.
# absolute prominence: choose value relative to dynamic range
abs_prom = max(1e-4, prominence_rel * (np.nanmax(R_k0) - np.nanmin(R_k0))) #The code converts that relative prominence into an absolute threshold. It calculates the dynamic range of reflectivity (max - min), multiplies by 0.02 (your relative factor).
peaks_k0, props_k0 = find_peaks(-R_k0, prominence=abs_prom, distance=3) #-R_k0 flips the reflectivity trace vertically, so dips become peaks. find_peaks() then finds these “peaks” (the original dips). prominence=abs_prom ensures only strong enough dips are kept. distance=3 ensures detected dips are at least 3 data points apart (to avoid double-counting small ripples).

# Sort peaks by prominence (largest first)
if 'prominences' in props_k0:
    order = np.argsort(props_k0['prominences'])[::-1]
else:
    order = np.argsort(-R_k0[peaks_k0])
peaks_k0 = peaks_k0[order]

# Plot the k=0 reflectivity trace and mark dips
plt.figure(figsize=(7,5), dpi=200)
plt.plot(E_axis, R_k0, lw=1.5, label='R (k//=0)')
colors = plt.cm.tab10(np.linspace(0,1,max(1,len(peaks_k0))))
for idx, p in enumerate(peaks_k0):
    E_p = E_axis[p]
    R_p = R_k0[p]
    prom = props_k0['prominences'][order[idx]] if 'prominences' in props_k0 else None
    plt.scatter(E_p, R_p, color=colors[idx % len(colors)], s=60, edgecolor='k', zorder=5)
    #txt = f"E={E_p:.3f} eV\nR={R_p:.3f}"
    #if prom is not None:
       #txt += f"\nprom={prom:.3f}"
    #plt.annotate(txt, xy=(E_p, R_p), xytext=(5, -30-idx*10), textcoords='offset points',
                 #fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

#plt.gca().invert_yaxis()  # optional: dips appear downward, invert for visual preference (comment out if undesired)
plt.xlabel("Energy (eV)")
plt.ylabel("Reflectivity (a.u.)")
plt.title("Reflectivity at k|| = 0 with detected dips")
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.savefig(out_k0, dpi=300, bbox_inches='tight')
plt.show()

# ----- 2) Extract dips across k to build dispersion (LP and UP) -----
num_cols = R_all.shape[1]
E_LP = np.full(num_cols, np.nan)
E_UP = np.full(num_cols, np.nan)

# dynamic absolute prominence for each column (robust to varying contrast)
for i in range(num_cols):
    col = R_all[:, i]
    if np.all(np.isnan(col)):
        continue
    rng = np.nanmax(col) - np.nanmin(col)
    abs_prom_col = max(1e-5, 0.02 * rng)
    peaks, props = find_peaks(-col, prominence=abs_prom_col, distance=3)
    if peaks.size == 0:
        # nothing detected for this k
        continue
    # For each detected minima, get energy and reflectivity value
    E_peaks = E_axis[peaks]
    R_peaks = col[peaks]

    # separate peaks below/above exciton
    below_mask = E_peaks < E_X
    above_mask = E_peaks > E_X

    # choose the deepest (smallest R) peak below exciton for LP
    if np.any(below_mask):
        idxs = np.where(below_mask)[0]
        chosen = idxs[np.argmin(R_peaks[idxs])]
        E_LP[i] = E_peaks[chosen]
    # choose the deepest (smallest R) peak above exciton for UP
    if np.any(above_mask):
        idxs = np.where(above_mask)[0]
        chosen = idxs[np.argmin(R_peaks[idxs])]
        E_UP[i] = E_peaks[chosen]

# Interpolate NaNs to get continuous-looking curves (linear interpolation)
def interp_nans(x, y):
    # x: indices (or k positions); y: values with NaNs
    good = ~np.isnan(y)
    if good.sum() < 2:
        return y  # nothing to interpolate
    xi = np.arange(len(y))
    y_interp = y.copy()
    y_interp[~good] = np.interp(xi[~good], xi[good], y[good])
    return y_interp

E_LP_interp = interp_nans(np.arange(num_cols), E_LP)
E_UP_interp = interp_nans(np.arange(num_cols), E_UP)

# Map column indices to k|| axis (kx_All) used for plotting
# kx_All should match R_all number of columns. If mismatch, construct linear mapping.
if len(kx_All) == num_cols:
    k_axis = kx_All
else:
    k_axis = np.linspace(-k_max, k_max, num_cols)

# ----- 4) Plot Energy vs calculated k -----

# Calculate the free-space wavevector k = 2*pi*E / (hc)
# where hc = 1.23984 eV*um
hc = 1.23984

# Light line in vacuum (n=1)
k_light_line = 2 * np.pi * E_axis / hc

# k values for the polariton branches from their energies (using raw dip data, not interpolated)
k_LP = 2 * np.pi * E_LP / hc
k_UP = 2 * np.pi * E_UP / hc

plt.figure(figsize=(7,5), dpi=200)
plt.plot(E_axis,k_light_line, 'k--', label='Light Line (n=1)')
plt.scatter(E_LP, k_LP, color='cyan', s=15, label='LPB dips')
plt.scatter(E_UP, k_UP, color='magenta', s=15, label='UPB dips')
plt.xlabel("Calculated k (μm⁻¹)")
plt.ylabel("Energy (eV)")
plt.title("Energy vs. Calculated Wavevector")
plt.legend()
plt.grid(alpha=0.4)

# --- Auto-adjust axis to focus on the data ---
# Combine all valid data points to find the plotting range
k_data = np.concatenate((k_LP, k_UP))
E_data = np.concatenate((E_LP, E_UP))

# Find min/max of the data, ignoring NaNs, and add a 5% margin
k_min_lim, k_max_lim = np.nanmin(k_data) * 0.95, np.nanmax(k_data) * 1.05
E_min_lim, E_max_lim = np.nanmin(E_data) * 0.98, np.nanmax(E_data) * 1.02

plt.axis([E_min_lim, E_max_lim, k_min_lim, k_max_lim])
plt.show()