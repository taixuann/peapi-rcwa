#!/usr/bin/env python3
#---------------------------------------------------------------------------------------------------------%
#Packages
#---------------------------------------------------------------------------------------------------------%

import numpy as np #numpy / numpy.matlib: arrays, math operations.
import numpy.matlib 
import math #math: trigonometric functions, etc.
import argparse #argparse: parse command-line arguments.
import sys, getopt #system operations.
import S4 as S4 #electromagnetic solver (RCWA method).
from progress.bar import IncrementalBar #shows progress bar.
import e_n_PEAPI #your custom dielectric model function (Lorentz oscillator).
import e_n_SiO2
import matplotlib.pyplot as plt #plotting (though in this script it’s unused).
np.float=float

#---------------------------------------------------------------------------------------------------------%
#Parser values
#---------------------------------------------------------------------------------------------------------%

parser = argparse.ArgumentParser() #Sets up input parameters from command line, so you run the script like:
parser.add_argument('-d', type=float, required=True, help="Period of the photonic structure (d)") #the spacing of the repeating structure
parser.add_argument('-t', type=float, required=True, help="Thickness of the SiO2 (t2)")  #Thickness of spacer under PEAPI
parser.add_argument('--height', type=float, required=True, help="Thickness of the periodic layer (h)") #The height of periodic modulation
parser.add_argument('-f', type=float, required=True, help="Filling factor (FF)") #Ratio of the grating ridge width to the period d (For example, FF = 0.5 means half the period is “ridge” and half is “groove.”)
parser.add_argument('-a', type=float, required=True, help="Thickness of the active layer (t1)") #PEAI thickness
parser.add_argument('--emin', type=float, required=True, help="Minimum radiation energy") #Lower bound of the photon energy range for the simulation (in eV).
parser.add_argument('--emax', type=float, required=True, help="Maximum radiation energy") #Upper bound of the photon energy range (in eV).
parser.add_argument('--estep', type=float, required=True, help="Radiation energy resolution") #Step size (in eV) between energies in the simulation sweep. Smaller step = finer spectral resolution, but longer computation time.
parser.add_argument('--kmax', type=float, required=True, help="Maximum inplane radiation wavenumber") #The maximum parallel momentum (k‖) of incoming photons to scan, usually normalized to k0 (free-space wavenumber).
parser.add_argument('--kstep', type=float, required=True, help="Radiation inplane radiation resolution") #Step size in k‖ between simulation points. Controls how finely the dispersion is sampled.
parser.add_argument('--pos', type=float, required=True, help="Position of active layer in the slab (from 0 to 1)") #Describes the vertical position of the PEAPI layer within the slab geometry.
parser.add_argument('--NoX', type=int, required=True, help="Calculation without exciton (--NoX 1)") #NoX = 1 (ignore excitonic), NoX = 0 (include excitonic effect)
args = parser.parse_args()

#---------------------------------------------------------------------------------------------------------%
#Print values 
#---------------------------------------------------------------------------------------------------------%

#print(args)- Assigns the parsed arguments to variables.
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
#Function of S4 simulation
#---------------------------------------------------------------------------------------------------------%

def S4Module(eV, e_r_PEAPI, e_i_PEAPI,n_PEAPI, d_PEAPI, e_r_SiO2, e_i_SiO2,n_SiO2, d_SiO2, D, h, k, N): #Defines a helper function to run S4 simulation for given photon energy eV and structure parameters.
    S = S4.New(Lattice=((D,0),(0,0)), NumBasis=N) #Lattice = ((D,0),(0,0)): 1D periodic structure with period D. NumBasis=N: number of Fourier modes used in RCWA.

    #---------------------------------------------------------------------------------------------------------%
    #Material definition
    #---------------------------------------------------------------------------------------------------------%

    S.SetMaterial(Name = "Air", Epsilon = 1 + 0*1j) #Air with permittivity = 1.
    S.SetMaterial(Name = "Perovskite", Epsilon = e_r_PEAPI + e_i_PEAPI*1j) #Perovskite with complex permittivity from your dielectric model.
    S.SetMaterial(Name = "SiO2", Epsilon = e_r_SiO2 + e_i_SiO2*1j) #Perovskite with complex permittivity from your dielectric model.

    #---------------------------------------------------------------------------------------------------------%
    #Structure construction
    #---------------------------------------------------------------------------------------------------------%

    S.AddLayer('Above',0,'SiO2') #The simulation assumes light is coming from the top of the PEAI
    #S.AddLayer('SiO2_top', (d_SiO2), 'SiO2')
    S.AddLayer('PEAPI', d_PEAPI, 'Perovskite')
    #S.AddLayer('SiO2_bottom', (d_SiO2),'SiO2')
    S.AddLayer('Below', 0,'SiO2') #This layer is semi-infinite air

    #---------------------------------------------------------------------------------------------------------%
    #Source construction
    #---------------------------------------------------------------------------------------------------------%

    angle = abs(np.arcsin(k*1.2398419/eV)*180/math.pi) #Converts in-plane wavevector k into incidence angle (degrees), using dispersion relation. 1.2398419 is the conversion factor (eV·µm) between energy and wavelength.
    Ampl = 1
    S.SetExcitationPlanewave(IncidenceAngles=(angle,0), sAmplitude=Ampl+0*1j, pAmplitude=1, Order=0) #Incident plane wave at calculated angle. sAmplitude and pAmplitude: polarization. Here both s and p polarizations are excited. Order = 0: normal order input.
    S.SetFrequency(eV/1.2398419) #Sets the frequency for S4, converting energy (eV) → frequency (1/µm).

    #---------------------------------------------------------------------------------------------------------%
    #Transmissiton, Reflection result
    #---------------------------------------------------------------------------------------------------------%

    (Tc_0, Rc_0) = S.GetPowerFlux('Above',-1) # gives the transmitted or reflected power flux (Above = Top of the structure, -1 means reflective output)
    ##Rc_0 = S.GetLayerVolumeIntegral('PEAPI', Quantity = 'U')
    return(Tc_0, Rc_0)

#---------------------------------------------------------------------------------------------------------%
#Define before simulation
#---------------------------------------------------------------------------------------------------------%

N = 1 #Sets number of Fourier harmonics N=1, only 0th and ±1st harmonics

#(e_r_PEAPI_cav, e_i_PEAPI_cav, n_PEAPI_cav, k_PEAPI_cav) = e_n_PEAPI.Cal(1.2398419/d) #Calculates dielectric function of PEAPI at energy = lattice reciprocal value (just for cavity baseline).

eV_range = np.arange(eV_min, eV_max + np.float(eV_step/2), eV_step) #Builds an array of photon energies (in eV) from eV_min → eV_max with step eV_step. + np.float(eV_step/2) ensures the last point is included.
kx = np.arange(k_min, k_max + np.float(k_step/2), k_step) #Builds array of in-plane momenta from 0 → k_max with step k_step.
Rc_above = np.zeros((eV_range.size, kx.size)) #Rc_above, Tc_above: store results.
Tc_above = np.zeros((eV_range.size, kx.size))
A_above = np.zeros((eV_range.size, kx.size))

#---------------------------------------------------------------------------------------------------------%
#Processing section
#---------------------------------------------------------------------------------------------------------%

eV = eV_min
j = 0
if ((t1 != 0) and (calculNoX == 1)): #Calculate if we don't include exciton
    bar = IncrementalBar('Calculating', max = eV_range.size*2, suffix = '%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
else: #Include exciton
    bar = IncrementalBar('Calculating', max = eV_range.size, suffix = '%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
while eV < (eV_max + eV_step/2): #Outer loop over photon energy
    (e_r_PEAPI, e_i_PEAPI, n_PEAPI, k_PEAPI) = e_n_PEAPI.Cal(eV) #Gets dielectric constants.
    (e_r_SiO2, e_i_SiO2, n_SiO2, k_SiO2) = e_n_SiO2.Cal(eV) #Gets dielectric constants.
    k = 0
    wv = k_min
    while wv < (k_max + k_step/2):
        (Tc_0, Rc_0) = S4Module(eV, e_r_PEAPI, e_i_PEAPI, n_PEAPI, t1,e_r_SiO2, e_i_SiO2, n_SiO2, t2, d, h, wv, N)
        Tc_above[j,k] = np.abs(Tc_0.real) #Computes reflection/transmission ratio for each (energy, momentum) point.
        Rc_above[j,k] = np.abs(Rc_0.real)
        wv += k_step
        k +=1
    eV += eV_step
    bar.next()
    j += 1

Ratio_above = Rc_above/(Tc_above+1e-12) #+1e-12 to avoid division by zero
A_above = 1 - Rc_above - Tc_above

np.savetxt('./data/'+str(d)+'_'+str(t2)+'_'+str(h)+'_'+str(FF)+'_'+str(t1)+'_'+str(pos)+'_'+str(eV_min)+'_'+str(eV_max)+'_'+str(eV_step)+'_'+str(k_max)+'_'+str(k_step)+'_'+'k-E-refl (SiO2).csv',Ratio_above,delimiter=',')

bar.finish()
