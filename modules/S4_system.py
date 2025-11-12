#!usr/bin/env python3

import numpy as np
import numpy.matlib
import math
import argparse
import sys, getopt
import csv
import sys, getopt
import S4 as S4
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from progress.bar import IncrementalBar
from modules import material

# === Define the variables for Air/PEAPI/Air ===

def S4Module_Air_PEAPI_Air(eV, e_r_PEAPI, e_i_PEAPI, n_PEAPI, a, d, k, N):
    S = S4.New(Lattice=((d,0),(0,0)), NumBasis=N)

    # === Set Material ===
    S.SetMaterial(Name = "Air", Epsilon = 1 + 0*1j)
    S.SetMaterial(Name = "Perovskite", Epsilon = e_r_PEAPI + e_i_PEAPI*1j)

    # === Structure construction ===

    S.AddLayer("Above", 0, "Air")
    S.AddLayer("PEAPI", a, "Perovskite")
    S.AddLayer("Below", 0, "Air")

    # === Source construction ===

    angle = abs(np.arcsin(k*1.2398419/eV)*180/math.pi)
    Ampl = 1
    S.SetExcitationPlanewave(IncidenceAngles=(angle,0), sAmplitude=Ampl+0*1j, pAmplitude=1, Order=0)
    S.SetFrequency(eV/1.2398419)

    # === Transmission, Reflection results

    (Tc_0, Rc_0) = S.GetPowerFlux("Above", -1)
    return (Tc_0, Rc_0)

# === Define the variables for SiO2/PEAPI/SiO2 ===

def S4Module_SiO2_PEAPI_SiO2(eV, e_r_PEAPI, e_i_PEAPI, e_r_SiO2, e_i_SiO2, a, d, k, N):
    S = S4.New(Lattice=((d,0),(0,0)), NumBasis=N)

    # === Set Material ===

    S.SetMaterial(Name = "Air", Epsilon = 1 + 0*1j)
    S.SetMaterial(Name = "Perovskite", Epsilon = e_r_PEAPI + e_i_PEAPI*1j)
    S.SetMaterial(Name = "SiO2", Epsilon = e_r_SiO2 + e_i_SiO2*1j)

    # === Structure construction ===

    S.AddLayer("Above", 0, "SiO2")
    S.AddLayer("PEAPI", a, "Perovskite")
    S.AddLayer("Below", 0, "SiO2")

    # === Source construction ===

    angle = abs(np.arcsin(k*1.2398419/eV)*180/math.pi)
    Ampl = 1
    S.SetExcitationPlanewave(IncidenceAngles=(angle,0), sAmplitude=Ampl+0*1j, pAmplitude=1, Order=0)
    S.SetFrequency(eV/1.2398419)

    # === Transmission, Reflection results

    (Tc_0, Rc_0) = S.GetPowerFlux("Above", -1)
    return (Tc_0, Rc_0)




