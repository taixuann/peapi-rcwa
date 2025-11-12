import math
import sys

def Cal(eV):
    p = math.pi
   
    f1 = 0.134045 #1: main exciton resonance (lowest-energy excitonic transition - 1s)
    E1 = 2.3478 #Correct with supplementary information
    y1 = 0.008 #Correct with supplementary information
    f2 = 0
    E2 = 2.3468 #2: higher-energy resonance (Higher excitonic states - 2s,3s,... or interband)
    y2 = 0.45
    f3 = 0
    E3 = 5.1 #another resonance at ~5 eV (capture the deep UV absorption features of the material)
    y3 = 3.9

    e_inf = 3.76 #Correct with supplementary information

    e1 = f1*E1**2/(E1**2 - eV**2 + 1j*y1*eV) #Dielectric function of lowest excitonic transition
    e2 = f2*E2**2/(E2**2 - eV**2 + 1j*y2*eV) #Dielectric function of higher-energy transition
    e3 = f3*E3**2/(E3**2 - eV**2 + 1j*y3*eV) #Dielectrci function of another resonance

    e = e_inf + e1 + e2 + e3
    e_r = e.real
    e_i = abs(e.imag)
    n = (e**0.5).real
    k = abs((e**0.5).imag)

    return (e_r, e_i, n, k)