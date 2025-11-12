import numpy as np
from progress.bar import IncrementalBar
import modules.S4_system as S4_system
from modules import material

def S4_calculation(eV_min, eV_max, eV_step, k_min, k_max, k_step, a, NoX, d, t, config, N):
    eV_range = np.arange(eV_min, eV_max + np.float(eV_step/2), eV_step)
    kx = np.arange(k_min, k_max + np.float(k_step/2), k_step) 
    Rc_above = np.zeros((eV_range.size, kx.size))
    Tc_above = np.zeros((eV_range.size, kx.size))

    eV = eV_min
    j = 0
    if ((a != 0) and (NoX == 1)): #Calculate if we don't include exciton
        bar = IncrementalBar('Calculating', max = eV_range.size*2, suffix = '%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
    else: #Include exciton
        bar = IncrementalBar('Calculating', max = eV_range.size, suffix = '%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
    while eV < (eV_max + eV_step/2): #Outer loop over photon energy
        (e_r_PEAPI, e_i_PEAPI, n_PEAPI, k_PEAPI) = material.PEAPI(eV) #Gets dielectric constants.
        (e_r_SiO2, e_i_SiO2, n_SiO2, k_SiO2) = material.SiO2(eV)
        k = 0
        wv = k_min
        while wv < (k_max + k_step/2):
            if config == "air":
                (Tc_0, Rc_0) = S4_system.S4Module_Air_PEAPI_Air(eV, e_r_PEAPI, e_i_PEAPI, n_PEAPI, a, d, wv, N)
            elif config =="sio2":
                (Tc_0, Rc_0) = S4_system.S4Module_SiO2_PEAPI_SiO2(eV, e_r_PEAPI, e_i_PEAPI, e_r_SiO2, e_i_SiO2, a, d, wv, N)
            Tc_above[j,k] = np.abs(Tc_0.real) #Computes reflection/transmission ratio for each (energy, momentum) point.
            Rc_above[j,k] = np.abs(Rc_0.real)
            wv += k_step
            k +=1
        eV += eV_step
        bar.next()
        j += 1

    Ratio_above = Rc_above/(Tc_above)

    if config == "air":
        config_name = "Configuration of Air_PEAPI_Air_"
    else:
        config_name = "Configuration of SiO2_PEAPI_SiO2_"

    filename = ('./data/' + config_name + "d(" + str(d) + ")_" + "t(" + str(t) + ")_" + "a(" + str(a) + ")_" + "eV_min(" + str(eV_min)+')_' + "eV_max(" + str(eV_max)+')_' + "eV_step(" + str(eV_step) + ')_' + "k_max(" + str(k_max) + ')_'+ "k_step(" + str(k_step) + ')_' + 'k-E-reflectivity.csv')
    np.savetxt(filename, Ratio_above, delimiter=',')
    
    bar.finish()


