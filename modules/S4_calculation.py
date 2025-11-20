import numpy as np
from progress.bar import IncrementalBar
import modules.S4_system as S4_system
from modules import material, utils

def S4_calculation(eV_min, eV_max, eV_step, k_min, k_max, k_step, a, t_air, t_SiO2, d, NoX, config, N):
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
        (e_r_PEAPI, e_i_PEAPI, n_PEAPI, _) = material.PEAPI(eV) #Gets dielectric constants.
        (e_r_SiO2, e_i_SiO2, _, _) = material.SiO2(eV)
        k = 0
        wv = k_min
        while wv < (k_max + k_step/2):
            if config == "air":
                (Tc_0, Rc_0) = S4_system.S4Module_Air_PEAPI_Air(eV, e_r_PEAPI, e_i_PEAPI, n_PEAPI, a, d, wv, N)
            elif config =="sio2":
                (Tc_0, Rc_0) = S4_system.S4Module_SiO2_PEAPI_SiO2(eV, e_r_PEAPI, e_i_PEAPI, e_r_SiO2, e_i_SiO2, a, d, wv, N)
            elif config =="air-sio2-sio2-air":
                (Tc_0, Rc_0) = S4_system.S4Module_SiO2_Air_PEAPI_Air_SiO2(eV, e_r_PEAPI, e_i_PEAPI, e_r_SiO2, e_i_SiO2, a, t_air, t_SiO2, d, wv, N)
            elif config =="air-sio2-air":
                (Tc_0, Rc_0) = S4_system.S4Module_Air_PEAPI_SiO2_Air(eV, e_r_PEAPI, e_i_PEAPI, e_r_SiO2, e_i_SiO2, a, t_air, t_SiO2, d, wv, N)
            Tc_above[j,k] = np.abs(Tc_0.real) #Computes reflection/transmission ratio for each (energy, momentum) point.
            Rc_above[j,k] = np.abs(Rc_0.real)
            wv += k_step
            k +=1
        eV += eV_step
        bar.next()
        j += 1

    Ratio_above = Rc_above/(Tc_above)

    config_name = utils.config(config)
    filename = utils.generate_filename_base(config_name, d, a, t_air, t_SiO2, eV_min, eV_max, eV_step, k_max, k_step)
    np.savetxt(f"./data/" + filename + 'k-E-reflectivity.csv', Ratio_above, delimiter=',')
    
    bar.finish()


