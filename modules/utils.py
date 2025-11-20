#!usr/bin/env python3

from progress.bar import IncrementalBar
import numpy as np

def config(config):
    if config == "air":
        config_name = "Configuration of Air_PEAPI_Air_"
    elif config == "sio2":
        config_name = "Configuration of SiO2_PEAPI_SiO2_"
    elif config == "air-sio2-sio2-air":
        config_name = "Configuration of Air_SiO2_PEAPI_SiO2_Air_"
    elif config == "air-sio2-air":
        config_name = "Configuration of Air_PEAPI_SiO2_Air_"
    
    return config_name

def generate_filename_base(config_name, d, a, t_air, t_SiO2, eV_min, eV_max, eV_step, k_max, k_step):
    """
    Generates a standardized filename base from simulation parameters.
    
    Uses f-string formatting for readability.
    """
    
    # f-strings (f"...") automatically convert variables like d, t, etc., to strings
    filename_base = (
        f"{config_name}d({d})_a({a})_t_air({t_air})_t_sio2({t_SiO2})_"
        f"eV_min({eV_min})_eV_max({eV_max})_eV_step({eV_step})_"
        f"k_max({k_max})_k_step({k_step})_"
    )
    
    return filename_base

def read_reflectivity_file(filename):
    """
    Reads the reflectivity data from a CSV file.
    """
    bar = IncrementalBar('Calculating',max=1, suffix='%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
    with open(f"./data/" + filename + 'k-E-reflectivity.csv', 'r') as csvfileR:
        R = np.loadtxt(csvfileR, delimiter=',')
    bar.next()
    bar.finish()

    return R