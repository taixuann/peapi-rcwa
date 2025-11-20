import numpy as np
import argparse
from modules import S4_calculation, plotting
import os

# === Assign parser value ===

def main():
    parser = argparse.ArgumentParser()
    np.float = float

    # --- Create directories if they don't exist ---
    
    if not os.path.exists('./data'):
        os.makedirs('./data')
    if not os.path.exists('./graphics'):
        os.makedirs('./graphics')

    # --- Parameters to be passed from command line ---
    parser.add_argument('--config', type=str, required=True, choices=['air', 'sio2', 'air-sio2-sio2-air', 'air-sio2-air'], help="Structure configuration: 'air' for Air/PEAPI/Air, 'sio2' for SiO2/PEAPI/SiO2, or 'sio2-air' for SiO2/Air/PEAPI/Air/SiO2")
    
    # Create a mutually exclusive group for the run mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Run the full simulation (calculation and plotting).')
    group.add_argument('--structure', action='store_true', help='Only visualize the structure drawing.')
    group.add_argument('--perp', action='store_true', help='Only plot the reflectivity dip at k_parallel = 0.')
    group.add_argument('--paral', action='store_true', help='Plot the reflectivity vs energy and mark the polariton resonances.')
    group.add_argument('--calc', action='store_true', help='Only run the S4 calculation and save the reflectivity data.')
    

    args = parser.parse_args()

    # --- Hardcoded constant parameters ---
    eV_min = 1.6 #eV
    eV_max = 3.2 #eV
    eV_step = 0.001 #eV
    k_min = 0.0 #2pi/um
    k_max = 2.6 #2pi/um
    k_step = 0.01 #2pi/um
    d = 1.0 #um
    NoX = 1 
    t_SiO2 = 1000.0 #thickness of SiO2
    t_air = 9999999999.0 #thickness of Air
    N = 1 
    E_X = 2.3478 #exciton energy
    config = args.config

    # --- Parameters to vary in a loop ---
    # You can vary multiple parameters using nested loops.
    # For example, to vary 'a' (active layer thickness):
    
    active_layer_thicknesses = [6.0] # Example values in um for t1

    for a in active_layer_thicknesses:
        if args.all:
            print(f"\n--- Running simulation for active layer thickness a = {a} um ---")
            S4_calculation.S4_calculation(eV_min, eV_max, eV_step, k_min, k_max, k_step, a, t_air, t_SiO2, d, NoX, config, N)
            plotting.polariton_dispersion_parallel(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, a, t_air, t_SiO2, E_X, config)
            plotting.polariton_dispersion_perp(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, a, t_air, t_SiO2, E_X, config)
        elif args.calc:
            S4_calculation.S4_calculation(eV_min, eV_max, eV_step, k_min, k_max, k_step, a, t_air, t_SiO2, d, NoX, config, N)
        elif args.perp:
            plotting.polariton_dispersion_perp(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, a, t_air, t_SiO2, E_X, config)
        elif args.paral:
            plotting.polariton_dispersion_parallel(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, a, t_air, t_SiO2, E_X, config)
        elif args.structure:
            print(f"\n--- Visualizing the structure drawing ---")
            plotting.visualize_structure(config, a, d, t_SiO2)

if __name__ == '__main__':
    main()
