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
    parser.add_argument('--config', type=str, required=True, choices=['air', 'sio2'], help="Structure configuration: 'air' for Air/PEAPI/Air or 'sio2' for SiO2/PEAPI/SiO2")
    
    # Create a mutually exclusive group for the run mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Run the full simulation (calculation and plotting).')
    group.add_argument('--structure', action='store_true', help='Only visualize the structure drawing.')
    group.add_argument('--perp', action='store_true', help='Only plot the reflectivity dip at k_parallel = 0.')
    group.add_argument('--paral', action='store_true', help='Plot the reflectivity vs energy and mark the polariton resonances.')
    group.add_argument('--calc', action='store_true', help='Only run the S4 calculation and save the reflectivity data.')
    

    args = parser.parse_args()

    # --- Hardcoded constant parameters ---
    eV_min = 1.6
    eV_max = 3.2
    eV_step = 0.001
    k_min = 0.0
    k_max = 2.6
    k_step = 0.01
    d = 1.0
    NoX = 1
    t = 0.0
    N = 1
    E_X = 2.3478
    config = args.config

    # --- Parameters to vary in a loop ---
    # You can vary multiple parameters using nested loops.
    # For example, to vary 'a' (active layer thickness):
    
    active_layer_thicknesses = [6.0] # Example values in um for t1

    for a in active_layer_thicknesses:
        if args.all:
            print(f"\n--- Running simulation for active layer thickness a = {a} um ---")
            S4_calculation.S4_calculation(eV_min, eV_max, eV_step, k_min, k_max, k_step, a, NoX, d, t, config, N)
            plotting.polariton_dispersion_parallel(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, t, a, E_X, config)
            plotting.polariton_dispersion_perp(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, t, a, E_X, config)
        elif args.calc:
            S4_calculation.S4_calculation(eV_min, eV_max, eV_step, k_min, k_max, k_step, a, NoX, d, t, config, N)
        elif args.perp:
            plotting.polariton_dispersion_perp(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, t, a, E_X, config)
        elif args.paral:
            plotting.polariton_dispersion_parallel(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, t, a, E_X, config)
        elif args.structure:
            print(f"\n--- Visualizing the structure drawing ---")
            plotting.visualize_structure(config, a, d)

if __name__ == '__main__':
    main()

