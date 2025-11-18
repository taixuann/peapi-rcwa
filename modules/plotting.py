import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib.font_manager as font_manager
from progress.bar import IncrementalBar
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from modules import material


# === Reflectivity vs k_parallel ===

def polariton_dispersion_parallel(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, t, a, E_X, config):
    
    # Determine the configuration name for the filename
    if config == "air":
        config_name = "Configuration of Air_PEAPI_Air_"
    else:
        config_name = "Configuration of SiO2_PEAPI_SiO2_" 

    # === Construct the axes with eV range, wavevector, and XY grid ===

    eV_range = np.arange(eV_min,eV_max+np.float(eV_step/2),eV_step)

    kx_All = np.arange(-k_max,k_max+k_step/2,k_step)
    X, Y = np.meshgrid(kx_All, eV_range)

    # === Read the data calculated from S4 ===

    bar = IncrementalBar('Calculating',max=1, suffix='%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
    filename = ('./data/' + config_name + "d(" + str(d) + ")_" + "t(" + str(t) + ")_" + "a(" + str(a) + ")_" + "eV_min(" + str(eV_min)+')_' + "eV_max(" + str(eV_max)+')_' + "eV_step(" + str(eV_step) + ')_' + "k_max(" + str(k_max) + ')_'+ "k_step(" + str(k_step) + ')_' + 'k-E-reflectivity.csv')
    with open(filename, 'r') as csvfileR:
        R = np.loadtxt(csvfileR, delimiter=',')
    bar.next()
    bar.finish()
    R_gr = R

    # Font setup

    font = font_manager.FontProperties(family='DejaVu Sans', style='normal', size = 25)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 20
    
    # Splitting index for upper and lower band

    split_index = np.searchsorted(eV_range, E_X)
    R_lower = R_gr[:split_index, :]
    R_upper = R_gr[split_index:, :]
    R_gr = np.vstack((R_lower, R_upper))
    R_all = np.concatenate((np.fliplr(np.delete(R_gr,0,1)),R),axis=1)
    print("r1", R_lower.shape)
    print("r2", R_upper.shape)

    # === Plotting ===
    plt.figure(figsize=(7,5), dpi=200)
    plt.pcolormesh(X, Y, R_all, cmap='hot',shading='auto',vmin=0,vmax=0.4)
    plt.axhline(E_X, color='cyan', linestyle='--', linewidth=1)
    plt.xlabel(r"kxy (2$\pi \mu^{-1}$)", fontweight='bold', labelpad=15)
    plt.ylabel("Energy (eV)", fontweight='bold', labelpad=15)
    plt.axis([-k_max, k_max, eV_min, eV_max])
    plt.xticks([t for t in np.arange(-k_max, k_max+0.1, k_max/2)])
    plt.yticks(np.arange(eV_min, eV_max, 0.2))
    plt.tight_layout()

    cbar = plt.colorbar(label='Reflectivity')
    cbar.set_ticks([t for t in np.arange(0,1.1,0.2)])
    cbar.set_ticklabels(["{:.2f}".format(t) for t in np.arange(0,1.1,0.2)])
    
    save_filename = ('./graphics/'+ config_name + "d(" + str(d) + ")_" + 
               "t(" + str(t) + ")_" + 
               "a(" + str(a) + ")_" + 
               "eV_min(" + str(eV_min)+')_' + 
               "eV_max(" + str(eV_max)+')_' + 
               "eV_step(" + str(eV_step)+')_' +
               "k_max(" + str(k_max) + ')_'+
               "k_step(" + str(k_step) +')_'+
               'k-E-reflectivity.png')
    plt.savefig(save_filename, dpi=200, bbox_inches='tight')
    plt.show()
    
    bar.next()
    bar.finish()
    return R_all

def polariton_dispersion_perp(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, t, a, E_X, config):
    """
    Extract and plot the reflectivity dip at k_parallel = 0 (normal incidence).
    Shows reflectivity vs energy and marks the polariton resonances.
    """
    # === Identify general variables ===

    # Config
    if config == "air":
        config_name = "Configuration of Air_PEAPI_Air_"
    else:
        config_name = "Configuration of SiO2_PEAPI_SiO2_"

    # Energy & k grids
    eV_range = np.arange(eV_min, eV_max + np.float(eV_step/2), eV_step)

    # Load reflectivity data
    filename = ('./data/' + config_name + "d(" + str(d) + ")_" +
                "t(" + str(t) + ")_" + "a(" + str(a) + ")_" +
                "eV_min(" + str(eV_min)+')_' +
                "eV_max(" + str(eV_max)+')_' +
                "eV_step(" + str(eV_step) + ')_' +
                "k_max(" + str(k_max) + ')_' +
                "k_step(" + str(k_step) + ')_' +
                'k-E-reflectivity.csv')
    R = np.loadtxt(filename, delimiter=',')

    # Save file path for faster saving
    save_path = ('./graphics/' + config_name + "d(" + str(d) + ")_" +
                "t(" + str(t) + ")_" + "a(" + str(a) + ")_" +
                "eV_min(" + str(eV_min)+')_' +
                "eV_max(" + str(eV_max)+')_' +
                "eV_step(" + str(eV_step) + ')_' +
                "k_max(" + str(k_max) + ')_' +
                "k_step(" + str(k_step) + ')_')

    # === Construct two regime for reflectivity ===

    split_index = np.searchsorted(eV_range, E_X)
    R_lower = R[:split_index, :]
    R_upper = R[split_index:, :]
    R = np.vstack((R_lower, R_upper))
    R_all = np.concatenate((np.fliplr(np.delete(R, 0, 1)), R), axis=1)

    # --- Extract reflectivity at k|| = 0 ---
    center_idx = R_all.shape[1] // 2
    E_axis = eV_range
    R_k0 = R_all[:, center_idx]

    # === Find reflectivity dips and peaks relative to E_X ===

    def find_and_plot_dips():
        """
        This nested function finds the polariton dips from the k=0 reflectivity
        and then generates the first plot (Reflectivity vs. Energy).
        It returns the found dip energies for the next step.
        """
        prominence_rel = 0.03
        abs_prom = max(1e-4, prominence_rel * (np.nanmax(R_k0) - np.nanmin(R_k0)))

        # --- Define energy regime ---
        below_mask = E_axis < E_X
        above_mask = E_axis > E_X

        E_below, R_below = E_axis[below_mask], R_k0[below_mask]
        E_above, R_above = E_axis[above_mask], R_k0[above_mask]

        # --- Dips finding ---
        dips_lp, dips_lp_props = find_peaks(-R_below, prominence=abs_prom, distance=3)
        dips_up, dips_up_props = find_peaks(-R_above, prominence=abs_prom, distance=3)

        # --- Dip conditions ---
        min_detuning = 0.1  # 20 meV
        valid = (E_above[dips_up] - E_X) > min_detuning
        dips_up = dips_up[valid]
        for key in dips_up_props:
            dips_up_props[key] = dips_up_props[key][valid]

        # --- Dips assigning value ---
        E_up_dips = E_above[dips_up]
        E_lp_dips = E_below[dips_lp]
        print("Detected dips (<E_X):", E_lp_dips)
        print("Detected peaks (>E_X):", E_up_dips)

        # --- Plotting the reflectivity dips ---
        plt.figure(figsize=(7,5), dpi=200)
        plt.plot(E_axis, R_k0, lw=1.5, label='Reflectivity (k∥=0)')
        plt.scatter(E_lp_dips, R_below[dips_lp], color='blue', label='Dips (<E_X)', edgecolor='k', s=60, zorder=5)
        plt.scatter(E_up_dips, R_above[dips_up], color='red', label='Peaks (>E_X)', edgecolor='k', s=60, zorder=5)

        plt.axvline(E_X, color='cyan', linestyle='--', label='Exciton energy')
        plt.xlabel("Energy (eV)", fontweight='bold', labelpad=15)
        plt.ylabel("Reflectivity (a.u.)", fontweight='bold', labelpad=15)
        plt.title("Reflectivity at k∥ = 0")
        plt.grid(alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path + 'reflectivity_dip_k0_split.png', dpi=300, bbox_inches='tight')
        plt.show()

        return E_up_dips, E_lp_dips
    
    # === Drawing the energy vs k perpendicular ===

    def plot_dispersion(E_up_dips, E_lp_dips):
        """
        This nested function takes the found dip energies, calculates the
        k_perp dispersion, and generates the second plot.
        """
        _, _, n_PEAPI_all, _ = material.PEAPI(eV_range)

        # --- Calculate k_perp for the dips ---
        idx_d = np.array([np.argmin(np.abs(eV_range - E)) for E in E_lp_dips])
        n_dips = n_PEAPI_all[idx_d]
        k_perp_lp_dips = np.sqrt(((E_lp_dips * n_dips) / 1.2398419)**2)

        idx_p = np.array([np.argmin(np.abs(eV_range - E)) for E in E_up_dips])
        n_peaks = n_PEAPI_all[idx_p]
        k_perp_up_dips = np.sqrt(((E_up_dips * n_peaks) / 1.2398419)**2)

        # --- Calculation ---
        # Uncoupled photon
        e_inf = 3.23 #from material peapi
        n_background = np.sqrt(e_inf)
        k_photon = (eV_range * n_background) / 1.2398419

        # Rabi splitting
        k_intersect = (E_X * n_background) / 1.2398419
        LP_idx = np.argmin(np.abs(k_perp_lp_dips - k_intersect))
        UP_idx = np.argmin(np.abs(k_perp_up_dips - k_intersect))
        LP_E = E_lp_dips[LP_idx]
        UP_E = E_up_dips[UP_idx]
        rabi_meV = (UP_E - LP_E) * 1000 

        # --- Plotting k_perp vs E ---
        plt.figure(figsize=(7,5), dpi=200)
        plt.plot(E_lp_dips, k_perp_lp_dips, 'bo-', label='Lower polariton dips (k⊥)')
        plt.plot(E_up_dips, k_perp_up_dips, 'ro-', label='Upper polariton peaks (k⊥)')

        plt.xlabel("Energy (eV)")
        plt.ylabel(r"$k_\perp$ (2π/µm)")
        plt.title("Polariton dispersion at k∥=0")
        plt.grid(alpha=0.3)
        plt.plot(eV_range, k_photon, 'k--', label='Uncoupled Photon')
        plt.axvline(E_X, color='cyan', linestyle='--', label='Exciton energy')
        plt.text(0.95, 0.05, f'Rabi Splitting: {rabi_meV:.2f} meV',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes,
                 color='brown', fontsize=10)
        plt.subplots_adjust(right=0.8)
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)
        plt.savefig(save_path + 'polariton_dispersion_kperp_vs_E.png', dpi=300, bbox_inches='tight')
        plt.show()
        return rabi_meV

    # === Main execution flow ===
    # Step 1: Call the first nested function to find dips and plot reflectivity.
    # It returns the dip energies needed for the next step.
    E_up_dips, E_lp_dips = find_and_plot_dips()

    # Step 2: Pass the dip energies to the second nested function to plot the dispersion.
    plot_dispersion(E_up_dips, E_lp_dips)

# === Visualization of structure and incident light ===

def visualize_structure(config, a, d, incident_angle=0):
    """
    Visualize the layer structure and light injection direction.

    Parameters
    ----------
    config : str
        "air" or "sio2" configuration.
    a : float
        Thickness of PEAPI layer (nm).
    d : float
        Period of structure (nm).
    incident_angle : float
        Incident light angle in degrees (from normal).
    """

    # Define layers based on configuration
    if config.lower() == "air":
        layers = [("Air (top)", 0, "skyblue"),
                  ("PEAPI", a, "gold"),
                  ("Air (bottom)", 0, "skyblue")]
        title = "Configuration: Air / PEAPI / Air"
    else:
        layers = [("SiO₂ (top)", 50, "lightgrey"),
                  ("PEAPI", a, "gold"),
                  ("SiO₂ (bottom)", 50, "lightgrey")]
        title = "Configuration: SiO₂ / PEAPI / SiO₂"

    fig, ax = plt.subplots(figsize=(6, 3))
    z = 0
    for name, thickness, color in layers:
        ax.fill_between([0, d], z, z + thickness, color=color, label=name)
        z += thickness

    total_height = z
    ax.set_xlim(0, d)
    ax.set_ylim(-0.3 * a, total_height + 0.3 * a)
    ax.set_xlabel("x (periodic direction, nm)")
    ax.set_ylabel("z (depth, nm)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    # Plot incident light direction arrow
    theta = np.radians(incident_angle)
    arrow_len = 0.5 * a
    ax.arrow(0.2 * d, total_height + 0.2 * a,
             arrow_len * np.sin(theta),
             -arrow_len * np.cos(theta),
             width=0.01 * d, color="red", length_includes_head=True)
    ax.text(0.25 * d, total_height + 0.25 * a, "Incident light", color="red", fontsize=10)

    save_path = f'./graphics/structure_{config}_a({a})_d({d})_angle({incident_angle}).png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
