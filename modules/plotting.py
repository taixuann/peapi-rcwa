import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib.font_manager as font_manager
from progress.bar import IncrementalBar
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from modules import material


# === Reflectivity vs k_parallel ===

def reflectivity_dispersion(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, t, a, E_X, config):
    
    # Determine the configuration name for the filename
    if config == "air":
        config_name = "Configuration of Air_PEAPI_Air_"
    else:
        config_name = "Configuration of SiO2_PEAPI_SiO2_" 

    # === Construct the axes with eV range, wavevector, and XY grid ===

    eV_range = np.arange(eV_min,eV_max+np.float(eV_step/2),eV_step)
    ev_min = np.arange(eV_min,E_X+np.float(eV_step/2),eV_step)
    ev_max = np.arange(E_X,eV_max+np.float(eV_step/2),eV_step)

    kx_All = np.arange(-k_max,k_max+k_step/2,k_step)
    X, Y = np.meshgrid(kx_All, eV_range)

    # === Read the data calculated from S4 ===

    bar = IncrementalBar('Calculating',max=1, suffix='%(percent)d%% Elapsed %(elapsed_td)s - ETA %(eta_td)s')
    filename = ('./data/' + config_name + "d(" + str(d) + ")_" + "t(" + str(t) + ")_" + "a(" + str(a) + ")_" + "eV_min(" + str(eV_min)+')_' + "eV_max(" + str(eV_max)+')_' + "eV_step(" + str(eV_step) + ')_' + "k_max(" + str(k_max) + ')_'+ "k_step(" + str(k_step) + ')_' + 'k-E-reflectivity.csv')
    with open(filename, 'r') as csvfileR:
        R = np.loadtxt(csvfileR, delimiter=',')
    bar.next()
    bar.finish()

    # Font setup

    font = font_manager.FontProperties(family='DejaVu Sans', style='normal', size = 25)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 20

    # === Configuration ===

    # Laplacian for symmetry

    apply_laplacian = False
    if apply_laplacian:
        R_gr = cv.Laplacian(R, cv.CV_64F, ksize=3)
        R_gr = R_gr / np.abs(R_gr).max()
    else:
        R_gr = R

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

def reflectivity_dip_k0(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, t, a, E_X, config):
    """
    Extract and plot the reflectivity dip at k_parallel = 0 (normal incidence).
    Shows reflectivity vs energy and marks the polariton resonances.
    """
    # Identify config
    if config == "air":
        config_name = "Configuration of Air_PEAPI_Air_"
        n_medium = 1.0
    else:
        config_name = "Configuration of SiO2_PEAPI_SiO2_"
        _, _, n_SiO2, _ = material.SiO2(eV_min)
        n_medium = n_SiO2

    # Energy & k grids
    eV_range = np.arange(eV_min, eV_max + np.float(eV_step/2), eV_step)
    _, _, n_PEAPI_all, _ = material.PEAPI(eV_range)

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

    # Apply optional Laplacian filter
    apply_laplacian = False
    if apply_laplacian:
        R_gr = cv.Laplacian(R, cv.CV_64F, ksize=3)
        R_gr = R_gr / np.abs(R_gr).max()
    else:
        R_gr = R

    # Combine for symmetry (same logic as reflectivity_dispersion)
    split_index = np.searchsorted(eV_range, E_X)
    R_lower = R_gr[:split_index, :]
    R_upper = R_gr[split_index:, :]
    R_gr = np.vstack((R_lower, R_upper))
    R_all = np.concatenate((np.fliplr(np.delete(R_gr, 0, 1)), R), axis=1)

    # --- Extract reflectivity at k|| = 0 ---
    center_idx = R_all.shape[1] // 2
    E_axis = eV_range
    R_k0 = R_all[:, center_idx]

    # --- Find reflectivity dips and peaks relative to E_X ---
    prominence_rel = 0.02
    abs_prom = max(1e-4, prominence_rel * (np.nanmax(R_k0) - np.nanmin(R_k0)))

    # Split the energy and reflectivity at E_X
    below_mask = E_axis < E_X
    above_mask = E_axis > E_X

    E_below, R_below = E_axis[below_mask], R_k0[below_mask]
    E_above, R_above = E_axis[above_mask], R_k0[above_mask]

    # Dips below E_X (minima)
    dips_idx, dips_props = find_peaks(-R_below, prominence=abs_prom, distance=3)
    # Peaks above E_X (maxima)
    peaks_idx, peaks_props = find_peaks(R_above, prominence=abs_prom, distance=3)
    valid = R_above[peaks_idx] < 0.5
    peaks_idx = peaks_idx[valid]
    for key in peaks_props:
        peaks_props[key] = peaks_props[key][valid]

    # Then get the energies of those peaks
    E_peaks = E_above[peaks_idx]
    E_dips = E_below[dips_idx]
    E_peaks = E_above[peaks_idx]

    # --- Plot ---
    plt.figure(figsize=(7,5), dpi=200)
    plt.plot(E_axis, R_k0, lw=1.5, label='Reflectivity (k∥=0)')

    # Mark dips and peaks
    plt.scatter(E_dips, R_below[dips_idx], color='blue', label='Dips (<E_X)', edgecolor='k', s=60, zorder=5)
    plt.scatter(E_peaks, R_above[peaks_idx], color='red', label='Peaks (>E_X)', edgecolor='k', s=60, zorder=5)

    plt.axvline(E_X, color='cyan', linestyle='--', label='Exciton energy')
    plt.xlabel("Energy (eV)", fontweight='bold', labelpad=15)
    plt.ylabel("Reflectivity (a.u.)", fontweight = 'bold', labelpad=15)
    plt.title("Reflectivity at k∥ = 0 (dips < E_X, peaks > E_X)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./graphics/' + config_name + "d(" + str(d) + ")_" +
                "t(" + str(t) + ")_" + "a(" + str(a) + ")_" +
                "eV_min(" + str(eV_min)+')_' +
                "eV_max(" + str(eV_max)+')_' +
                "eV_step(" + str(eV_step) + ')_' +
                "k_max(" + str(k_max) + ')_' +
                "k_step(" + str(k_step) + ')_' + 'reflectivity_dip_k0_split.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Detected dips (<E_X):", E_dips)
    print("Detected peaks (>E_X):", E_peaks)
    # === Plotting the k_perp vs energy ===

    idx_d = np.array([np.argmin(np.abs(eV_range - E)) for E in E_dips])
    n_dips  = n_PEAPI_all[idx_d]
    k_perp_dips = np.sqrt(((E_dips * n_dips) / 1.2398419)**2)

    # For peaks
    idx_p = np.array([np.argmin(np.abs(eV_range - E)) for E in E_peaks])
    n_peaks = n_PEAPI_all[idx_p]
    k_perp_peaks = np.sqrt(((E_peaks * n_peaks) / 1.2398419)**2)
    plt.figure(figsize=(7,5), dpi=200)

    plt.plot(E_dips, k_perp_dips, 'bo-', label='Lower polariton dips (k⊥)')
    plt.plot(E_peaks, k_perp_peaks, 'ro-', label='Upper polariton peaks (k⊥)')

    plt.xlabel("Energy (eV)")
    plt.ylabel(r"$k_\perp$ (2π/µm)")
    plt.title("Polariton dispersion at k∥=0")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./graphics/' + config_name + "d(" + str(d) + ")_" +
                "t(" + str(t) + ")_" + "a(" + str(a) + ")_" +
                "eV_min(" + str(eV_min)+')_' +
                "eV_max(" + str(eV_max)+')_' +
                "eV_step(" + str(eV_step) + ')_' +
                "k_max(" + str(k_max) + ')_' +
                "k_step(" + str(k_step) + ')_' + 'polariton_dispersion_kperp_vs_E.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_rabi_splitting(lower_polariton_dips, upper_polariton_peaks, E_X):
    """
    Takes the dips and peaks from the k=0 spectrum and calculates
    the Rabi splitting.
    """
    print("\n--- Inside calculate_rabi_splitting ---")
    print(f"Received dips: {lower_polariton_dips}")
    print(f"Received peaks: {upper_polariton_peaks}")
    
    # Example calculation:
    # Find the lower polariton (LP) and upper polariton (UP)
    # LP is the dip closest to E_X
    # UP is the peak closest to E_X
    
    rabi_splitting_mev = None
    
    if len(lower_polariton_dips) > 0 and len(upper_polariton_peaks) > 0:
        # np.max(dips) gives the dip with the highest energy (closest to E_X)
        lp_energy = np.max(lower_polariton_dips)
        
        # np.min(peaks) gives the peak with the lowest energy (closest to E_X)
        up_energy = np.min(upper_polariton_peaks)
        
        # Calculate splitting and convert from eV to meV
        splitting_eV = up_energy - lp_energy
        rabi_splitting_mev = splitting_eV * 1000
        
        print(f"Lower Polariton (LP) at: {lp_energy:.4f} eV")
        print(f"Upper Polariton (UP) at: {up_energy:.4f} eV")
        print(f"Calculated Rabi Splitting: {rabi_splitting_mev:.2f} meV")
    
    else:
        print("Could not find both a dip and a peak to calculate splitting.")
        
    # You can also return the calculated value
    return rabi_splitting_mev


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

