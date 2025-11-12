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
    
    plt.pcolormesh(X, Y, R_all, cmap='hot',shading='auto',vmin=0,vmax=1)
    plt.axhline(E_X, color='cyan', linestyle='--', linewidth=1)
    plt.xlabel(r"kxy (2$\pi \mu^{-1}$)", fontweight='bold', labelpad=15)
    plt.ylabel("Energy (eV)", fontweight='bold', labelpad=15)
    plt.axis([-k_max, k_max, eV_min, eV_max])
    plt.xticks([t for t in np.arange(-k_max, k_max+0.1, k_max/2)])
    plt.yticks(np.arange(eV_min, eV_max, 0.2))

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
    plt.savefig(save_filename, dpi=600)
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
    kx_All = np.arange(-k_max, k_max + k_step/2, k_step)

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

    # --- Find reflectivity dips ---
    prominence_rel = 0.02
    abs_prom = max(1e-4, prominence_rel * (np.nanmax(R_k0) - np.nanmin(R_k0)))
    peaks_k0, props_k0 = find_peaks(-R_k0, prominence=abs_prom, distance=3)

    # Sort dips by prominence (strongest first)
    if 'prominences' in props_k0:
        order = np.argsort(props_k0['prominences'])[::-1]
    else:
        order = np.argsort(-R_k0[peaks_k0])
    peaks_k0 = peaks_k0[order]

    # --- Compute k⊥ at k|| = 0 ---
    hc = 1.23984
    k_total = 2 * np.pi * n_medium * eV_range / hc
    k_perp = k_total  # at k∥ = 0

    # --- Plot ---
    plt.figure(figsize=(7,5), dpi=200)
    plt.plot(E_axis, R_k0, lw=1.5, label='Reflectivity (k∥=0)')
    colors = plt.cm.tab10(np.linspace(0,1,max(1,len(peaks_k0))))
    for idx, p in enumerate(peaks_k0):
        E_p = E_axis[p]
        R_p = R_k0[p]
        plt.scatter(E_p, R_p, color=colors[idx % len(colors)], s=60, edgecolor='k', zorder=5)
        plt.text(E_p + 0.01, R_p + 0.02, f"{E_p:.3f} eV", fontsize=8)

    plt.xlabel("Energy (eV)")
    plt.ylabel("Reflectivity (a.u.)")
    plt.title("Reflectivity at k∥ = 0 with detected dips")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./graphics/reflectivity_dip_k0.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Detected dip energies (polariton modes):", E_axis[peaks_k0])


def polariton_dips_connect(eV_min, eV_max, eV_step, k_min, k_max, k_step, d, t, a, E_X, config):
    """
    Extract reflectivity dips from S4 output, plot E (x-axis) vs k_perp (y-axis)
    with LPB and UPB connections, and exciton energy line.
    """
    from scipy.signal import find_peaks

    # --- Config setup ---
    if config == "air":
        config_name = "Configuration of Air_PEAPI_Air_"
        n_medium = 1.0
    else:
        config_name = "Configuration of SiO2_PEAPI_SiO2_"
        _, _, n_SiO2, _ = material.SiO2(eV_min)
        n_medium = np.real(n_SiO2)

    eV_range = np.arange(eV_min, eV_max + np.float(eV_step/2), eV_step)
    kx_All = np.arange(-k_max, k_max + k_step/2, k_step)

    # --- Load reflectivity ---
    filename = ('./data/' + config_name + "d(" + str(d) + ")_" +
                "t(" + str(t) + ")_" + "a(" + str(a) + ")_" +
                "eV_min(" + str(eV_min)+')_' +
                "eV_max(" + str(eV_max)+')_' +
                "eV_step(" + str(eV_step) + ')_' +
                "k_max(" + str(k_max) + ')_'+
                "k_step(" + str(k_step) + ')_' +
                'k-E-reflectivity.csv')
    R = np.loadtxt(filename, delimiter=',')

    LPB, UPB, kx_valid = [], [], []
    for i in range(R.shape[1]):
        R_col = R[:, i]
        peaks, props = find_peaks(-R_col, prominence=0.01)
        if len(peaks) == 0:
            continue
        energies = eV_range[peaks]
        energies.sort()
        LPB.append(energies[0])
        UPB.append(energies[-1] if len(energies) > 1 else np.nan)
        kx_valid.append(kx_All[i])

    LPB = np.array(LPB)
    UPB = np.array(UPB)
    kx_valid = np.array(kx_valid)

    # --- Convert to k_perp ---
    hc = 1.23984
    k_perp_L = np.sqrt(np.maximum((2*np.pi*n_medium*LPB/hc)**2 - kx_valid**2, 0))
    k_perp_U = np.sqrt(np.maximum((2*np.pi*n_medium*UPB/hc)**2 - kx_valid**2, 0))

    # --- Plot ---
    plt.figure(figsize=(6,5), dpi=200)
    plt.plot(LPB, k_perp_L, 'r-', lw=2, label='Lower polariton')
    plt.plot(UPB, k_perp_U, 'b-', lw=2, label='Upper polariton')
    plt.axvline(E_X, color='k', linestyle='--', label='Exciton energy')
    plt.xlabel("Energy (eV)")
    plt.ylabel(r"$k_\perp$ ($\mu m^{-1}$)")
    plt.title("Polariton dispersion from reflectivity dips")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./graphics/polariton_dispersion_lines.png', dpi=400)
    plt.show()


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

