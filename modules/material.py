import math
import matplotlib.pyplot as plt
import os

def PEAPI(eV):
    p = math.pi
    e_inf = 3.23 

    f1 = 0.134045 
    E1 = 2.3478 
    y1 = 0.008 
    e1 = f1*E1**2/(E1**2 - eV**2 + 1j*y1*eV) #Dielectric function of lowest excitonic transition

    f2 = 0
    E2 = 2.3468 
    y2 = 0.45
    e2 = f2*E2**2/(E2**2 - eV**2 + 1j*y2*eV) #Dielectric function of higher-energy transition
    
    f3 = 0
    E3 = 5.1
    y3 = 3.9
    e3 = f3*E3**2/(E3**2 - eV**2 + 1j*y3*eV) #Dielectrci function of another resonance   

    e = e_inf + e1 + e2 + e3
    e_r_PEAPI = e.real
    e_i_PEAPI = abs(e.imag)
    n_PEAPI = (e**0.5).real
    k = abs((e**0.5).imag)

    return (e_r_PEAPI, e_i_PEAPI, n_PEAPI, k)

def SiO2(eV):
    p = math.pi
    e_inf = 1
	
    f = 1.12
    E0 = 12		#eV 
    h = 4.135667e-15	# Planck constan - eV.s
	
    e = e_inf + (f*E0**2)/(E0**2-eV**2)
    e_r_SiO2 = e.real
    e_i_SiO2 = e.imag
    n = (e**.5).real
    k = (e**.5).imag
	
    return (e_r_SiO2, e_i_SiO2, n, k)

def generate_PEAPI_equation_image(filename="PEAPI_equation.png"):
    """
    Generates an image of the PEAPI dielectric function equation with parameter values.
    """
    # Parameter values
    e_inf = 3.23
    f1 = 0.134045
    E1 = 2.3478
    y1 = 0.008

    # LaTeX equation string
    equation = r"$\epsilon(E) = \epsilon_\infty + \frac{f E_X^2}{E_X^2 - E^2 + i \gamma_X E_X}$"

    # Parameter values string
    params = (
        r"$\epsilon_\infty = $" + f"{e_inf:.2f}\n"
        r"$f = $" + f"{f1:.4f}\n"
        r"$E_X = $" + f"{E1:.4f} eV\n"
        r"$\gamma = $" + f"{y1:.4f} eV"
    )

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")  # Turn off axes

    # Add equation and parameter values as text
    ax.text(0.5, 0.7, equation, fontsize=24, ha="center")
    ax.text(0.5, 0.2, params, fontsize=14, ha="center")

    # Define the output directory and create it if it doesn't exist
    output_dir = './graphics/equations/'
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure with a transparent background
    plt.savefig(os.path.join(output_dir, filename), transparent=True, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # Close the figure to free memory

if __name__ == "__main__":
    print("--- Generating PEAPI equation image ---")
    generate_PEAPI_equation_image()  # Call the function to generate the image