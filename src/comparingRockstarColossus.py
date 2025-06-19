import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load Rockstar simulation results
summary_csv_rockstar = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/bulk_flow_radius_series/bulk_flow_summary.csv")
df_rockstar = pd.read_csv(summary_csv_rockstar)
radii_rockstar = df_rockstar['Radius'].values  # Mpc/h
sim_bulk_flow_rockstar = df_rockstar['Mean_Bulk_Velocity'].values  # km/s

# Load FOF simulation results
summary_csv_fof = os.path.expanduser("~/bulk-flow-Rockstar/Data/fof_simulation/bulk_flow_radius_series/bulk_flow_summary.csv")
df_fof = pd.read_csv(summary_csv_fof)
radii_fof = df_fof['Radius'].values  # Mpc/h
sim_bulk_flow_fof = df_fof['Mean_Bulk_Velocity'].values  # km/s

# Now, get the theoretical prediction from Colossus
from colossus.cosmology import cosmology
from scipy.integrate import quad
# from scipy.special import spherical_jn

# Define your simulation's cosmology parameters
my_cosmo_params = {
    'flat': True,                # Most simulations assume flat ΛCDM
    'H0': 67.77,                 # Hubble constant (in km/s/Mpc)
    'Om0': 0.307115,               # Matter density parameter at z=0
    'Ob0': 0.048206,               # Baryon density parameter (optional but recommended)
    'sigma8': 0.8228,            # σ₈ value used in your sim
    'ns': 0.96                # Spectral index (scalar tilt)
}

# Register the custom cosmology
cosmology.addCosmology('mySimCosmo', my_cosmo_params)

# Set it as the current cosmology
cosmology.setCosmology('mySimCosmo')

# Access it for use
cosmo = cosmology.getCurrent()

# Growth rate at z=0
f = cosmo.Om0**0.55

def W(k, R):
    return 3 * (np.sin(k*R) - k*R * np.cos(k*R)) / ((k*R)**3 + 1e-10)

def integrand(k, R):
    pk = cosmo.matterPowerSpectrum(k, 0)
    return pk * W(k, R)**2

def bulk_flow_rms(R):
    k_min = 1e-4
    k_max = 10
    integral, _ = quad(lambda k: integrand(k, R), k_min, k_max, limit=200000)
    H0 = cosmo.H0  # km/s/Mpc
    sigma_v = H0**2 * f**2 /(2 * np.pi**2)  * integral
    return np.sqrt(sigma_v)

theory_bulk_flow = np.array([bulk_flow_rms(R) for R in radii])

# Plotting
plt.figure(figsize=(8,5))
plt.plot(radii_rockstar, sim_bulk_flow_rockstar, 'o-', label='Simulation (Rockstar)')
plt.plot(radii_fof, sim_bulk_flow_fof, 's-', label='Simulation (FOF)')
plt.plot(radii_rockstar, theory_bulk_flow, 'r-', label='\u039BCDM Theory (Colossus)')
plt.xlabel('Radius [Mpc/h]')
plt.ylabel('RMS Bulk Flow [km/s]')
plt.title('Bulk Flow: Simulation vs. \u039BCDM Theory')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
output_dir = os.path.expanduser('~/bulk-flow-Rockstar/Results')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'bulk_flow_comparison_plot_with_simulations.png')
plt.savefig(output_path)
print(f"Plot saved to: {output_path}")