import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your simulation results
summary_csv_Rockstar = "~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/bulk_flow_radius_series/bulk_flow_summary.csv"
df_Rockstar = pd.read_csv(summary_csv)

summary_csv_fof = "~/bulk-flow-Rockstar/Data/mdpl2_fof_mass12/mdpl2_fof_mass12.csv"
df_fof = pd.read_csv(summary_csv)

radii_Rockstar = df_Rockstar['Radius'].values  # Should be in Mpc/h
sim_bulk_flow_Rockstar = df_Rockstar['Mean_Bulk_Velocity'].values  # In km/s

radii_fof = df_Rockstar['Radius'].values  # Should be in Mpc/h
sim_bulk_flow_fof = df_Rockstar['Mean_Bulk_Velocity'].values  # In km/s

output_dir = "~/bulk-flow-Rockstar/Results"

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

# Plot
plt.figure(figsize=(8,5))
plt.plot(radii, sim_bulk_flow_Rockstar, 'o-', label='Simulation (Rockstar)')
plt.plot(radii, sim_bulk_flow_fof, 'p-', label='Simulation (Rockstar)')
plt.plot(radii, theory_bulk_flow, 'r-', label='ΛCDM Theory (Colossus)')
plt.xlabel('Radius [Mpc/h]')
plt.ylabel('RMS Bulk Flow [km/s]')
plt.title('Bulk Flow: Simulation vs. ΛCDM Theory')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Save the plot
    plt.tight_layout()
    output_path = os.path.join(output_dir, "colossus Vs Simulation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
