import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import camb
# from classy import Class
from scipy.integrate import quad
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology

# Load Rockstar simulation results
summary_csv_rockstar = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/bulk_flow_radius_series/bulk_flow_summary.csv")
df_rockstar = pd.read_csv(summary_csv_rockstar)
radii_rockstar = df_rockstar['Radius'].values  # Mpc/h
sim_bulk_flow_rockstar = df_rockstar['Mean_Bulk_Velocity_Squared'].values  # km/s

# # Load FOF simulation results
# summary_csv_fof = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_fof_mass12/bulk_flow_radius_series/bulk_flow_summary.csv")
# df_fof = pd.read_csv(summary_csv_fof)
# radii_fof = df_fof['Radius'].values  # Mpc/h
# sim_bulk_flow_fof = df_fof['Mean_Bulk_Velocity'].values  # km/s

# Define your simulation's cosmology parameters
my_cosmo_params = {
    'flat': True,
    'H0': 67.77,
    'Om0': 0.307115,
    'Ode0': 0.692885,
    'Ob0': 0.048206,
    'sigma8': 0.8228,
    'ns': 0.96
}

# Register the custom cosmology in Colossus
cosmology.addCosmology('mySimCosmo', my_cosmo_params)
cosmology.setCosmology('mySimCosmo')
cosmo = cosmology.getCurrent()

# Growth rate at z=0
f = cosmo.Om0**0.55

# Define top-hat window function
def W(k, R):
    return 3 * (np.sin(k*R) - k*R * np.cos(k*R)) / ((k*R)**3 + 1e-10)

# ======================== THEORETICAL CALCULATIONS ========================
# Define k values for integration
k_vals = np.logspace(-4, 1, 500)  # k in h/Mpc

# ----- Colossus calculation -----
def integrand_colossus(k, R):
    pk = cosmo.matterPowerSpectrum(k, 0)
    return pk * W(k, R)**2

def bulk_flow_rms_colossus(R):
    k_min = 1e-4
    k_max = 10
    integral, _ = quad(lambda k: integrand_colossus(k, R), k_min, k_max, limit=200000)
    H0 = cosmo.H0
    sigma_v = H0**2 * f**2 /(2 * np.pi**2) * integral
    return sigma_v

theory_colossus = np.array([bulk_flow_rms_colossus(R) for R in radii_rockstar])

# # ----- CAMB calculation -----
# # Set up CAMB parameters
# pars = camb.CAMBparams()
# pars.set_cosmology(
#     H0=my_cosmo_params['H0'],
#     ombh2=my_cosmo_params['Ob0']*(my_cosmo_params['H0']/100)**2,
#     omch2=(my_cosmo_params['Om0']-my_cosmo_params['Ob0'])*(my_cosmo_params['H0']/100)**2
# )
# pars.InitPower.set_params(ns=my_cosmo_params['ns'])
# pars.set_matter_power(redshifts=[0], kmax=max(k_vals))
# results = camb.get_results(pars)
# kh, z, pk_camb = results.get_matter_power_spectrum(minkh=min(k_vals), maxkh=max(k_vals), npoints=len(k_vals))
# interp_pk_camb = interp1d(kh, pk_camb, bounds_error=False, fill_value=0)
#
# def integrand_camb(k, R):
#     pk = interp_pk_camb(k)
#     return pk * W(k, R)**2
#
# def bulk_flow_rms_camb(R):
#     k_min = 1e-4
#     k_max = 10
#     integral, _ = quad(lambda k: integrand_camb(k, R), k_min, k_max, limit=200000)
#     H0 = my_cosmo_params['H0']
#     sigma_v = H0**2 * f**2 /(2 * np.pi**2) * integral
#     return np.sqrt(sigma_v)
#
# theory_camb = np.array([bulk_flow_rms_camb(R) for R in radii_rockstar])

# # ----- CLASS calculation -----
# params_class = {
#     'h': my_cosmo_params['H0']/100,
#     'omega_b': my_cosmo_params['Ob0']*(my_cosmo_params['H0']/100)**2,
#     'omega_cdm': (my_cosmo_params['Om0']-my_cosmo_params['Ob0'])*(my_cosmo_params['H0']/100)**2,
#     'A_s': 2.1e-9,  # Adjusted to match sigma8
#     'n_s': my_cosmo_params['ns'],
#     'output': 'mPk',
#     'P_k_max_h/Mpc': max(k_vals),
# }
# cosmo_class = Class()
# cosmo_class.set(params_class)
# cosmo_class.compute()
# pk_class = np.array([cosmo_class.pk(k, 0) for k in k_vals])
# interp_pk_class = interp1d(k_vals, pk_class, bounds_error=False, fill_value=0)
#
# def integrand_class(k, R):
#     pk = interp_pk_class(k)
#     return pk * W(k, R)**2
#
# def bulk_flow_rms_class(R):
#     k_min = 1e-4
#     k_max = 10
#     integral, _ = quad(lambda k: integrand_class(k, R), k_min, k_max, limit=200000)
#     H0 = my_cosmo_params['H0']
#     sigma_v = H0**2 * f**2 /(2 * np.pi**2) * integral
#     return np.sqrt(sigma_v)
#
# theory_class = np.array([bulk_flow_rms_class(R) for R in radii_rockstar])

# ======================== PLOTTING ========================
plt.figure(figsize=(10, 6))
plt.plot(radii_rockstar, sim_bulk_flow_rockstar, 'o-', label='Simulation (Rockstar)')
# plt.plot(radii_fof, sim_bulk_flow_fof, 's-', label='Simulation (FOF)')
plt.plot(radii_rockstar, theory_colossus, 'r-', label='ΛCDM Theory (Colossus)')
# plt.plot(radii_rockstar, theory_camb, 'g--', label='ΛCDM Theory (CAMB)')
# plt.plot(radii_rockstar, theory_class, 'b-.', label='ΛCDM Theory (CLASS)')
plt.xlabel('Radius [Mpc/h]')
plt.ylabel('RMS Bulk Flow [km/s]^2')
plt.title('Bulk Flow Comparison: Simulation vs. ΛCDM Theories')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
output_dir = os.path.expanduser('~/bulk-flow-Rockstar/Results')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'bulk_flow_comparison_with_multiple_theories.png')
plt.savefig(output_path, dpi=300)
print(f"Plot saved to: {output_path}")

# Print sample values for verification
print("\nSample theoretical values (R=50 Mpc/h):")
print(f"Colossus: {theory_colossus[9]:.1f} km/s")
# print(f"CAMB: {theory_camb[9]:.1f} km/s")
# print(f"CLASS: {theory_class[9]:.1f} km/s")
