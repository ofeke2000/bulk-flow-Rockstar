import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.integrate import simpson
import csv
import os

# Set cosmology (Planck 2018 parameters)
cosmology.setCosmology('planck18')
cosmo = cosmology.getCurrent()

# Parameters
radii = np.arange(5, 251, 5)  # Radii in h^-1Mpc
k_vals = np.logspace(-3, 1, 1000)  # k in h/Mpc
z = 0.0

# Numerical derivative of ln(D) w.r.t ln(a)
def f_z(z=0)
    a = 1.0 / (1.0 + z)

    delta = 1e-4
    D1 = cosmo.growthFactor(z)
    D2 = cosmo.growthFactor(z + delta)

    lnD1 = np.log(D1)
    lnD2 = np.log(D2)
    lna1 = np.log(a)
    lna2 = np.log(1.0 / (1.0 + z + delta))

    return (lnD2 - lnD1) / (lna2 - lna1)

# Get power spectrum at z=0
P_k = cosmo.matterPowerSpectrum(k_vals, 0)

# Define top-hat window function
def W(kR):
    return 3 * (np.sin(kR) - kR * np.cos(kR)) / (kR**3 + 1e-10)

# Compute bulk flow
bulk_flows = []
f = f_z
H0 = cosmo.H0  # Hubble constant in km/s/Mpc

for R in radii:
    kR = k_vals * R
    integrand = k_vals**2 * P_k * W(kR)**2
    integral = simpson(integrand, k_vals)
    V2 = (H0 * f)**2 / (2 * np.pi**2) * integral
    bulk_flows.append(np.sqrt(V2))

# Save to CSV
output_csv = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/colossus_bulk_flow_radius_series.csv")
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Radius_Mpc", "Theoretical_Bulk_Flow_km_s"])
    writer.writerows(zip(radii, bulk_flows))

print(f"Theoretical bulk flow saved to: {output_csv}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(radii, bulk_flows, label='ΛCDM (Colossus)', color='green')
plt.xlabel("Radius [Mpc]")
plt.ylabel("Bulk Flow [km/s]")
plt.title("Theoretical Bulk Flow (ΛCDM, z=0, Planck18)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
