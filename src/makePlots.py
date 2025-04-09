import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
results_file = os.path.expanduser("~/bulk-flow-Rockstar/Results/bulk_flow.csv")
output_folder = os.path.expanduser("~/bulk-flow-Rockstar/Results")
plot_mean_file = os.path.join(output_folder, 'mean_velocities_plot.png')
plot_variance_file = os.path.join(output_folder, 'velocity_variances_plot.png')

# Load precomputed bulk flow data
results = pd.read_csv(results_file)

# Extract data for plotting
radii = results['Radius']
vx_means = results['Vx_Mean']
vy_means = results['Vy_Mean']
vz_means = results['Vz_Mean']
vx_variances = results['Vx_Variance']
vy_variances = results['Vy_Variance']
vz_variances = results['Vz_Variance']

# Plot mean velocities
plt.figure(figsize=(10, 6))
plt.plot(radii, vx_means, label='Vx Mean', marker='o')
plt.plot(radii, vy_means, label='Vy Mean', marker='o')
plt.plot(radii, vz_means, label='Vz Mean', marker='o')
plt.xlabel('Radius [h-1.mpc]')
plt.ylabel('Mean Velocity [km.s-1]')
plt.title('Mean Velocities vs Radius')
plt.legend()
plt.grid(True)
plt.savefig(plot_mean_file)
plt.close()
print(f"Mean velocities plot saved to: {plot_mean_file}")

# Plot velocity variances
plt.figure(figsize=(10, 6))
plt.plot(radii, vx_variances, label='Vx Variance', marker='o')
plt.plot(radii, vy_variances, label='Vy Variance', marker='o')
plt.plot(radii, vz_variances, label='Vz Variance', marker='o')
plt.xlabel('Radius [h-1.mpc]')
plt.ylabel('Velocity Variance [km2.s-2]')
plt.title('Velocity Variances vs Radius')
plt.legend()
plt.grid(True)
plt.savefig(plot_variance_file)
plt.close()
print(f"Velocity variances plot saved to: {plot_variance_file}")
