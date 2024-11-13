import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths
output_folder = os.path.expanduser("~/bulk-flow-Rockstar/Results")

# Load data from each direction
x_data = pd.read_csv(os.path.join(output_folder, 'X_velocity.csv'))
y_data = pd.read_csv(os.path.join(output_folder, 'Y_velocity.csv'))
z_data = pd.read_csv(os.path.join(output_folder, 'Z_velocity.csv'))

# Define the radii of the spheres (in arbitrary units, change as required)
radii = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example distances

# Initialize lists to store average velocities
avg_velocity_x = []
avg_velocity_y = []
avg_velocity_z = []

# Loop over each radius to calculate the average velocities
for radius in radii:
    # Select data within the current radius
    # Note: This example assumes a hypothetical way of filtering the data points within each sphere,
    # as real implementation requires spatial information of each point to calculate distances.

    # Compute the average velocity for each direction within the current radius
    avg_velocity_x.append(x_data['X_velocity'].sample(n=100).mean())  # Adjust n as necessary
    avg_velocity_y.append(y_data['Y_velocity'].sample(n=100).mean())
    avg_velocity_z.append(z_data['Z_velocity'].sample(n=100).mean())

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(radii, avg_velocity_x, label='X Direction', marker='o')
plt.plot(radii, avg_velocity_y, label='Y Direction', marker='o')
plt.plot(radii, avg_velocity_z, label='Z Direction', marker='o')

plt.xlabel('Radius')
plt.ylabel('Average Velocity')
plt.title('Bulk Flow Velocity vs. Radius')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, 'bulk_flow_velocity_vs_radius.png'), dpi=300)
plt.show()
