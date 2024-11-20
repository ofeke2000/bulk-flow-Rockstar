import pandas as pd
import numpy as np
import os

# Define input and output folders
input_folder = os.path.expanduser("~/bulk-flow-Rockstar/Data")  # Input folder
output_folder = os.path.expanduser("~/bulk-flow-Rockstar/Results")  # Output folder

# Define input file paths
r_sorted_file = os.path.join(input_folder, 'rSorted.csv')  # Sorted R file
vx_sorted_file = os.path.join(input_folder, 'vx_sorted.csv')
vy_sorted_file = os.path.join(input_folder, 'vy_sorted.csv')
vz_sorted_file = os.path.join(input_folder, 'vz_sorted.csv')

# Define output file path
bulk_flow_file = os.path.join(output_folder, 'bulk_flow.csv')

# Parameters
num_bins = 10  # Number of radial bins

# Load data
r_sorted = pd.read_csv(r_sorted_file)
vx_sorted = pd.read_csv(vx_sorted_file)
vy_sorted = pd.read_csv(vy_sorted_file)
vz_sorted = pd.read_csv(vz_sorted_file)

# Ensure all vectors have the same length
assert len(r_sorted) == len(vx_sorted) == len(vy_sorted) == len(vz_sorted), \
    "Mismatch in lengths of R and velocity files."

# Extract R values
r_values = r_sorted['R'].values

# Determine radial bins (r_start is always 0)
r_min = 0
r_max = r_values.max()
bin_edges = np.linspace(r_min, r_max, num_bins + 1)

# Initialize lists to store results
end_radii = []  # Bin end radii
vx_means, vy_means, vz_means = [], [], []
vx_variances, vy_variances, vz_variances = [], [], []

# Calculate mean and variance for each bin
for i in range(1, num_bins + 1):  # Start at the second edge to include r_start = 0
    r_end = bin_edges[i]

    # Get indices of points within the current radial bin (from 0 to r_end)
    mask = (r_values >= r_min) & (r_values < r_end)

    if mask.sum() == 0:  # Skip empty bins
        continue

    # Extract velocities within the bin
    vx_bin = vx_sorted.iloc[mask].values.flatten()
    vy_bin = vy_sorted.iloc[mask].values.flatten()
    vz_bin = vz_sorted.iloc[mask].values.flatten()

    # Calculate mean and variance for each direction
    vx_means.append(np.mean(vx_bin))
    vy_means.append(np.mean(vy_bin))
    vz_means.append(np.mean(vz_bin))

    vx_variances.append(np.var(vx_bin))
    vy_variances.append(np.var(vy_bin))
    vz_variances.append(np.var(vz_bin))

    # Append the end radius of the bin
    end_radii.append(r_end)

# Save results to a DataFrame
results = pd.DataFrame({
    'Radius': end_radii,
    'Vx_Mean': vx_means,
    'Vy_Mean': vy_means,
    'Vz_Mean': vz_means,
    'Vx_Variance': vx_variances,
    'Vy_Variance': vy_variances,
    'Vz_Variance': vz_variances
})

# Save to CSV
results.to_csv(bulk_flow_file, index=False)
print(f"Bulk flow results saved to: {bulk_flow_file}")
