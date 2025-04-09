import pandas as pd
import os

# Define paths
output_folder = os.path.expanduser("~/bulk-flow-Rockstar/Data")
r_sorted_file = os.path.join(output_folder, 'rSorted.csv')  # Sorted R file
vx_file = os.path.join(output_folder, 'vx.csv')
vy_file = os.path.join(output_folder, 'vy.csv')
vz_file = os.path.join(output_folder, 'vz.csv')

# Output files
vx_sorted_file = os.path.join(output_folder, 'vx_sorted.csv')
vy_sorted_file = os.path.join(output_folder, 'vy_sorted.csv')
vz_sorted_file = os.path.join(output_folder, 'vz_sorted.csv')

# Read the sorted R file to get the sorting order (Index column)
r_sorted = pd.read_csv(r_sorted_file)
sorted_indices = r_sorted['Index'] - 1  # Convert 1-based to 0-based index

# Read the original vector files
vx_data = pd.read_csv(vx_file)
vy_data = pd.read_csv(vy_file)
vz_data = pd.read_csv(vz_file)

# Apply the sorted order to the vector files
vx_sorted = vx_data.iloc[sorted_indices].reset_index(drop=True)
vy_sorted = vy_data.iloc[sorted_indices].reset_index(drop=True)
vz_sorted = vz_data.iloc[sorted_indices].reset_index(drop=True)

# Save the sorted vectors to new files
vx_sorted.to_csv(vx_sorted_file, index=False)
vy_sorted.to_csv(vy_sorted_file, index=False)
vz_sorted.to_csv(vz_sorted_file, index=False)

print(f"Reordered vx saved to: {vx_sorted_file}")
print(f"Reordered vy saved to: {vy_sorted_file}")
print(f"Reordered vz saved to: {vz_sorted_file}")
