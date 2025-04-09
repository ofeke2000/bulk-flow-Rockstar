import pandas as pd
import numpy as np
import os

# Define paths
output_folder = os.path.expanduser("~/bulk-flow-Rockstar/Data")
x_file = os.path.join(output_folder, 'x.csv')
y_file = os.path.join(output_folder, 'y.csv')
z_file = os.path.join(output_folder, 'z.csv')
r_file = os.path.join(output_folder, 'r.csv')  # Output file for r values

# Read the x, y, and z velocity files
x_data = pd.read_csv(x_file)
y_data = pd.read_csv(y_file)
z_data = pd.read_csv(z_file)

# Check if the columns have consistent names and extract their data
x_column = x_data.columns[0]
y_column = y_data.columns[0]
z_column = z_data.columns[0]

# Ensure all columns have the same length
if len(x_data) != len(y_data) or len(y_data) != len(z_data):
    raise ValueError("The x, y, and z files must have the same number of rows.")

# Calculate r = sqrt(x^2 + y^2 + z^2)
r_values = np.sqrt(x_data[x_column]**2 + y_data[y_column]**2 + z_data[z_column]**2)

# Create a DataFrame for the results
r_df = pd.DataFrame({
    'Index': range(1, len(r_values) + 1),  # Running numbers starting from 1
    'R': r_values
})

# Save the result to the output CSV file
r_df.to_csv(r_file, index=False)
print(f"File with r values and running numbers saved at: {r_file}")
