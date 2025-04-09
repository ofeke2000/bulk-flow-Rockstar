import pandas as pd
import os

# Define paths
output_folder = os.path.expanduser("~/bulk-flow-Rockstar/Data")
r_file = os.path.join(output_folder, 'r.csv')  # Input file
sorted_file = os.path.join(output_folder, 'rSorted.csv')  # Output file

# Read the R_values.csv file
r_data = pd.read_csv(r_file)

# Sort the DataFrame by the R column
r_sorted = r_data.sort_values(by='R', ascending=True)

# Save the sorted DataFrame to a new file
r_sorted.to_csv(sorted_file, index=False)
print(f"Sorted file saved at: {sorted_file}")
