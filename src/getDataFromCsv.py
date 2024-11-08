import pandas as pd
import os

# Define paths
input_file = os.path.expanduser("~/bulk-flow-Rockstar/Data/Rockstar_snapnum125.csv")
output_folder = os.path.expanduser("~/bulk-flow-Rockstar/Data")
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file with just the header to get the column names
header_df = pd.read_csv(input_file, nrows=0)
columns = header_df.columns.tolist()  # Get the list of column names

# Iterate over each column, read it one at a time, and save it
for column in columns:
    # Read just the current column
    column_data = pd.read_csv(input_file, usecols=[column])  # Read only the current column

    # Get the first entry of the column for the filename
    first_entry = column_data.iloc[0, 0]  # Get the first entry of the column

    # Sanitize the first entry to create a valid filename
    filename_safe = str(first_entry).replace('/', '_').replace('\\', '_')  # Example of sanitizing
    output_file = os.path.join(output_folder, f"{filename_safe}.csv")  # Create the output filename

    # Save the current column to CSV without the index
    column_data.to_csv(output_file, index=False)

    print(f"Saved column '{column}' as '{output_file}'.")

print("All columns have been saved as individual CSV files.")
