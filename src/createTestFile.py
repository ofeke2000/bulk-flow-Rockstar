import pandas as pd
import os

input_file = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid_-1/mdpl2_rockstar_pid_-1.csv")

# Read only the first 1000 rows from the CSV
df_subset = pd.read_csv(input_file, nrows=1000)

# Export to a new CSV file
df_subset.to_csv('test_database.csv', index=False)
