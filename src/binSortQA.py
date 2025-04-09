import h5py
import os

# Path to the HDF5 file
hdf5_file = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid_-1/grid_sorted_hdf5/x3y7z8_rockstar_125.h5")  # Replace with the path to the file you're inspecting

# Define the column names manually
columns = ["mvir", "rvir", "rs", "rockstarid", "pid", "x", "y", "z", "vx", "vy", "vz"]

# Load the HDF5 file
with h5py.File(hdf5_file, "r") as f:
    # Access the dataset
    data = f["data"][:]  # Read the data into memory (make sure it fits in RAM)

    # Let's pick the first row to show the full rockstarid
    first_entry = data[0]  # Just an example, you can change the index to print other rows

    # Get the full rockstarid (no approximation)
    rockstarid = first_entry[3]  # Assuming rockstarid is the 4th entry in the data
    print("Full rockstarid:", int(rockstarid))  # Ensure it's an integer, no rounding

    # Show the entire first row for context
    print("First entry:", first_entry)