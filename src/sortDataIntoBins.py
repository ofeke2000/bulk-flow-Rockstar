import time
import os
import h5py
import numpy as np
import csv

input_file = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid_-1/mdpl2_rockstar_pid_-1.csv")
output_dir = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid_-1/grid_sorted_hdf5")
os.makedirs(output_dir, exist_ok=True)

bin_size = 100.0
box_size = 1000.0
bins_per_axis = int(box_size / bin_size)
total_rows = 111958237

# Start timer
start_time = time.time()

# 1. Create and manage HDF5 bin files
file_map = {}
for x in range(bins_per_axis):
    for y in range(bins_per_axis):
        for z in range(bins_per_axis):
            fname = f"x{x}y{y}z{z}_rockstar_125.h5"
            path = os.path.join(output_dir, fname)
            f = h5py.File(path, "a")
            if "data" not in f:
                max_shape = (None, 11)
                dset = f.create_dataset("data", shape=(0, 11), maxshape=max_shape, dtype='f8', chunks=True)
                dset.attrs['columns'] = ["mvir", "rvir", "rs", "rockstarid", "pid", "x", "y", "z", "vx", "vy", "vz"]
            file_map[(x, y, z)] = f

# 2. Read input and append to appropriate file
with open(input_file, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header

    count = 0
    for row in reader:
        values = list(map(float, row))
        x, y, z = values[5:8]
        bx = int(x // bin_size) % bins_per_axis
        by = int(y // bin_size) % bins_per_axis
        bz = int(z // bin_size) % bins_per_axis

        bin_f = file_map[(bx, by, bz)]
        dset = bin_f["data"]
        dset.resize((dset.shape[0] + 1, 11))
        dset[-1] = values

        count += 1
        if count % 100000 == 0:
            elapsed = time.time() - start_time
            time_left = elapsed * (total_rows / count) - elapsed
            print(f"Done {count:,} rows out of {total_rows:,} in {elapsed:.2f} seconds, estimated {time_left / 60:.2f} minutes left")

            # QA check
            try:
                # Reconstruct bin index
                bx = int(x // bin_size) % bins_per_axis
                by = int(y // bin_size) % bins_per_axis
                bz = int(z // bin_size) % bins_per_axis
                bin_f = file_map[(bx, by, bz)]
                dset = bin_f["data"]

                hdf5_last_row = dset[-1]
                csv_row_array = np.array(values, dtype=np.float64)
                difference = np.abs(hdf5_last_row - csv_row_array)

                print("🧪 QA Check:")
                print("From HDF5:", hdf5_last_row)
                print("From CSV: ", csv_row_array)
                print("Diff:      ", difference)

                if not np.allclose(hdf5_last_row, csv_row_array, rtol=1e-5, atol=1e-8):
                    print("⚠️  Mismatch detected at row", count)
            except Exception as e:
                print(f"❌ QA failed at row {count}: {e}")


# 3. Close files
for f in file_map.values():
    f.close()

# End timer
elapsed = time.time() - start_time

print("Sorting complete!")