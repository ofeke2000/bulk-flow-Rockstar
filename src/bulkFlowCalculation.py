import os
import h5py
import numpy as np
import random
import csv
import time

#Run Parameters
num_points = 1  # Number of random points
radius = 100.0  # Sphere radius
box_size = 1000.0  # Simulation box size
bin_size = 100.0  # Each bin represents 100 units in space
bins_per_axis = int(box_size / bin_size)

data_dir = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid_-1/grid_sorted_hdf5")
output_csv = os.path.expanduser(f"~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid_-1/bulk_flow_R{radius}_samples{num_points}.csv")

# Load all bin file names into a dictionary
bin_file_map = {}
for filename in os.listdir(data_dir):
    if filename.endswith(".h5"):
        parts = filename.split("_")[0]
        x = int(parts[1])
        y = int(parts[2][1])
        z = int(parts[3][1])
        print({x,y,z})
        bin_file_map[(x, y, z)] = os.path.join(data_dir, filename)

# Function to get nearby bin indices using periodic boundaries
def get_bins_in_radius(center, radius, box_size=1000, bins_per_axis=10):
    bin_size = box_size / bins_per_axis
    bins = []

    # Convert center and radius to bin indices
    cx, cy, cz = center
    min_bin_x = int((cx - radius) % box_size // bin_size)
    max_bin_x = int((cx + radius) % box_size // bin_size)
    min_bin_y = int((cy - radius) % box_size // bin_size)
    max_bin_y = int((cy + radius) % box_size // bin_size)
    min_bin_z = int((cz - radius) % box_size // bin_size)
    max_bin_z = int((cz + radius) % box_size // bin_size)

    # Loop over bins in each direction
    for i in range(min_bin_x, max_bin_x + 1):
        for j in range(min_bin_y, max_bin_y + 1):
            for k in range(min_bin_z, max_bin_z + 1):
                # Use modulo for periodic boundary conditions
                ii = i % bins_per_axis
                jj = j % bins_per_axis
                kk = k % bins_per_axis
                bins.append((ii, jj, kk))

    return bins

# Periodic distance
def periodic_distance(p1, p2):
    delta = np.abs(np.array(p1) - np.array(p2))
    delta = np.where(delta > box_size / 2, box_size - delta, delta)
    return np.sqrt((delta ** 2).sum())

# Main computation
results = []

# Start timer
start_time = time.time()

for point in range(num_points):
    center = [random.uniform(0, box_size) for _ in range(3)]
    velocity_vectors = []

    for bin_idx in get_bins_in_radius(center, radius):
        bin_file = bin_file_map.get(bin_idx)
        if not bin_file:
            continue

        with h5py.File(bin_file, "r") as f:
            data = f["data"][:]
            for row in data:
                x, y, z = row[5:8]
                if periodic_distance(center, [x, y, z]) <= radius:
                    vx, vy, vz = row[8:11]
                    velocity_vectors.append([vx, vy, vz])

    if velocity_vectors:
        v = np.array(velocity_vectors)
        bulk_velocity = np.sqrt(np.mean(v[:, 0]) ** 2 + np.mean(v[:, 1]) ** 2 + np.mean(v[:, 2]) ** 2)
    else:
        bulk_velocity = np.nan  # No data in sphere

    results.append(center + [bulk_velocity])
    elapsed = time.time() - start_time
    time_left = elapsed * (num_points / point) - elapsed
    print(f"Done {point:,} rows out of {num_points:,} in {elapsed:.2f} seconds, estimated {time_left / 60:.2f} minutes left")

# Calculate average bulk flow across all points
average_bulk_flow = sum(bulk_velocity) / num_points

# Save to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
                        f"Radius = {radius:,} Num of points = {num_points:,} time = {elapsed:.2f}s bulk flow = {average_bulk_flow:.3f}"])
    writer.writerow(["x", "y", "z", "bulk_velocity"])
    writer.writerows(results)

print(f"Done! Bulk flow moments for {num_points} random points saved to:\n{output_csv}")
