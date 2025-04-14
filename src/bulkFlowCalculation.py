import os
import h5py
import numpy as np
import random
import csv
import time
import re

#Run Parameters
num_points = 10  # Number of random points
radius = 120.0  # Sphere radius
box_size = 1000.0  # Simulation box size
bin_size = 100.0  # Each bin represents 100 units in space
bins_per_axis = int(box_size / bin_size)

data_dir = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid_-1/grid_sorted_hdf5")
output_csv = os.path.expanduser(f"~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid_-1/bulk_flow_R = {radius}_samples = {num_points}.csv")

def extract_bin_coords(filename):
    match = re.search(r"x(\d+)y(\d+)z(\d+)", filename)
    if match:
        return tuple(map(int, match.groups()))
    else:
        raise ValueError(f"Filename does not match bin pattern: {filename}")

# Load all bin file names into a dictionary
bin_file_map = {}

for filename in os.listdir(data_dir):
    if not filename.endswith(".h5"):
        continue

    try:
        x, y, z = extract_bin_coords(filename)
        bin_file_map[(x, y, z)] = os.path.join(data_dir, filename)
    except ValueError as e:
        print(f"Skipping file: {e}")

# Function to get nearby bin indices using periodic boundaries
def get_bins_in_radius(center, radius, box_size = 1000,bin_size = 100):

    bins_per_axis = box_size / bin_size

    #Define cube corners: min and max coords
    x_min, y_min, z_min = [(c - radius) % box_size for c in center]
    x_max, y_max, z_max = [(c + radius) % box_size for c in center]

    #Convert to bin indices
    ix_min, iy_min, iz_min = [int(v // bin_size) for v in (x_min, y_min, z_min)]
    ix_max, iy_max, iz_max = [int(v // bin_size) for v in (x_max, y_max, z_max)]

    #Handle wrap-around using periodic boundary conditions
    def wrap_range(min_idx, max_idx):
        if max_idx >= min_idx:
            return list(range(min_idx, max_idx + 1))
        else:
            # Wrap around
            return list(range(min_idx, bins_per_axis)) + list(range(0, max_idx + 1))

    x_bins = wrap_range(ix_min, ix_max)
    y_bins = wrap_range(iy_min, iy_max)
    z_bins = wrap_range(iz_min, iz_max)

    #Generate all bin combinations
    bins = [(x, y, z) for x in x_bins for y in y_bins for z in z_bins]
    return bins

# Periodic distance
def periodic_distance(p1, p2):
    delta = np.abs(np.array(p1) - np.array(p2))
    delta = np.where(delta > box_size / 2, box_size - delta, delta)
    return np.sqrt((delta ** 2).sum())

# Main computation
results = []
bulk_flow = []

# Start timer
start_time = time.time()

for point in range(num_points):
    center = [int(random.uniform(0, box_size)) for _ in range(3)]
    velocity_vectors = []
    num_bins_used = 0

    for bin_idx in get_bins_in_radius(center, radius):
        bin_file = bin_file_map.get(bin_idx)
        if not bin_file:
            continue

        num_bins_used += 1

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

    num_halos = len(velocity_vectors)

    results.append(center + [bulk_velocity] + [num_bins_used] + [num_halos])
    bulk_flow.append(bulk_velocity)
    elapsed = time.time() - start_time
    time_left = elapsed * (num_points / (point+1)) - elapsed
    print(f"Done {point+1:,} points out of {num_points:,} in {elapsed:.2f} seconds, estimated {time_left / 60:.2f} minutes left")

# Calculate average bulk flow across all points
valid_flows = [v for v in bulk_flow if not np.isnan(v)]
if valid_flows:
    average_bulk_flow = sum(valid_flows) / len(valid_flows)
else:
    average_bulk_flow = float('nan')

# Save to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([f"Radius = {radius:.0f}", f"Num of points = {num_points}", f"time = {elapsed:.0f}s", f"bulk flow = {average_bulk_flow:.2f}"])
    writer.writerow(["x", "y", "z", "bulk_velocity", "Num_Bins", "Num_Halo's"])
    writer.writerows(results)

print(f"Done! Bulk flow moments for {num_points} random points saved to:\n{output_csv}")
