import pandas as pd
import numpy as np
import random
import os
import csv
import time

# Parameters
num_points = 500
radius = 160.0
box_size = 1000.0

# Input/output paths
input_csv = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/mdpl2_rockstar_125_pid-1_mvir12.csv")
output_csv = os.path.expanduser(
    f"~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/bulk_flow_R={radius}_samples={num_points}_fromCSV.csv"
)

# Load the CSV file (assume it has columns: mvir, rvir, rs, rockstarid, pid, x, y, z, vx, vy, vz)
df = pd.read_csv(input_csv)
positions = df[['x', 'y', 'z']].to_numpy()
velocities = df[['vx', 'vy', 'vz']].to_numpy()

# Function to compute periodic distance
def periodic_distance(p1, p2, box_size):
    delta = np.abs(p1 - p2)
    delta = np.where(delta > box_size / 2, box_size - delta, delta)
    return np.sqrt((delta ** 2).sum(axis=1))

# Run sampling
results = []
bulk_flow = []
start_time = time.time()

for i in range(num_points):
    center = np.array([random.uniform(0, box_size) for _ in range(3)])
    dists = periodic_distance(positions, center, box_size)
    mask = dists <= radius

    selected_vels = velocities[mask]
    num_halos = selected_vels.shape[0]

    if num_halos > 0:
        vx_mean = np.mean(selected_vels[:, 0])
        vy_mean = np.mean(selected_vels[:, 1])
        vz_mean = np.mean(selected_vels[:, 2])
        bulk_velocity = np.mean(np.sqrt(vx_mean ** 2 + vy_mean ** 2 + vz_mean ** 2))
    else:
        vx_mean = vy_mean = vz_mean = bulk_velocity = np.nan

    results.append(
        list(center)
        + [vx_mean, vy_mean, vz_mean, bulk_velocity, num_halos]
    )
    bulk_flow.append(bulk_velocity)

    elapsed = time.time() - start_time
    time_left = elapsed * (num_points / (i + 1)) - elapsed
    print(f"Done {i + 1}/{num_points} in {elapsed:.1f}s, ~{time_left / 60:.0f} min left")

# Save results
valid_flows = [v for v in bulk_flow if not np.isnan(v)]
average_bulk_flow = np.mean(valid_flows) if valid_flows else float('nan')

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([f"Radius = {radius:.0f}", f"Points = {num_points}", f"Time = {elapsed:.0f}s", f"Mean Bulk Flow = {average_bulk_flow:.2f}"])
    writer.writerow(["x", "y", "z", "vx_mean", "vy_mean", "vz_mean", "bulk_velocity", "Num_Halos"])
    writer.writerows(results)

output_csv
