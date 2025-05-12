import matplotlib.pyplot as plt

# === INPUT DATA (you'll fill these in) ===
radii = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]  # Replace with your radii
bulk_flow_magnitude = [273, 248, 234, 216, 193, 181, 172, 161, 149, 144, 130, 125, 120, 112, 106, 97, 96, 89, 80, 80, 76]  # Replace with your computed bulk flow magnitudes

# Optional: per-direction bulk flow if you have them
#vx_means = [100, 90, 85, 80, 75]
#vy_means = [200, 150, 130, 120, 110]
#vz_means = [300, 260, 230, 210, 200]

# === PLOT TOTAL BULK FLOW MAGNITUDE ===
plt.figure(figsize=(8, 6))
plt.plot(radii, bulk_flow_magnitude, marker='o', color='purple', label='Total Bulk Flow')
plt.plot([50, 250], [275, 75], marker='o', label='fit', color='red')
plt.xlabel('Radius')
plt.ylabel('Bulk Flow Magnitude')
plt.title('Bulk Flow vs Radius')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("bulk_flow_vs_radius.png")
plt.show()

# === PLOT PER-DIRECTION (optional) ===
#plt.figure(figsize=(8, 6))
#plt.plot(radii, vx_means, marker='o', label='Vx Mean', color='red')
#plt.plot(radii, vy_means, marker='o', label='Vy Mean', color='green')
#plt.plot(radii, vz_means, marker='o', label='Vz Mean', color='blue')
#plt.xlabel('Radius')
#plt.ylabel('Mean Velocity in Each Direction')
#plt.title('Directional Mean Velocities vs Radius')
#plt.grid(True)
#plt.legend()
#plt.tight_layout()
#plt.savefig("directional_means_vs_radius.png")
#plt.show()
