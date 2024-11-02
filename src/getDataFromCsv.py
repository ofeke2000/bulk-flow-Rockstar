import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the first 1,000,000 rows of the CSV file
df = pd.read_csv('trial_cosmosim.csv', nrows=10000000)

# Plot a density heatmap using column 6 (x) and column 7 (y)
plt.figure(figsize=(10, 6))
heatmap = sns.histplot(
    data=df,
    x=df.iloc[:, 5],
    y=df.iloc[:, 6],
    bins=75,                # Increase bin count for finer granularity
    pmax=0.9,               # Clip the highest densities at 90% to reveal more contrast
    cmap='plasma',          # Use a different colormap with more contrast
    cbar=True,              # Add color bar (legend)
    log_scale=(False, False) # Use log scale for color intensity (more contrast)
)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('halos densities')

# Save the heatmap plot as an image (PNG format)
plt.savefig('density_heatmap_column6_vs_column7_contrast.png', dpi=300)

# Optionally, display the heatmap
# plt.show()

# Print success message
print("Density heatmap saved as density_heatmap_column6_vs_column7_contrast.png")
