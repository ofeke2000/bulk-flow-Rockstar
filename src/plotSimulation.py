import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Path to your CSV file
csv_file = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/mdpl2_rockstar_125_pid-1_mvir12.csv")  # Change this to your actual filename

# Output directory for plots
output_dir = os.path.expanduser("~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/heatmap_slices")
os.makedirs(output_dir, exist_ok=True)

# Load the full CSV file into memory
df = pd.read_csv(csv_file)

# Loop over 10 slices of y (0–100, 100–200, ..., 900–1000)
for n in range(4):
    y_min = n * 250
    y_max = (n + 1) * 250

    # Filter data in the y-range
    slice_df = df[(df['y'] >= y_min) & (df['y'] < y_max)]
    num_halos = len(slice_df)
    print(y_min)

    # Create the plot
    fig, ax = plt.subplots()

    # Set black background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Create the hexbin plot with a purple colormap
    hb = ax.hexbin(slice_df['x'], slice_df['z'], gridsize=500, cmap='magma')

    # Add color bar with white label
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Counts', color='white')

    # Set labels and title in white
    ax.set_xlabel('x', color='white')
    ax.set_ylabel('z', color='white')
    ax.set_title(f"Slice {n+1}: y in ({y_min}, {y_max}), Halos: {num_halos}", color='white')

    # Change tick colors to white
    ax.tick_params(colors='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # Save the plot
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"heatmap_y_{y_min}_{y_max}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

print("✅ Done generating heatmaps.")
