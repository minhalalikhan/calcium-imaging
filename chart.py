import numpy as np
import matplotlib.pyplot as plt
import os

# --- File paths ---
folder_path = "data"              # Folder containing data
file_name = "losses.npy"          # Input file
output_image = "loss_plot.png"    # Output plot file
file_path = os.path.join(folder_path, file_name)
output_path = os.path.join(folder_path, output_image)

# --- Load NumPy array ---
losses = np.load(file_path)

# --- Plot the losses ---
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, markersize=4)
plt.title('Training Loss Over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')

# Add y-axis ticks every 0.001
y_min, y_max = plt.ylim()
plt.yticks(np.arange(y_min, y_max + 0.001, 0.005))

# --- Save the plot ---
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Plot saved to: {output_path}")
