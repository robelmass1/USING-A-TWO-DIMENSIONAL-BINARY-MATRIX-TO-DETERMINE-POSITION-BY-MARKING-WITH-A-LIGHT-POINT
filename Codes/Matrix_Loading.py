import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
# Include file directory here below

file_path = '...\Kernel_Size_589_589_2D.npy' # Bigger Matrix
# file_path = '...\Kernel_Size_10_10_2D.npy'    # Smaller Matrix

data = np.load(file_path)

# Display information about the data
print("Data Shape:", data.shape)
print("Data Type:", data.dtype)
print("Data Contents:\n", data)

# Plot the data with gridlines
plt.matshow(data, cmap='gray_r')
plt.grid(color='black', linestyle='-', linewidth=0.5)  # Set grid color, style, and width
plt.xticks([])  # Remove x-axis tick labels
plt.yticks([])  # Remove y-axis tick labels
plt.show()

# Optionally, save the image
output_path = 'binary_image_with_grid.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
