# Laser Pointer Detection and Grid Extraction

This project uses Python and OpenCV to detect a laser pointer in a live video feed, stabilize its position, and extract a 5x5 matrix from the surrounding area. The matrix is then visualized alongside intermediate processing steps.

## Features
- **Laser Detection**: Detects a red laser pointer using HSV color thresholds.
- **Position Stabilization**: Ensures consistent detection by stabilizing the laser position over multiple frames.
- **Perspective Transformation**: Extracts a 5x5 grid of cells centered on the laser position.
- **Visualization**: Displays the original frame, binary image, warped perspective, and extracted matrix.

## Demonstration
Watch the project in action on [YouTube](https://youtu.be/jqcH8mrlLNA).

## How It Works
1. **Laser Detection**: The program detects the laser pointer by applying an HSV mask and calculating the centroid of the detected region.
2. **Perspective Transform**: A perspective transformation is applied to the region around the laser to extract a 5x5 grid.
3. **Matrix Extraction**: The transformed area is downsampled into a 5x5 binary matrix.
4. **Visualization**: Results are displayed using Matplotlib.

## Requirements
- Python 
- OpenCV
- NumPy
- Matplotlib


