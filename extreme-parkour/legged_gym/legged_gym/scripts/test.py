import numpy as np
import argparse
import matplotlib.pyplot as plt

def visualize_heightmap(heightmap, title="Heightmap", cmap="terrain"):
    """
    Visualize a 2D heightmap using a heatmap.
    
    Parameters:
    - heightmap (np.ndarray): 2D array of height values.
    - title (str): Title of the plot (default: "Heightmap").
    - cmap (str): Colormap for visualization (default: "terrain").
                  Options: "viridis", "plasma", "inferno", "terrain", etc.
    """
    # Ensure heightmap is a 2D NumPy array
    if not isinstance(heightmap, np.ndarray) or heightmap.ndim != 2:
        raise ValueError("Heightmap must be a 2D NumPy array")

    # Create figure and axis
    plt.figure(figsize=(8, 6))
    plt.imshow(heightmap, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Height")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)  # Optional: turn off grid for cleaner look
    plt.show()

# Parse command-line arguments
# parser = argparse.ArgumentParser(description="Visualize a 2D heightmap.")
# parser.add_argument("--file", type=str, required=True, help="Path to the heightmap file (NumPy .npy format).")
# args = parser.parse_args()

# Save the file path to a variable
# heightmap_file = args.file

heightmap = np.load(f"./heightmaps/CEP_14/0.npy")
visualize_heightmap(heightmap, title="Heightmap Visualization", cmap="terrain")