import numpy as np
import argparse

import matplotlib.pyplot as plt

def visualize_npy(file_path):
    data = np.load(file_path)
    if data.ndim == 1:
        plt.plot(data)
        plt.title("1D Data Visualization")
        plt.xlabel("Index")
        plt.ylabel("Value")
    elif data.ndim == 2:
        plt.imshow(data, cmap='viridis', aspect='auto')
        plt.title("2D Data Visualization")
        plt.colorbar(label="Value")
    else:
        print(f"Data has {data.ndim} dimensions, which is not supported for visualization.")
        return
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize .npy files")
    parser.add_argument("file_path", type=str, help="Path to the .npy file")
    args = parser.parse_args()

    visualize_npy(args.file_path)