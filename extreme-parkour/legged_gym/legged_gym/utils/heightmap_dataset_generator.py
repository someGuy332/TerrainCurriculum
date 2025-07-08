import os
# import sys

# sys.path = [p for p in sys.path if 'ros' not in p]

# import random
from legged_gym.utils.terrain_utils import generate_heightfield
import numpy as np

env_length = 360
env_width = 80

height_field = np.zeros((env_length, env_width), dtype=np.float32)

def generate_terrain():
    """
    Generate a terrain with random heightfield
    :return: heightfield, terrain type
    """
    # Generate a random terrain type
    height_field = generate_heightfield(
        env_length,
        env_width,
        0.05,
        seed=np.random.randint(0, 10000)
    )

    return height_field

def save_terrain(name):
    """
    Save the generated terrain to a file
    :param name: name of the file
    """
    # Ensure the heightmap_dataset directory exists
    dataset_dir = os.path.join(os.path.dirname(__file__), "heightmap_dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # Generate and save the terrain
    height_field = generate_terrain()
    file_path = os.path.join(dataset_dir, f"{name}.npy")
    np.save(file_path, height_field)
    print(f"Terrain saved as {file_path}")

if __name__ == "__main__":
    # Test the terrain generation
    save_terrain("test_terrain")