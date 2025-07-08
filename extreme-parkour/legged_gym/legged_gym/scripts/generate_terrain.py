import os
import numpy as np

file_dir = os.path.dirname(os.path.abspath(__file__))

def generate_heightmap(size, seed=None, difficulty):
    """
    Generate a heightmap with a given size and seed.

    Parameters:
    size (int): The size of the heightmap.
    seed (int): The seed for the random number generator.

    Returns:
    np.array: The generated heightmap.
    """
    if seed is not None:
        np.random.seed(seed)

    heightmap = np.random.rand(size, size)
    return heightmap

def save_heightmap(heightmap, run_id, iter, save_dir):
    """
    Save the AI generated heightmap to a specific directory with a filename based on run_id and iter.

    Parameters:
    heightmap (np.array): The generated heightmap.
    run_id (int): The run identifier.
    iter (int): The iteration number.
    save_dir (str): The directory where the heightmap will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = f"heightmap_run-{run_id}_iter-{iter}.npy"
    filepath = os.path.join(save_dir, filename)
    
    np.save(filepath, heightmap)
    print(f"Heightmap saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # Example heightmap
    heightmap = np.random.rand(100, 100)
    run_id = 1
    iter = 10
    save_dir = Path(f"{file_dir}/../heightmaps/")

    parser = argparse.ArgumentParser()
    add_shared_args(parser)

    parser.add_argument("--run_id", type=str, help="Name of the run")
    parser.add_argument("--iter", type=str, help="Number of iteration step")
    parser.add_argument("--difficulty", type=str, help="Difficulty level of the terrain")
    # parser.add_argument("--size", type=int, help="Size of the heightmap")
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")

    args = parser.parse_args()
    args = process_args(args)
    if not args.headless:
        print("Setting headless to True, overriding")
        args.headless = True

    args.script = "train"
    train(args)
    
    save_heightmap(heightmap, run_id, iter, save_dir)