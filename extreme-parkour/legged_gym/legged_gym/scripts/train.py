# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
import argparse
from legged_gym.envs import *
from legged_gym.utils import task_registry, add_shared_args, process_args
import shutil
import torch
import wandb
import subprocess
from pathlib import Path
import pickle
import json
import random
import time
from omegaconf import OmegaConf

from legged_gym.scripts.algo.flow_matching_CEP import FlowMatchingCEP
import matplotlib.pyplot as plt

random.seed(time.time())
np.random.seed(int(time.time()))

os.environ["WANDB_SILENT"] = "False"
file_dir = os.path.dirname(os.path.abspath(__file__))  # Location of this file

NUM_ROWS = 10
NUM_COLS = 30

# Create a dictionary with instance identifiers as keys and priorities as values
priority_dict = {}

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

def update_energy_network(heightmaps, max_rewards, mean_rewards, max_goals, avg_goals, cep_type):
    """
    Load the energy network model from the specified path.
    Train the model using the training data. (input: Heightmaps, output: target values)
    """
    energy_dir = Path(os.getcwd()) / "gen_models" / "energy_network"
    energy_model_path = os.path.join(energy_dir, "energy.pt")
    flow_dir = Path(os.getcwd()) / "gen_models" / "diffusion"
    flow_model_path = os.path.join(flow_dir, "diffusion_150000.pt")
    
    diffusion_arg = {
        "eval_batch_size": 16,
        "eval_dt": 4,
        "device": "cuda",
        "debug": True,
        "input_h": 368,
        "input_w": 80,
        "hidden_size": 256,
        "depth": 12,
        "num_heads": 8,
        "tokenize_scale": 8,
        "tokenize_dim": 32,
        "ema_decay": 0.99,
        "denoising_step": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "energy_lr": 1e-4,
        "energy_weight_decay": 1e-6,
        "lamda_guidance": 7,
    }
    diffusion_cfg = OmegaConf.create(diffusion_arg)
    flow_matching = FlowMatchingCEP(diffusion_cfg)
    flow_matching.model.load_state_dict(torch.load(flow_model_path, weights_only=True), strict=False)
    # Check if the model exists at path
    if os.path.exists(energy_model_path):
        # Load the model
        energy_ckpt = torch.load(energy_model_path, weights_only=True)
        flow_matching.energy_network.load_state_dict(energy_ckpt, strict=False)
        flow_matching.energy_network.to("cuda")

    padding = ((4,4), (0,0))
    heightmaps = [np.pad(heightmap, padding, mode='constant') for heightmap in heightmaps]
    heightmaps = [np.expand_dims(heightmap, axis=-1).transpose((2,0,1)) for heightmap in heightmaps]
    heightmaps = [torch.from_numpy(heightmap).to(torch.float32).to("cuda") for heightmap in heightmaps]
    # latents = [flow_matching.vae.encode(torch.tensor(heightmap, dtype=torch.float32).unsqueeze(0).to("cuda")) for heightmap in heightmaps]

    # double the size of dataset
    heightmaps = heightmaps*2
    max_rewards = max_rewards*2
    mean_rewards = mean_rewards*2
    max_goals = max_goals*2
    avg_goals = avg_goals*2

    # Mix the heightmaps and create batches
    batch_size = 32
    num_batches = len(heightmaps) // batch_size
    indices = np.arange(len(heightmaps))
    num_updates = 500
    
    for i in range(num_updates):
        np.random.shuffle(indices)
        flow_matching.energy_network.train()
        for batch_idx in range(num_batches):
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # batch_latents = flow_matching.vae.encode([heightmaps[i] for i in batch_indices])
            batch_heightmaps = torch.stack([heightmaps[i] for i in batch_indices])
            batch_max_rewards = torch.tensor([max_rewards[i] for i in batch_indices], dtype=torch.float32).to("cuda")  # Shape: [batch_size]
            batch_mean_rewards = torch.tensor([mean_rewards[i] for i in batch_indices], dtype=torch.float32).to("cuda")  # Shape: [batch_size]
            batch_max_goals = torch.tensor([max_goals[i] for i in batch_indices], dtype=torch.float32).to("cuda")  # Shape: [batch_size]
            batch_avg_goals = torch.tensor([avg_goals[i] for i in batch_indices], dtype=torch.float32).to("cuda")  # Shape: [batch_size]
            if cep_type == "reward":
                batch_guidance_data = batch_max_rewards-batch_mean_rewards
            elif cep_type == "goals":
                batch_guidance_data = batch_max_goals/(batch_avg_goals+1e-5)
            elif cep_type == "combined":
                batch_guidance_data = (batch_max_rewards-batch_mean_rewards) * (batch_max_goals-batch_avg_goals+1e-5)
            elif cep_type == "combined_simple":
                batch_guidance_data = (batch_max_rewards-batch_mean_rewards) * (batch_max_goals-batch_avg_goals+1e-5) / batch_max_goals
            elif cep_type == "combined_avg":
                batch_guidance_data = (batch_max_rewards-batch_mean_rewards) * (batch_avg_goals + 8)/8
            if torch.any(batch_avg_goals == 0):
                    batch_guidance_data[batch_avg_goals == 0] = 0    
            # normalize the guidance data
            batch_guidance_data = (batch_guidance_data - batch_guidance_data.min()) / (batch_guidance_data.max() - batch_guidance_data.min())
            flow_matching.update_energy(batch_heightmaps, batch_guidance_data)
    # flow_matching.update_energy(heightmaps, max_rewards, mean_rewards)

    # save the updated model
    torch.save(flow_matching.energy_network.state_dict(), energy_model_path)

def create_energy_training_data(train_data, full_heightfield):
    """
    Create training data for the energy network.
    The training data consists of heightmaps and their corresponding rewards.
    """
    # Initialize lists to store heightmaps and rewards
    heightmaps = []
    max_rewards = []
    mean_rewards = []
    max_goals = []
    avg_goals = []

    # Iterate through the training data
    for terrain_index, data in train_data.items():
        j,i = terrain_index

        border = 100

        start_x = border + i * 360
        end_x = border + (i+1) * 360
        start_y = border + j * 80
        end_y = border + (j+1) * 80

        # Extract the heightmap for the current terrain type
        heightmap = full_heightfield[start_x:end_x, start_y:end_y]
        visualize_heightmap(heightmap)
        max_reward = data["max_reward"]
        mean_reward = data["avg_reward"]
        max_goal = data["max_goals_reached"] if "max_goals_reached" in data else 0
        avg_goal = data["avg_goals_reached"] if "avg_goals_reached" in data else 0

        # Append the data to the lists
        heightmaps.append(heightmap)
        max_rewards.append(max_reward)
        mean_rewards.append(mean_reward)
        max_goals.append(max_goal)
        avg_goals.append(avg_goal)

    return heightmaps, max_rewards, mean_rewards, max_goals, avg_goals

def extract_priority_dict(replay_buffer):
    """
    Extracts the priority dictionary from the replay buffer.
    The keys are unique identifiers for each instance, and the values are their priorities.
    """
    # Iterate through the replay buffer to extract priorities
    for terrain_type, data in replay_buffer.items():
        for instance, instance_data in data["train_data"].items():
            # Combine terrain_type and instance to create a unique identifier
            instance_id = f"{terrain_type}_{instance.replace('(', '').replace(')', '').replace(', ', '_')}"
            priority_dict[instance_id] = instance_data.get("priority", 0)

    # Print the resulting dictionary
    # print(priority_dict)

def generate_terrain_grid(num_rows, num_cols, priority_dict, terrain_type, randomness_threshold=0.5):
    """
    Generate a grid of terrains based on priority_dict or assign random terrain_type.

    Args:
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.
        priority_dict (dict): Dictionary with keys as terrain identifiers and values as priorities.
        terrain_type : List of possible terrain types.

    Returns:
        np.ndarray: A grid of terrain types.
    """
    grid = np.empty((num_rows, num_cols), dtype=object)

    # Normalize priorities in the priority_dict for PLR
    total_priority = sum(priority_dict.values())
    if total_priority > 0:
        probabilities = {k: v / total_priority for k, v in priority_dict.items()}
    else:
        probabilities = {k: 1 / len(priority_dict) for k in priority_dict}

    # Filter keys with probabilities in the top 20% to 50% range
    lower_bound = 0.5
    upper_bound = 0.9
    sorted_items = sorted(probabilities.items(), key=lambda item: item[1])  # Sort by probability values
    num_items = len(sorted_items)
    lower_index = int(num_items * lower_bound)
    upper_index = int(num_items * upper_bound)
    filtered_probabilities = dict(sorted_items[lower_index:upper_index])

    for row in range(num_rows):
        for col in range(num_cols):
            # Decide whether to use terrain_type or a key from priority_dict
            if random.random() < randomness_threshold and filtered_probabilities:
                # Select a key from priority_dict based on PLR probabilities
                selected_key = random.choices(list(filtered_probabilities.keys()), weights=filtered_probabilities.values(), k=1)[0]
                grid[row, col] = f"{selected_key}"
            else:
                # Assign the fixed terrain_type
                grid[row, col] = terrain_type

    return grid

def save_grid_as_npy(grid, replay_buffer_dir, filename="grid.npy"):
    """
    Save the generated grid as a .npy file in the replay buffer directory.

    Args:
        grid (np.ndarray): The generated terrain grid.
        replay_buffer_dir (Path): Path to the replay buffer directory.
        filename (str): Name of the .npy file (default: "grid.npy").
    """
    grid_file = replay_buffer_dir / filename
    np.save(grid_file, grid)
    print(f"Terrain grid saved to {grid_file}")

def save_file(file, log_dir):
    wandb.save(file, policy="now")
    filename = os.path.basename(file)
    if filename in os.listdir(log_dir):
        os.rename(log_dir / filename, log_dir / f"{filename}.old")
    shutil.copy(file, log_dir)

def compute_priority(data, replay_type="default"):
    """Compute priority based on the PLR algorithm."""
    avg_reward = data["avg_reward"] if "avg_reward" in data and data["avg_reward"] else 0
    avg_goals_reached = data["avg_goals_reached"] if "avg_goals_reached" in data and data["avg_goals_reached"] else 0
    max_reward = data["max_reward"] if "max_reward" in data and data["max_reward"] else 0
    
    if replay_type == "default":
        # Default priority: inverse of reward
        return 1 / (avg_reward + 1e-5)
    elif replay_type == "RD": #reward difference
        return (max_reward - avg_reward)  * (avg_goals_reached + 8 / 8)
    elif replay_type == "goals":
        # Priority based on goals reached
        return 8 / (avg_goals_reached + 1e-5)
    elif replay_type == "combined":
        # Combined priority: weighted sum of reward and goals
        return 0.5 * (1 / (avg_reward + 1e-5)) + 0.5 * (8 / (avg_goals_reached + 1e-5))
    elif replay_type == "uniform":
        # Uniform priority for all instances
        return 1.0
    else:
        raise ValueError(f"Unknown replay type: {replay_type}")

def train(args):
    wandb.init(
        project=args.proj_name,
        name=args.exptid,
        group=args.wandb_group,
        mode=("online" if args.use_wandb else "disabled"),
        dir=f"{file_dir}/../../logs",
        id=args.wandb_id,
        resume="allow"
    )

    # Save train_data as a JSON file in the correct directory
    replay_buffer_dir = Path(os.getcwd()) / "heightmaps" / f"{args.exptid}"
    replay_buffer_dir.mkdir(parents=True, exist_ok=True)
    replay_buffer_file = replay_buffer_dir / "replay_buffer.json"

    if replay_buffer_file.exists():
        with open(replay_buffer_file, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    
    extract_priority_dict(existing_data)

    if(args.no_replay):
        grid = np.ones((NUM_ROWS, NUM_COLS), dtype=object) * args.terrain_type
    else:
        grid = generate_terrain_grid(NUM_ROWS, NUM_COLS, priority_dict, args.terrain_type)
    # print(grid)
    save_grid_as_npy(grid, replay_buffer_dir)

    if args.render_images:
        # To avoid affecting training, we run rendering in a separate process
        # NOTE: If rendering if done with too many environments (like that used for training), there may be unexpected errors
        #       Thus, we cannot render and train with the same simulation setup
        print("Running render.py subprocess and waiting...")
        render_log_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / args.proj_name / args.exptid / "renders"
        if render_log_dir is not None:
            render_log_dir.mkdir(parents=True, exist_ok=True)
        render_command = f"python {file_dir}/render.py --task {args.task} --save_dir {render_log_dir} --terrain_type {args.terrain_type} --terrain_rows 1 --device {args.device}"
        process = subprocess.Popen(
            render_command.split(" "),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ.copy(), "TQDM_DISABLE": "1"}
        )
        _, stderr = process.communicate()
        if stderr:
            print("Error in render.py subprocess:")
            print(stderr.decode('utf-8'))
        wandb.log({"terrain_render": wandb.Image(str(render_log_dir / "summary.png"))}, commit=False)

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print("made env")
    ppo_runner, train_cfg, log_dir, _, _ = task_registry.make_alg_runner(env=env, args=args, name=args.task)
    if args.render_images:
        assert log_dir == render_log_dir.parent, "Log directory mismatch between train.py and render.py"

    # Save config as pickle
    with open(log_dir / "legged_robot_config.pkl", "wb") as f:
        cfg = (env_cfg, train_cfg)
        pickle.dump(cfg, f)
    
    print(f"Starting training, using log directory {log_dir}...")
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    wandb.finish(quiet=True)

    train_data = ppo_runner.get_train_data()
    # print(train_data)

    for key,value in train_data.items():
        # Extract cycle_name and terrain_type from the key
        # print(key)
        terrain_type, terrain_level = key #terrain_level, terrain_type = i,j
        grid_value = grid[int(terrain_level), int(terrain_type)]  # Get the corresponding grid value
        # print(grid_value)
        if "_" in grid_value:
            cycle_name, terrain_type, terrain_level = grid_value.split("_")
        else:
            cycle_name = grid_value

        sub_key = f"({terrain_type}, {terrain_level})"
        if cycle_name in existing_data:
            if sub_key in existing_data[cycle_name]["train_data"]:
                existing_data[cycle_name]["train_data"][sub_key] = value
            else:
                existing_data[cycle_name]["train_data"][sub_key] = value
        else:
            existing_data[cycle_name] = {
                "train_data": {sub_key: value}
            }

    # Add priority to each instance in existing_data
    for cycle_name, cycle_data in existing_data.items():
        for sub_key, instance_data in cycle_data["train_data"].items():
            if isinstance(instance_data, dict):
                instance_data["priority"] = compute_priority(instance_data, replay_type=args.replay_type)


    # Save the updated data back to the JSON file
    with open(replay_buffer_file, "w") as f:
        json.dump(existing_data, f)

    if args.CEP:
        full_heightfield = np.load(replay_buffer_dir / f"{args.terrain_type}.npy")
        heightmaps, max_rewards, mean_rewards, max_goals, avg_goals = create_energy_training_data(train_data, full_heightfield)
        if int(args.terrain_type) > 1:
            update_energy_network(heightmaps, max_rewards, mean_rewards, max_goals, avg_goals, cep_type=args.CEP_type)
    
    print(f"Replay buffer saved to {replay_buffer_file}")

    reward = 0

    return reward

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_shared_args(parser)

    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint")
    parser.add_argument("--load_run", type=str, help="Name of the run to load when resuming. If unspecified, will load exptid. Overrides config file if provided.")
    parser.add_argument("--checkpoint", type=int, default=-1, help="Which model checkpoint to load. If -1, will load the last checkpoint. Overrides config file if provided.")
    parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
    parser.add_argument("--render_images", action="store_true", default=False, help="Render the environment and save images")
    parser.add_argument("--no_replay", action="store_true", default=False, help="Do not use replay buffer")
    parser.add_argument("--replay_type", type=str, default='default', help="Type of priority for the replay buffer to save priorities")

    # diffusion args
    parser.add_argument("--CEP", action="store_true", default=False, help="Use CEP")
    parser.add_argument("--CEP_type", type=str, default="reward", help="Type of CEP guidance to use")
    parser.add_argument("--input_h", type=int, default=368)
    parser.add_argument("--input_w", type=int, default=80)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)

    args = parser.parse_args()
    args = process_args(args)
    if not args.headless:
        print("Setting headless to True, overriding")
        args.headless = True
    # args.headless = False
    # args.use_wandb = True

    args.script = "train"
    # args.max_iterations = 1
    # for i in range(100):
    #     train(args)
    reward = train(args)
    
