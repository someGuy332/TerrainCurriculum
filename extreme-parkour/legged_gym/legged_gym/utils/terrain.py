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
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
import os
import json  # Add import for loading replay buffer
import time
import torch
import copy

from scipy.ndimage import label, center_of_mass, gaussian_filter

from omegaconf import OmegaConf, DictConfig

from legged_gym.scripts.algo.vae import VAE
from legged_gym.scripts.algo.flow_matching import FlowMatching
from legged_gym.scripts.algo.flow_matching_CEP import FlowMatchingCEP

random.seed(time.time())
np.random.seed(int(time.time()))
# import statistics

import matplotlib.pyplot as plt

from noise import pnoise2
from pprint import pprint

from legged_gym.utils.terrain_utils import generate_heightfield

EXPTID = f"CEP_3"

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

class AdversarialTerrainGenerator:
    def __init__(self, terrain_width, terrain_length, num_envs):
        self.terrain_width = terrain_width
        self.terrain_length = terrain_length
        self.num_envs = num_envs
        self.difficulties = np.ones(num_envs) * 0.1

    def generate(self):
        height_maps = np.zeros((self.num_envs, self.terrain_width, self.terrain_length))
        for i in range(self.num_envs):
            height_maps[i] = self.generate_height_map(self.terrain_width, self.terrain_length, self.difficulties[i])
        return height_maps
    
    def perlin_noise(self, width, length, amp=10):
        return np.random.rand(width, length) * amp

    # Modify generate height map to change algorithm for terrain generation
    def generate_height_map(self, width, length, difficulty):
        height_map = np.zeros((width, length))
        # for i in range(width):
        #     for j in range(length):
        #         height_map[i, j] =  1 # The algorithm plaed here
        height_map = self.perlin_noise(width, length, amp=10)
        return height_map

    def update_difficulty(self, data):
        return 0

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        # pprint(vars(cfg))
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.num_envs = cfg.num_rows * cfg.num_cols
        self.rand = np.random.choice(np.arange(70,140))
        self.grid = None

        self.diffusion_arg = {
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
        }
        self.diffusion_cfg = OmegaConf.create(self.diffusion_arg)
        self.flow_matching = FlowMatching(self.diffusion_cfg)
        self.vae_ckpt = torch.load("/home/yoonho/Workspace/RLLab/Multiverse/extreme-parkour/legged_gym/legged_gym/scripts/gen_models/vae/vae.pt", weights_only=True)
        self.vae = VAE()
        self.vae.load_state_dict(self.vae_ckpt)
        self.vae.to('cuda')
        self.vae.eval()
        self.flow_matching.model.load_state_dict(torch.load("/home/yoonho/Workspace/RLLab/Multiverse/extreme-parkour/legged_gym/legged_gym/scripts/gen_models/diffusion/diffusion_150000.pt", weights_only=True))
        self.flow_matching.model.to('cuda')
        self.flow_matching.ema_model = copy.deepcopy(self.flow_matching.model) 
        self.flow_matching.ema_model.to('cuda')

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border
        self.generator = AdversarialTerrainGenerator(self.tot_rows, self.tot_cols, self.num_envs)
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        # print(self.height_field_raw.shape)
        self.load_grid()

        if cfg.type == "generate":
            print("generate")
            self.randomized_terrain()

        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)

        if not self.heightmap_exists(cfg.type, EXPTID):
            self.save_height_field(EXPTID)
        else:
            pass
        
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            if cfg.hf2mesh_method == "grid":
                print("Converting heightmap to trimesh...")
                print(self.cfg.vertical_scale)
                print(self.height_field_raw)
                print(self.height_field_raw.shape)
                # visualize_heightmap(self.height_field_raw)
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                # print(self.height_field_raw)
                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width*2+1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))
    
    def load_height_map(self, exptid, type):
        # height_field_raw = np.zeros((self.terrain_width, self.terrain_length), dtype=np.int16)
        height_field = np.load(os.path.join(os.getcwd(), "heightmaps", exptid ,f"{type}.npy"), mmap_mode='r')
        # visualize_heightmap(height_field)
        # height_field_raw = np.zeros((height_field.shape[0], height_field.shape[1]), dtype=np.float32)
        return height_field

    def diffusion_generate(self):
        with torch.no_grad():
            z_generated = self.flow_matching.sample(1, self.diffusion_cfg.eval_dt)
            x_generated = self.flow_matching.vae.decode(z_generated)
        
        # print(x_generated.shape)
        x_generated = x_generated.detach().cpu().numpy()
        x_generated = x_generated.transpose((0,2,3,1))
        x_generated = x_generated.reshape((-1, x_generated.shape[2], x_generated.shape[3]))
        # print(x_generated.shape)
        # visualize_heightmap(x_generated[:,:,0], title="Generated Heightmap", cmap="terrain")
        
        return x_generated[3:363,:,0]

    def load_grid(self):
        grid_file = os.path.join(os.getcwd(), "heightmaps", EXPTID, "grid.npy")
        if os.path.exists(grid_file):
            self.grid = np.load(grid_file, allow_pickle=True)
            print(f"Grid loaded successfully from {grid_file}")
        else:
            print(f"Grid file not found: {grid_file}")
            self.grid = np.ones((self.cfg.num_rows, self.cfg.num_cols), dtype=object) * self.terrain_type
    
    def save_terrain(self, height_field, name):
        """
        Save the generated terrain to a file
        :param name: name of the file
        """
        # Ensure the heightmap_dataset directory exists
        dataset_dir = os.path.join(os.path.dirname(__file__), "heightmap_dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        file_path = os.path.join(dataset_dir, f"subterrain_{name}_mixed.npy")
        np.save(file_path, height_field)
        print(f"Terrain saved as {file_path}")

    def randomized_terrain(self):
        for k in range(98000):
            # Env coordinates in the world
            # (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(0.5, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.save_terrain(terrain.height_field_raw * self.cfg.vertical_scale , f"{k+2010}")
        exit(0)
            # self.add_terrain_to_map(terrain, i, j)

    def save_height_field(self, exptid=EXPTID):
        """
        Save the height field to a folder named after the experiment ID (EXPTID).
        If the folder already exists, save the file inside it.
        """
        terrain_type = self.cfg.type
        # Create a folder named by EXPTID
        folder_path = os.path.join(os.getcwd(), "heightmaps", exptid)
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the height_field_raw as a .npy file inside the folder
        file_path = os.path.join(folder_path, f"{terrain_type}.npy")
        np.save(file_path, self.height_field_raw)
        print(f"Height field saved at: {file_path}")

    def heightmap_exists(self, terrain_type, exptid=EXPTID):
        """
        Check if a heightmap file already exists for the given exptid and terrain_type.

        Parameters:
        - exptid (str): The experiment ID.
        - terrain_type (str): The type of terrain.

        Returns:
        - bool: True if the heightmap file exists, False otherwise.
        """
        folder_path = os.path.join(os.getcwd(), "heightmaps", exptid)
        file_path = os.path.join(folder_path, f"{terrain_type}.npy")
        return os.path.exists(file_path)
    
    def update_terrains(self, data):
        self.generator.update_difficulty(data)
        self.generate_terrains()

    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / (self.cfg.num_rows-1) if self.cfg.num_rows > 1 else 0.5
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.7, 1))
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)
                    # print(terrain)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        print(terrain_type)
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def make_terrain(self, choice, difficulty):
        # print(self.proportions)
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        # ------------------------------------------- original code ------------------------------------------------
        if choice < self.proportions[0]:
            idx = 0
            if choice < self.proportions[0]/ 2:
                idx = 1
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # self.add_roughness(terrain)
        elif choice < self.proportions[2]:
            idx = 2
            if choice<self.proportions[1]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[4]:
            idx = 4
            if choice<self.proportions[3]:
                idx = 5
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[5]:
            idx = 6
            num_rectangles = 20
            rectangle_min_size = 0.5
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[6]:
            idx = 7
            stones_size = 1.5 - 1.2*difficulty
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
            half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            stepping_stones_terrain(terrain, stone_size=1.5-0.2*difficulty, stone_distance=0.0+0.4*difficulty, max_height=0.2*difficulty, platform_size=1.2)
            self.add_roughness(terrain)
        elif choice < self.proportions[7]:
            idx = 8
            # gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain)
        elif choice < self.proportions[8]:
            idx = 9
            self.add_roughness(terrain)
            # pass
        elif choice < self.proportions[9]:
            idx = 10
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[10]:
            idx = 11
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    # terrain.slope_vector = np.array([1, 0., 0]).astype(np.float32)
                    # return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            # terrain_utils.wall_terrain(terrain, height=1, start2center=0.7)
            # terrain_utils.tanh_terrain(terrain, height=1.0, start2center=0.7)
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        elif choice < self.proportions[11]:
            idx = 12
            # half platform terrain
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty )
            self.add_roughness(terrain, difficulty=1)
        elif choice < self.proportions[13]:
            idx = 13
            height = 0.1 + 0.3 * difficulty
            if choice < self.proportions[12]:
                idx = 14
                height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1., step_height=height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[14]:
            x_range = [-0.1, 0.1+0.3*difficulty]  # offset to stone_len
            y_range = [0.2, 0.3+0.1*difficulty]
            stone_len = [0.9 - 0.3*difficulty, 1 - 0.2*difficulty]#2 * round((0.6) / 2.0, 1)
            incline_height = 0.25*difficulty
            last_incline_height = incline_height + 0.1 - 0.1*difficulty
            parkour_terrain(terrain,
                            num_stones=self.num_goals - 2,
                            x_range=x_range, 
                            y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len,
                            stone_width=1.0, 
                            last_incline_height=last_incline_height,
                            pad_height=0,
                            pit_depth=[0.2, 1])
            idx = 15
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[15]:
            idx = 16
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.25*difficulty],
                                   pad_height=0,
                                   x_range=[1.2, 2.2],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.4, 0.8],
                                   )
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[16]:
            idx = 17
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.15*difficulty],
                                   pad_height=0,
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 1],
                                   flat=True
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[17]:
            idx = 18
            parkour_step_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   step_height=0.1 + 0.35*difficulty,
                                   x_range=[0.3,1.5],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.5, 1],
                                   pad_height=0,
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[18]:
            idx = 19
            parkour_gap_terrain(terrain,
                                num_gaps=self.num_goals - 2,
                                gap_size=0.1 + 0.7 * difficulty,
                                gap_depth=[0.2, 1],
                                pad_height=0,
                                x_range=[0.8, 1.5],
                                y_range=self.cfg.y_range,
                                half_valid_width=[0.6, 1.2],
                                # flat=True
                                )
            self.add_roughness(terrain)
        elif choice < self.proportions[19]:
            idx = 20
            demo_terrain(terrain)
            self.add_roughness(terrain)
    
        #------------------------------------------------------------original code------------------------------------------------
        # idx = 20
        # demo_terrain(terrain)
        # self.add_roughness(terrain)
        # self.add_roughness(terrain)
        # np.set_printoptions(precision=2)
        # print(np.array(self.proportions), choice)
        # print(self.length_per_env_pixels, self.width_per_env_pixels)
        # rand = self.rand # 
        # rand = np.random.choice(np.arange(100,140))
        if np.random.uniform(0, 1) < 0.3:
            terrain.height_field_raw = generate_heightfield(self.env_length, self.env_width , self.cfg.horizontal_scale, seed=np.random.randint(0,10000)) / self.cfg.vertical_scale
        # terrain.height_field_raw = self.diffusion_generate() / self.cfg.vertical_scale
        # visualize_heightmap(terrain.height_field_raw)
        # print(terrain.height_field_raw)
        # self.add_roughness(terrain)
        # print("1")
        # ----------------------------------------------------------------------------- manipulating terrain ------------------------------------------------
        # terrain.height_field_raw = self.generator.generate_height_map(self.length_per_env_pixels, self.width_per_env_pixels, difficulty=0)
        terrain.idx = idx
        # set_goals(terrain)
        # fix_desc = ""
        # fix_desc = fix_terrain(terrain)
        # visualize_heightmap(terrain.height_field_raw, title="Generated Heightmap", cmap="terrain")
        # if fix_desc != "":
        #     print(f"Automatically fixed terrain {terrain.idx}: {fix_desc}")
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row # terrain_level
        j = col # terrain_type
        print(f"Adding terrain {terrain.idx} to map at ({i}, {j})")
        # map coordinate system
        start_x = self.border + row * self.length_per_env_pixels
        end_x = self.border + (row + 1) * self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y = self.border + (col + 1) * self.width_per_env_pixels
        # coordinates to cut out of the existing heightmap
        
        if self.cfg.type == "benchmark":
            pass
        else:    
            grid_value = self.grid[row, col]
        if self.heightmap_exists(self.cfg.type, EXPTID):
            # print("loading subterrain")
            height_field = self.load_height_map(EXPTID, self.cfg.type)
            self.height_field_raw[start_x:end_x, start_y:end_y] = height_field[start_x:end_x, start_y:end_y]
        else:
            if self.cfg.type == "benchmark":
                # visualize_heightmap(terrain.height_field_raw, title="Generated Heightmap", cmap="terrain")
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw
                # visualize_heightmap(self.height_field_raw[start_x: end_x, start_y:end_y], title="Generated Heightmap", cmap="terrain")
            else:
                if int(grid_value) == int(self.cfg.type):
                    # print("generating new subterrain")
                    self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw
                else:
                    # print("loading subterrain")
                    col_to_copy = int(grid_value.split("_")[1])
                    row_to_copy = int(grid_value.split("_")[2])
                    # print(row_to_copy, col_to_copy)
                    # print(self.border, self.length_per_env_pixels, self.width_per_env_pixels)
                    copy_start_x = self.border + row_to_copy * self.length_per_env_pixels
                    copy_end_x = self.border + (row_to_copy + 1) * self.length_per_env_pixels
                    copy_start_y = self.border + col_to_copy * self.width_per_env_pixels
                    copy_end_y = self.border + (col_to_copy + 1) * self.width_per_env_pixels

                    # Load the heightmap from file
                    # print(grid_value.split("_")[0])
                    height_field = self.load_height_map(EXPTID, grid_value.split("_")[0])
                    # print(height_field.shape)
                    # Extract the partial heightfield for the given row and col
                    # height_field_ = height_field[copy_start_x:copy_end_x, copy_start_y:copy_end_y]
                    self.height_field_raw[start_x:end_x, start_y:end_y] = height_field[copy_start_x:copy_end_x, copy_start_y:copy_end_y]

        # visualize_heightmap(self.height_field_raw)

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]
        # self.env_slope_vec[i, j] = terrain.slope_vector
    
    def cut_out_subterrain(self,i,j,full_height_field):
        """
        Cut out a subterrain from the full height field based on the given indices.
        :param i: row index of the subterrain
        :param j: column index of the subterrain
        :param full_height_field: the full height field from which to cut out the subterrain
        :return: the cut-out subterrain
        """
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels

        return full_height_field[start_x:end_x, start_y:end_y]



def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def gap_parkour_terrain(terrain, difficulty, platform_size=2.):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -700
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    slope_angle = 0.1 + difficulty * 1
    offset = 1 + 9 * difficulty#10
    scale = 15
    wall_center_x = [center_x - x1, center_x, center_x + x1]
    wall_center_y = [center_y - y1, center_y, center_y + y1]

    # for i in range(center_y + y1, center_y + y2):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)
    
    # for i in range(center_y - y2, center_y - y1):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_x + x1, center_x + x2):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)
    
    # for i in range(center_x - x2, center_x - x1):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)

def parkour_terrain(terrain, 
                    platform_len=2.5, 
                    platform_height=0., 
                    num_stones=8, 
                    x_range=[1.8, 1.9], 
                    y_range=[0., 0.1], 
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones+2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    
    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len -  stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0
    
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2*(left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        
        goals[i+1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    
def parkour_gap_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_gaps=8,
                           gap_size=0.3,
                           x_range=[1.6, 2.4],
                           y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-700,
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    
    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, 
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_hurdle_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                           stone_len=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_step_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                        #    x_range=[1.5, 2.4],
                            x_range=[0.2, 0.4],
                           y_range=[-0.15, 0.15],
                           half_valid_width=[0.45, 0.5],
                           step_height = 0.2,
                           pad_width=0.1,
                           pad_height=0.5):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2
    
    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height
    
    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    
    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    
    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-1):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    def get_rand_dis_int(scale):
        return np.random.randint(int(- scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0

def fix_terrain(terrain):
    """Fix common errors with GPT-generated terrains"""
    # If goals are in units (indices), convert to meters
    # This doesn't count as a fix since we prompt GPT to return goals in units (for simplicity)
    env_length, env_width = terrain.width * terrain.horizontal_scale, terrain.length * terrain.horizontal_scale
    # if np.max(terrain.goals[:, 0]) > env_length or np.max(terrain.goals[:, 1]) > env_width:
    #     terrain.goals = terrain.goals.astype(np.float64) * terrain.horizontal_scale
    
    fix_descs = set()

    # min_terrain_height = np.min(terrain.height_field_raw)
    # if min_terrain_height < round(-3 / terrain.vertical_scale):
    #     terrain.height_field_raw[terrain.height_field_raw < -3] = round(-3 / terrain.vertical_scale)
    #     fix_descs.add(f"min terrain height {min_terrain_height} is below -3")


    # Fix goals that are unset or out of bounds
    def valid_goal(goal):
        return 0 < goal[0] < env_length and 0 < goal[1] < env_width  # We check > 0 since (0, 0) is the default
    num_goals_fixed = 0
    for i in range(1, len(terrain.goals)):
        if not valid_goal(terrain.goals[i]) and valid_goal(terrain.goals[i-1]):
            terrain.goals[i] = terrain.goals[i-1]
            num_goals_fixed += 1
    for i in range(len(terrain.goals) - 2, -1, -1):
        if not valid_goal(terrain.goals[i]) and valid_goal(terrain.goals[i+1]):
            terrain.goals[i] = terrain.goals[i+1]
            num_goals_fixed += 1
    if num_goals_fixed > 0:
        fix_descs.add(f"{num_goals_fixed} goal(s) out of bounds")
    assert num_goals_fixed <= round(len(terrain.goals) / 2), f'Fixed too many goals ({num_goals_fixed})!'
    for i in range(len(terrain.goals)):
        assert valid_goal(terrain.goals[i]), f'Goal {i} at ({terrain.goals[i, 0]}, {terrain.goals[i, 1]}) is invalid!'

    # Move goals away from edge
    clipped_goals_x = np.clip(terrain.goals[:, 0], a_min=0.5, a_max=(env_length - 0.5))
    clipped_goals_y = np.clip(terrain.goals[:, 1], a_min=0.5, a_max=(env_width - 0.5))
    if not np.allclose(clipped_goals_x, terrain.goals[:, 0]) or not np.allclose(clipped_goals_y, terrain.goals[:, 1]):
        fix_descs.add("goals too close to edge")
    terrain.goals[:, 0] = clipped_goals_x
    terrain.goals[:, 1] = clipped_goals_y

    # Check for sudden changes in height along the line connecting adjacent goals
    for i in range(len(terrain.goals) - 1):
        x1, y1 = terrain.goals[i]
        x2, y2 = terrain.goals[i + 1]
        x1_idx, y1_idx = int(x1 / terrain.horizontal_scale), int(y1 / terrain.horizontal_scale)
        x2_idx, y2_idx = int(x2 / terrain.horizontal_scale), int(y2 / terrain.horizontal_scale)
        # print(x1_idx, y1_idx, x2_idx, y2_idx)

        # Use Bresenham's line algorithm to get points along the line
        dx, dy = abs(x2_idx - x1_idx), abs(y2_idx - y1_idx)
        sx, sy = (dx/10.0 if x1_idx < x2_idx else -dx/10.0), (dy/10.0 if y1_idx < y2_idx else -dy/10.0)
        err = dx - dy

        x, y = x1_idx, y1_idx
        sudden_change_detected = False
        while True:
            r_x, r_y = round(x), round(y)
            if r_x == x2_idx and r_y == y2_idx:
                break
            if r_x < 0 or r_x >= terrain.height_field_raw.shape[0] or r_y < 0 or r_y >= terrain.height_field_raw.shape[1]:
                break
            if round(x+sx) < 0 or round(x+sx) >= terrain.height_field_raw.shape[0] or round(y+sy) < 0 or round(y+sy) >= terrain.height_field_raw.shape[1]:
                break
            if terrain.height_field_raw[r_x, r_y] < -0.5 and terrain.height_field_raw[r_x, r_y] > -2:
                if abs(terrain.height_field_raw[round(x), round(y)] - terrain.height_field_raw[round(x+sx), round(y+sy)]) > 0.5/terrain.vertical_scale:
                    sudden_change_detected = True
                    break
                # terrain.height_field_raw[x, y] = 0  # Fix sudden height change
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        # Merge goals if sudden change is detected
        if sudden_change_detected:
            terrain.goals[i] = [x2, y2]
            fix_descs.add("goal fallen in gap detected")

    

    # # Check and fix quadruped's spawn location
    # if np.max(terrain.height_field_raw[:round(2 / terrain.horizontal_scale), :]) > 0:
    #     terrain.height_field_raw[:round(2 / terrain.horizontal_scale), :] = 0
    #     fix_descs.add("spawn area not 0")
    # clipped_goals_x = np.clip(terrain.goals[:, 0], a_min=1.5, a_max=None)  # Move goals ahead of spawn
    # if not np.allclose(clipped_goals_x, terrain.goals[:, 0]):
    #     fix_descs.add("goals too close to spawn")
    # terrain.goals[:, 0] = clipped_goals_x

    # Check and fix small obstacles that have an extreme aspect ratio
    # This only works for axis-aligned obstacles, but the mistake is rare enough to not warrant a more complex fix
    min_terrain_height = np.min(terrain.height_field_raw)
    valid_ratio_threshold = 2
    min_obstacle_length, min_obstacle_width = 0.6 / terrain.horizontal_scale, 0.4 / terrain.horizontal_scale
    floodfill_dz_threshold = 1 / terrain.vertical_scale
    obstacles = {}
    floodfill = np.zeros_like(terrain.height_field_raw)

    def bfs(x, y, id):
        q = [(x, y)]
        while len(q) > 0:
            x, y = q.pop(0)
            if floodfill[x, y] != 0:
                continue
            floodfill[x, y] = id
            obstacles[id] = [
                (min(obstacles[id][0][0], x), min(obstacles[id][0][1], y)),
                (max(obstacles[id][1][0], x+1), max(obstacles[id][1][1], y+1))
            ]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < terrain.height_field_raw.shape[0] and 0 <= ny < terrain.height_field_raw.shape[1]:
                    if terrain.height_field_raw[nx, ny] != min_terrain_height and floodfill[nx, ny] == 0 and abs(terrain.height_field_raw[nx, ny] - terrain.height_field_raw[x, y]) < floodfill_dz_threshold:
                        q.append((nx, ny))
    obstacle_counter = 0
    for i in range(terrain.height_field_raw.shape[0]):
        for j in range(terrain.height_field_raw.shape[1]):
            if terrain.height_field_raw[i, j] != min_terrain_height and floodfill[i, j] == 0:
                obstacle_counter += 1
                obstacles[obstacle_counter] = [(i, j), (i, j)]
                bfs(i, j, obstacle_counter)
    
    for obstacle in obstacles:
        x1, y1 = obstacles[obstacle][0]
        x2, y2 = obstacles[obstacle][1]
        obstacle_length, obstacle_width = x2 - x1, y2 - y1
        if max(obstacle_length, obstacle_width) / min(obstacle_width, obstacle_length) < valid_ratio_threshold:
            continue
        
        if obstacle_length < min_obstacle_length and obstacle_width < min_obstacle_width:
            # Erase small obstacles
            terrain.height_field_raw[x1:x2, y1:y2] = 0
            fix_descs.add("obstacles length and width too small (erased)")
        if obstacle_length < min_obstacle_length:
            # Extend length on both sides
            extend_length = max(round((min_obstacle_length - obstacle_length) // 2), 1)
            nx1, nx2 = max(0, x1 - extend_length), min(terrain.height_field_raw.shape[0], x2 + extend_length)
            terrain.height_field_raw[nx1:x1, y1:y2] = terrain.height_field_raw[x1, y1:y2][None, :]
            terrain.height_field_raw[x2:nx2, y1:y2] = terrain.height_field_raw[x2-1, y1:y2][None, :]
            fix_descs.add("obstacles length too small")
        if obstacle_width < min_obstacle_width:
            # Extend width on both sides
            extend_width = max(round((min_obstacle_width - obstacle_width) // 2), 1)
            ny1, ny2 = max(0, y1 - extend_width), min(terrain.height_field_raw.shape[1], y2 + extend_width)
            terrain.height_field_raw[x1:x2, ny1:y1] = terrain.height_field_raw[x1:x2, y1][..., None]
            terrain.height_field_raw[x1:x2, y2:ny2] = terrain.height_field_raw[x1:x2, y2-1][..., None]
            fix_descs.add("obstacles width too small")
    
    return ", ".join(fix_descs)

def set_goals(terrain):
    """
    Set the goals according to the terrain
    """
    idx = [40, 70, 100, 130, 160, 190, 220, 250, 280, 300]
    # idx = [x + np.random.randint(-3, 4) for x in idx]

    for i in range(len(terrain.goals)):
        # print(terrain.height_field_raw.shape)
        # print(idx[i])
        array = terrain.height_field_raw[idx[i]-15:idx[i], 25:55]
        array_shape = array.shape
        # print(array_shape)
        idx_max = np.argmax(array)
        # print(idx_max)
        idx_max = np.unravel_index(idx_max, array_shape)
        # print(idx_max)
        idx_1 = idx_max[0] + idx[i]-15
        idx_2 = idx_max[1] + 25
        idx_x = idx_1 * terrain.horizontal_scale
        idx_y = idx_2 * terrain.horizontal_scale
        # print(idx_x, idx_y)
        idx_y = 2.0 + np.random.uniform(-0.3, 0.3)# (idx_y + 2) / 2.0
        # print(idx_x, idx_y)
        terrain.goals[i] = [idx_x, idx_y] 

    noncontinuous_terrain =  check_sudden_height_changes(terrain.height_field_raw, threshold=0.3, vertical_scale=terrain.vertical_scale)
    # print("Noncontinuous terrain:", noncontinuous_terrain)
    # print(check_negative_height(terrain.height_field_raw))
    # print("Pillar checking")
    if noncontinuous_terrain:
        if 15000 > check_negative_height(terrain.height_field_raw) > 3700:
            # print("Pillar detected")
            # print(terrain.height_field_raw.shape)
            pillar_centers = detect_pillars(terrain.height_field_raw[:,25:55], min_height=-0.37, max_height=0.2, min_size=15, vertical_scale=terrain.vertical_scale)
            # print("Pillar centers:", pillar_centers)
            # for centers in pillar_centers:
            #     centers[1] += 25

            num_of_pillars = len(pillar_centers)
            if num_of_pillars > 0:
                #sort pillars by x coordinate
                goal = []
                pillar_centers = sorted(pillar_centers, key=lambda x: x[0])
                # print(num_of_pillars)
                for i in range(num_of_pillars):
                    idx_x = pillar_centers[i][0] * terrain.horizontal_scale
                    idx_y = (pillar_centers[i][1]+25) * terrain.horizontal_scale
                    if len(goal) < 8:
                        goal.append([idx_x, idx_y])

                for goals in terrain.goals:
                    if goal[-1][0] < goals[0] and len(goal) < 8:
                        goal.append(goals)
                
                while len(goal) < 8:
                    goal.append(goal[-1])
                terrain.goals = np.array(goal)
                # print(terrain.goals)

    return 0

def check_for_ramps(np_array, vertical_scale, horizontal_scale):
    """
    Check if 1D numpy array has a ramp
    """
    threshold = 0.3
    min_length = 10

    for i in range(len(np_array) - min_length + 1):
        slopes = np.diff(np_array[i:i + min_length]) / (vertical_scale / horizontal_scale)
        if np.all(np.abs(np.diff(slopes)) <= 0.1) and np.all(np.abs(slopes) <= 1.5) and np.all(np.abs(slopes) >= 0.3):
            return i, True
        
    return None , False

def detect_pillars(heightmap, min_height=-0.5, max_height=0.1, min_size=10, vertical_scale=0.005):
    """
    Detect pillars in the heightmap that start from a negative height value and end near zero.

    Parameters:
        heightmap (np.ndarray): 2D array representing the heightmap.
        min_height (float): Minimum height threshold for pillar detection.
        max_height (float): Maximum height threshold for pillar detection.
        min_size (int): Minimum size of a connected region to be considered a pillar.
        vertical_scale (float): Scale factor for height values.

    Returns:
        List[Tuple[int, int]]: List of (row, col) indices of the middle points of the pillars.
    """
    heightmap_smooth = heightmap.copy()
    # heightmap_smooth = gaussian_filter(heightmap_smooth, sigma=0.7)

    start_x = 0
    end_x = heightmap_smooth.shape[0]
    for i in range(heightmap_smooth.shape[0]):
        if np.sum(heightmap_smooth[i, :]*vertical_scale < -0.35) > 10:
            # print(i, np.sum(heightmap_smooth[i, :]*vertical_scale < -0.35))
            start_x = i
            break
    for i in range(heightmap_smooth.shape[0]-1, -1, -1):
        if np.sum(heightmap_smooth[i, :]*vertical_scale < -0.35) > 10:
            # print(i, np.sum(heightmap_smooth[i, :]*vertical_scale < -0.35))
            end_x = i
            break
    
    if start_x >= end_x-50:
        # print("No pillars detected")
        return []
    heightmap_smooth = heightmap_smooth[start_x:end_x, :]*1.3
    # Create a binary mask for the heightmap
    binary_mask = np.zeros_like(heightmap_smooth, dtype=np.uint8)
    binary_mask[(heightmap_smooth * vertical_scale > min_height) & (heightmap_smooth * vertical_scale < max_height)] = 1

    labeled_array, num_features = label(binary_mask)

    # Check if there is a region with large height difference in the labeled array
    for region_id in range(1, num_features + 1):
        region_mask = (labeled_array == region_id)
        region_heights = heightmap_smooth[region_mask]

        # Check mean of areas close to and far from the midline within the region
        midline = heightmap_smooth.shape[1] // 2

        # Apply the region_mask to the close and far regions
        close_region_mask = region_mask & (np.abs(np.arange(heightmap_smooth.shape[1]) - midline) < 4)
        far_region_mask = region_mask & ((np.arange(heightmap_smooth.shape[1]) < midline - 8) | 
                                         (np.arange(heightmap_smooth.shape[1]) > midline + 8))

        close_mean = np.mean(heightmap_smooth[close_region_mask])
        far_mean = np.mean(heightmap_smooth[far_region_mask])

        close_region_mask = region_mask & (np.abs(np.arange(heightmap_smooth.shape[1]) - midline) < 7)

        if close_mean < far_mean - 0.2 / vertical_scale:
            # print(f"Significant mean difference detected in region {region_id}: close_mean={close_mean}, far_mean={far_mean}")
            binary_mask[close_region_mask] = 0
    

    labeled_array, num_features = label(binary_mask)

    centers = []
    for region_id in range(1, num_features + 1):
        if np.sum(labeled_array == region_id) >= min_size:
            # center = center_of_mass(binary_mask, labeled_array, region_id)
            region_coords = np.argwhere(labeled_array == region_id)
            center = tuple(map(int, np.round(region_coords.mean(axis=0))))
            if heightmap_smooth[int(round(center[0])), int(round(center[1]))] * vertical_scale < -0.1:
                binary_mask[labeled_array == region_id] = 0
                # print(f"Erased region {region_id} due to low center height: {heightmap_smooth[center[0], center[1]] * vertical_scale}")
            else:
                # print(f"Region {region_id} center: {center}")
                centers.append((int(round(center[0]))+start_x, int(round(center[1]))))

    return centers

def check_sudden_height_changes(height_field, threshold=0.5, vertical_scale=0.1):
    """
    Check for sudden height changes in the height field.

    Parameters:
        height_field (np.ndarray): 2D array representing the height field.
        threshold (float): Maximum allowed height difference between adjacent cells.

    Returns:
        List[Tuple[int, int]]: List of coordinates where sudden height changes occur.
    """
    sudden_changes = []
    sudden_change = False
    rows, cols = height_field.shape

    for i in range(rows - 1):
        for j in range(cols - 1):
            # Check height differences with right and bottom neighbors
            if abs(height_field[i, j] - height_field[i, j + 1]) > threshold / vertical_scale and height_field[i, j] < 0:
                sudden_changes.append((i, j))
            if abs(height_field[i, j] - height_field[i + 1, j]) > threshold / vertical_scale and height_field[i, j] < 0:
                sudden_changes.append((i, j))

    if len(sudden_changes) > 5:
        sudden_change = True
        # print("Sudden height changes detected at:", sudden_changes)
    return sudden_change

def check_negative_height(height_field, threshold=-0.4):
    """
    Check for negative height values in the height field.

    Parameters:
        height_field (np.ndarray): 2D array representing the height field.
        threshold (float): Minimum allowed height value.

    Returns:
        int: number of negative height values found
    """
    # print(np.sum(height_field<threshold))
    vertical_scale = 0.005
    return np.sum(height_field*vertical_scale < threshold)

def save_image_with_goals(
    image,
    goals,
    idx,
    save_dir,
    goal_radius=10,
    goal_color=(255, 0, 0),
    goal_thickness=2,
):
    """
    Save an image with goals drawn on it.

    Args:
        image (np.ndarray): The image to draw on.
        goals (np.ndarray): The goals to draw.
        goal_radius (float): The radius of the goals.
        goal_color (tuple): The color of the goals.
        goal_thickness (int): The thickness of the goal lines.
    """
    for goal in goals:
        goal = goal * 20
        x, y = int(goal[0]), int(goal[1])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if (i - x) ** 2 + (j - y) ** 2 <= goal_radius ** 2:
                    image[i, j] = 50  # Assuming single-channel grayscale image

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, f"image{idx}.npy")
    np.save(filename, image)