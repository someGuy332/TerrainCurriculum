import torch
from torch.utils.data import Dataset
import os
import numpy as np
import h5py


# def build_dataset(cfg):
#     entire_terrains = np.load(cfg.dataset_path)
#     # entire_terrains = entire_terrains.transpose((0,3,1,2)) # N C H W
#     num_data = entire_terrains.shape[0]

#     perm_idxs = np.random.permutation(num_data)
#     val_idxs = perm_idxs[:num_data//20]
#     train_idxs = perm_idxs[num_data//20 :]

#     val_dataset = TerrainDataset(entire_terrains[val_idxs], device=cfg.device)
#     train_dataset = TerrainDataset(entire_terrains[train_idxs], device=cfg.device)

#     return train_dataset, val_dataset

def build_dataset(cfg):
    # List all .npy files in the dataset_path directory
    all_files = [os.path.join(cfg.dataset_path, f) for f in os.listdir(cfg.dataset_path) if f.endswith('.npy')]
    num_data = len(all_files)

    # Shuffle and split into train and validation sets
    perm_idxs = np.random.permutation(num_data)
    val_idxs = perm_idxs[:num_data // 20]
    train_idxs = perm_idxs[num_data // 20:]

    val_files = [all_files[i] for i in val_idxs]
    train_files = [all_files[i] for i in train_idxs]

    val_dataset = TerrainDataset(val_files, device=cfg.device)
    train_dataset = TerrainDataset(train_files, device=cfg.device)

    return train_dataset, val_dataset


class TerrainDataset(Dataset):
    def __init__(self, terrains_paths, device):
        """
        Args:
            file_path (str): Path to the .pt file containing the images
        """
        self.terrain_files = terrains_paths

        self.device = device

    def __len__(self):
        return len(self.terrain_files)

    def __getitem__(self, idx):
        """
        Returns:
            A dictionary (or tuple) with:
              - terrain: torch.Tensor shape [1, H, W]
        """
        terrain = self.terrain_files[idx]
        terrain_np = np.load(terrain)
        # print(terrain_np.shape)
        if terrain_np.ndim == 2:
            terrain_np = np.expand_dims(terrain_np, axis=-1)
        # print(terrain_np.shape)
        terrain_np = terrain_np.transpose((2, 0, 1))
        terrain_tensor = torch.from_numpy(terrain_np).to(torch.float32).to(self.device)
        
        return terrain_tensor
