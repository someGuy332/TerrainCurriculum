import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os
from omegaconf import OmegaConf
from legged_gym.scripts.algo.flow_matching_CEP import FlowMatchingCEP
import copy

import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

energy_dir = Path(os.getcwd()) / "gen_models" / "energy_network"
energy_model_path = os.path.join(energy_dir, "energy_mnist.pt")
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
    "tokenize_dim": 1,
    "ema_decay": 0.99,
    "denoising_step": 128,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "energy_lr": 1e-4,
    "energy_weight_decay": 1e-4,
    "lamda_guidance": 5,
}
diffusion_cfg = OmegaConf.create(diffusion_arg)
flow_matching = FlowMatchingCEP(diffusion_cfg)

# # Load MNIST dataset and split into training and validation sets
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Resize MNIST images to 32x32 with 1 channel
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset: 90% for training, 10% for validation
train_size = int(0.9 * len(mnist_dataset))
val_size = len(mnist_dataset) - train_size
mnist_train, mnist_val = torch.utils.data.random_split(mnist_dataset, [train_size, val_size])

train_loader = DataLoader(mnist_train, batch_size=diffusion_cfg.eval_batch_size, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=diffusion_cfg.eval_batch_size, shuffle=False)

flow_matching.energy_network.eval()
total_loss = 0
with torch.no_grad():
    for data, labels in val_loader:
        data = data.to(diffusion_cfg.device)  # Flatten the images
        labels = (labels/10.0).to(diffusion_cfg.device).float()
        loss = flow_matching.compute_energy_loss(data, labels)  # Assuming a loss computation method exists
        total_loss += loss.item()

average_loss = total_loss / len(val_loader)
print(f"Validation Loss: {average_loss}")

# Train the energy network
epochs = 100
for epoch in range(epochs):
    flow_matching.energy_network.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(diffusion_cfg.device)  # Flatten the images
        labels = (labels/10).to(diffusion_cfg.device).float()
        flow_matching.update_energy(data, labels)  # Use the existing update function

    print(f"Epoch {epoch + 1}/{epochs} completed")

# Evaluate the energy network on the validation set
flow_matching.energy_network.eval()
total_loss = 0
with torch.no_grad():
    for data, labels in val_loader:
        data = data.to(diffusion_cfg.device)  # Flatten the images
        labels = (labels/10).to(diffusion_cfg.device).float()
        loss = flow_matching.compute_energy_loss(data, labels)  # Assuming a loss computation method exists
        total_loss += loss.item()

average_loss = total_loss / len(val_loader)
print(f"Validation Loss: {average_loss}")

# Save the trained energy network
torch.save(flow_matching.energy_network.state_dict(), energy_model_path)
print(f"Energy network saved to {energy_model_path}")