import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os
from omegaconf import OmegaConf
from legged_gym.scripts.algo.flow_matching_CEP import FlowMatchingCEP
import copy

import random

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
    "energy_weight_decay": 5e-3,
    "lamda_guidance": 10,
}
diffusion_cfg = OmegaConf.create(diffusion_arg)
flow_matching = FlowMatchingCEP(diffusion_cfg)
flow_matching.model.load_state_dict(torch.load(flow_model_path, weights_only=True), strict=False)
flow_matching.ema_model = copy.deepcopy(flow_matching.model)
flow_matching.ema_model.to("cuda")
energy_ckpt = torch.load(energy_model_path, weights_only=True)
flow_matching.energy_network.load_state_dict(energy_ckpt, strict=False)
flow_matching.energy_network.to("cuda")
flow_matching.model.eval()
flow_matching.ema_model.eval()

# Check if the model exists at path
if os.path.exists(energy_model_path):
    # Load the model
    energy_ckpt = torch.load(energy_model_path, weights_only=True)
    flow_matching.energy_network.load_state_dict(energy_ckpt, strict=False)
    flow_matching.energy_network.to("cuda").eval()

def sample(random_seed, step_size=diffusion_cfg.eval_dt):
    flow_matching.time_scheduler.reset()
    torch.manual_seed(random_seed)
    x_t = torch.randn(torch.Size([1, *flow_matching.data_shape])).to("cuda")

    done = False
    while not done:
        t = flow_matching.time_scheduler.current_timestep
        t = t * torch.ones(1, dtype=int).to("cuda")

        with torch.no_grad():
            v_t = flow_matching.ema_model(x_t,t)
        grad_x = flow_matching.compute_energy_gradient(x_t)
        x_t += (v_t + flow_matching.lamda_guidance * grad_x) * (step_size / flow_matching.denoising_step)
        done = flow_matching.time_scheduler.step(step_size)

    return x_t

random_seed_arr = [random.randint(0, 10000) for _ in range(10)]

z_t = []
x_t = []

for i, random_seed in enumerate(random_seed_arr):
    z_t.append(sample(random_seed))
    print(flow_matching.energy_network(z_t[i]).detach().cpu().numpy())
    hf = flow_matching.vae.decode(z_t[i]).detach().cpu().numpy()
    hf = hf.transpose(0, 2, 3, 1)
    hf = hf.reshape((-1, hf.shape[2], hf.shape[3]))
    x_t.append(hf[3:363,:,0])

# Visualize the heightmaps
for i, heightmap in enumerate(x_t):
    plt.figure(figsize=(8, 6))
    plt.imshow(heightmap, cmap="terrain", interpolation="nearest")
    plt.colorbar(label="Height")
    plt.title(f"Heightmap Visualization {i+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)  # Optional: turn off grid for cleaner look
    plt.show()
    