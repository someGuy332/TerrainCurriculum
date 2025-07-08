from omegaconf import DictConfig
from abc import *
import numpy as np
import torch
import torch.nn as nn
import copy
from .vae import VAE

from .utils import BasicTimeScheduler
from .models.DiT_2D import DiT2D

class EnergyNetwork(nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, 1)
        
    
    def forward(self, x):
        # x is of shape (batch_size, dim, height, width)
        batch_size, dim, height, width = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, 1, height * dim, width)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FlowMatchingCEP(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.debug = self.cfg.debug

        self.model = DiT2D(
            cfg.input_h // cfg.tokenize_scale,
            cfg.input_w // cfg.tokenize_scale,
            2,
            cfg.tokenize_dim,
            cfg.hidden_size,
            cfg.depth, 
            cfg.num_heads,
        )
        self.ema_model = copy.deepcopy(self.model)
        self.ema_decay = cfg.ema_decay
        self.model.to(cfg.device)
        self.ema_model.to(cfg.device)

        self.energy_network = EnergyNetwork(
            height=cfg.input_h // cfg.tokenize_scale,
            width=cfg.input_w // cfg.tokenize_scale,
            dim = cfg.tokenize_dim,
        ).to(cfg.device)
        self.energy_optimizer = torch.optim.AdamW(
            self.energy_network.parameters(), lr=cfg.energy_lr, weight_decay=cfg.energy_weight_decay
        )

        self.vae = VAE()
        self.vae_ckpt = torch.load('/home/yoonho/Workspace/RLLab/Multiverse/gp_terrain/results/vae/ckpts/vae.pt', weights_only=True)
        self.vae.load_state_dict(self.vae_ckpt['state_dict'], strict=False)
        self.vae.eval().to(cfg.device)

        self.time_scheduler = BasicTimeScheduler(cfg)
        self.denoising_step = cfg.denoising_step

        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.device = cfg.device
        
        self.data_shape = (
            cfg.tokenize_dim,
            cfg.input_h // cfg.tokenize_scale,
            cfg.input_w // cfg.tokenize_scale,
        )

        self.lamda_guidance = cfg.lamda_guidance
        self.num_negatives = cfg.num_negatives



    def get_target(self, x_1, t):
        t_full = (
            t.view(tuple(t.shape) + tuple([1] * (x_1.ndim - t.ndim)))
            / self.denoising_step
        )

        x_0 = torch.randn_like(x_1)
        x_t = x_1 * t_full + x_0 * (1 - (1 - 1e-5) * t_full)
        v_t = x_1 - (1 - 1e-5) * x_0

        return x_t, v_t

    def compute_energy_gradient(self, x_t): 
        x_t.requires_grad_(True) # (batch_size, 1, height, width)
        energy = self.energy_network(x_t).sum()
        grad_x = torch.autograd.grad(energy, x_t, create_graph=True)[0] # dimension of grad_x : (batch_size, 1, height, width)
        return grad_x

    def update(self, x_1):
        # print(tuple(x_1.shape[1:]))
        # print(self.data_shape)
        assert tuple(x_1.shape[1:]) == self.data_shape

        t = self.time_scheduler.sample(x_1.shape[0]).to(self.device)
        x_t, v_t = self.get_target(x_1, t)
        v_t_pred = self.model(x_t, t)

        loss = self.mse_loss(v_t, v_t_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for param, ema_param in zip(
                self.model.parameters(), self.ema_model.parameters()
            ):
                ema_param.mul_(self.ema_decay).add_(param, alpha=1 - self.ema_decay)

        return loss.item()
    
    def update_energy(self, x, max_rewards, mean_rewards):
        # x: (batch_size, 1, input_height, input_width)
        # max_rewards: (batch_size,)
        # mean_rewards: (batch_size,)
        with torch.no_grad():
            latent_vectors = self.vae.encode(x)
        energy_target = max_rewards - mean_rewards
        energy_pred = self.energy_network(latent_vectors).squeeze(-1)
        loss = self.mse_loss(energy_pred, energy_target)
        self.energy_optimizer.zero_grad()
        loss.backward()
        self.energy_optimizer.step()
        return loss.item()


    def sample(self, batch_size, step_size=1):
        """
        sample data from guassian noise, without external conditions or history
        """
        assert self.denoising_step % step_size == 0
        self.time_scheduler.reset()

        x_t = torch.randn(torch.Size([batch_size, *self.data_shape])).to(self.device)
        done = False

        while not done:
            t = self.time_scheduler.current_timestep
            t = t * torch.ones(batch_size, dtype=int).to(self.device)

            with torch.no_grad():
                v_t = self.ema_model(x_t,t)
                x_t += (v_t + self.lamda_guidance * self.compute_energy_gradient(x_t)) * (step_size / self.denoising_step)
            done = self.time_scheduler.step(step_size)

        return x_t
