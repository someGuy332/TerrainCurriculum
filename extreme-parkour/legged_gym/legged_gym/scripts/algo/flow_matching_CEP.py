from omegaconf import DictConfig
from abc import *
import numpy as np
import torch
import torch.nn as nn
import copy
from .vae import VAE
import time

from .utils import BasicTimeScheduler
from .models.DiT_2D import DiT2D

torch.manual_seed(int(time.time()*1000))

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class EnergyNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels+1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, 1)

        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"Warning: Parameter {name} has requires_grad=False")
                param.requires_grad_(True)
            # print(f"Parameter {name} requires_grad: {param.requires_grad}")
        
    
    def forward(self, x, t):
        # x: (batch_size, dim, height, width)
        t_dim = t.float().view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))  # Convert t to float and expand to match spatial dimensions
        x = torch.cat((x, t_dim), dim=1)  # Concatenate t as an additional channel
        x = self.conv(x)  # Conv2d
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Linear
        return x
    
# class EnergyNetwork(nn.Module):
#     def __init__(self, in_channels=1):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Linear(in_channels, 64),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.AdaptiveAvgPool2d((1,1)),
#         )
#         self.fc = nn.Linear(32, 1)

#     def forward(self, x):
#         # x: (batch_size, dim, height, width)
#         x = x.permute(0,2,3,1)
#         # x = self.conv(x)  # Conv2d
#         x = self.conv[0:2](x)
#         x = x.permute(0,3,1,2)

#         x = self.conv[2](x)
#         x = x.permute(0,2,3,1)

#         x = self.conv[3:5](x)
#         x = x.permute(0,3,1,2)
        
#         x = self.conv[5](x)
#         x = self.conv[6](x)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.fc(x)  # Linear
#         return x


class FlowMatchingCEP(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.debug = self.cfg.debug
        
        self.device = cfg.device

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

        self.data_shape = (
            cfg.tokenize_dim,
            cfg.input_h // cfg.tokenize_scale,
            cfg.input_w // cfg.tokenize_scale,
        )

        self.vae = VAE()
        self.vae_ckpt = torch.load('/home/yoonho/Workspace/RLLab/Multiverse/extreme-parkour/legged_gym/legged_gym/scripts/gen_models/vae/vae.pt', weights_only=True)
        self.vae.load_state_dict(self.vae_ckpt, strict=False)
        self.vae.eval().to(cfg.device)

        self.energy_network = EnergyNetwork(cfg.tokenize_dim)
        # Initialize the weights of the energy network
        self.energy_network.apply(initialize_weights)
        self.energy_network.to(self.device)
        
        self.energy_optimizer = torch.optim.AdamW(
            self.energy_network.parameters(), lr=cfg.energy_lr, weight_decay=cfg.energy_weight_decay
        )

        for name, param in self.energy_network.named_parameters():
            param.requires_grad_(True)

        self.time_scheduler = BasicTimeScheduler(cfg)
        self.denoising_step = cfg.denoising_step

        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        self.lamda_guidance = cfg.lamda_guidance

    def get_target(self, x_1, t):
        t_full = (
            t.view(tuple(t.shape) + tuple([1] * (x_1.ndim - t.ndim)))
            / self.denoising_step
        )

        x_0 = torch.randn_like(x_1)
        x_t = x_1 * t_full + x_0 * (1 - (1 - 1e-5) * t_full)
        v_t = x_1 - (1 - 1e-5) * x_0

        return x_t, v_t

    def compute_energy_gradient(self, x_t, t):
        x_t = x_t.clone().detach().requires_grad_(True)  # Clone and set requires_grad=True
        
        grad_x = torch.autograd.grad(self.energy_network(x_t, t/self.denoising_step), x_t, create_graph=True, allow_unused=True)[0] # dimension of grad_x : (batch_size, 1, height, width)
        if grad_x is None:
            print("grad_x is None. x_t might not be connected to the computation graph.")
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
    
    def update_energy(self, x, energy_guide):
        # x: (batch_size, 1, input_height, input_width)
        # max_rewards: (batch_size,)
        # mean_rewards: (batch_size,)
        with torch.no_grad():
            latent_vectors = self.vae.encode(x)
            t = self.time_scheduler.sample(latent_vectors.shape[0]).to(self.device)
            x_t, _ = self.get_target(latent_vectors, t)

        energy_target = energy_guide
        energy_pred = self.energy_network(x_t,t/self.denoising_step).squeeze(-1)
        loss = self.mse_loss(energy_pred, energy_target)
        self.energy_optimizer.zero_grad()
        loss.backward()
        self.energy_optimizer.step()
        return loss.item()


    def compute_energy_loss(self, data, labels):
        """
        Compute the loss for the energy network.
        This is a placeholder implementation and should be replaced with the actual loss logic.
        """
        predictions = self.energy_network(data)
        target = labels  # Replace with actual target values
        print(labels)
        print(predictions)
        loss = torch.nn.functional.mse_loss(predictions, target)
        return loss

    def sample(self, batch_size, step_size=1):
        """
        sample data from guassian noise, without external conditions or history
        """
        assert self.denoising_step % step_size == 0
        self.time_scheduler.reset()
        torch.manual_seed(int(time.time()*1000))
        x_t = torch.randn(torch.Size([batch_size, *self.data_shape]), requires_grad=True)
        x_t = x_t.to(self.device)
        x_t.requires_grad_(True)  # Ensure requires_grad is set to True
        done = False

        while not done:
            t = self.time_scheduler.current_timestep
            t = t * torch.ones(batch_size, dtype=int).to(self.device)
        
            with torch.no_grad():
                v_t = self.ema_model(x_t,t)

            # Compute energy gradient
            grad_x = self.compute_energy_gradient(x_t,t)
            x_t += (v_t + self.lamda_guidance * grad_x) * (step_size / self.denoising_step)
            done = self.time_scheduler.step(step_size)

        print(f"Energy of Terrain: {self.energy_network(x_t,t/self.denoising_step)}")

        return x_t
