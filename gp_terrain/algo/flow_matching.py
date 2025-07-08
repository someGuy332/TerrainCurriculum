from omegaconf import DictConfig
from abc import *
import numpy as np
import torch
import copy

from .utils import BasicTimeScheduler
from .models.DiT_2D import DiT2D

class FlowMatching(ABC):
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


    def get_target(self, x_1, t):
        t_full = (
            t.view(tuple(t.shape) + tuple([1] * (x_1.ndim - t.ndim)))
            / self.denoising_step
        )

        x_0 = torch.randn_like(x_1)
        x_t = x_1 * t_full + x_0 * (1 - (1 - 1e-5) * t_full)
        v_t = x_1 - (1 - 1e-5) * x_0

        return x_t, v_t

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
                x_t += self.ema_model(x_t, t) * (step_size / self.denoising_step)
            done = self.time_scheduler.step(step_size)

        return x_t
