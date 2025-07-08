import torch
from omegaconf import DictConfig

class BasicTimeScheduler:
    def __init__(self, cfg: DictConfig):
        self.denoising_step = cfg.denoising_step

        self.current_timestep = 0

    def sample(self, batch_size):

        return torch.randint(self.denoising_step, (batch_size,))

    def reset(self):
        self.current_timestep = 0

    def step(self, dt):
        self.current_timestep += dt
        done = False
        if self.current_timestep >= self.denoising_step:
            done = True

        return done
