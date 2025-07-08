import torch
import torch.nn as nn

from .models.autoencoder import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 32
        self.spatial_compression = 8
        self.encoder = Encoder(
            in_channels=1,
            channels=64,
            channels_mult=[1, 2, 2,],
            num_res_blocks=2,
            attn_resolutions=[10],
            dropout=0.0,
            resolution=80,
            z_channels=2 * self.latent_dim,
            spatial_compression=self.spatial_compression,
            patch_size=2,
        )

        self.decoder = Decoder(
            out_channels=1,
            channels=64,
            channels_mult=[1, 2, 2,],
            num_res_blocks=2,
            attn_resolutions=[10],
            dropout=0.0,
            resolution=80,
            z_channels=self.latent_dim,
            spatial_compression=self.spatial_compression,
            patch_size=2,
        )

        self.mse_loss = nn.MSELoss()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        return mu + torch.randn_like(std) * std

    def encode(self, x, reparam=True):
        z = self.encoder(x)
        mu = z[:, : self.latent_dim]
        logvar = z[:, self.latent_dim :]

        if reparam:
            return self.reparameterize(mu, logvar)

        return mu, logvar

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x, reparam=False)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar