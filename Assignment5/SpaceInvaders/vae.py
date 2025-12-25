import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, z_dim=128, input_channels=3, input_size=84):
        super().__init__()
        self.z_dim = z_dim
        self.input_channels = input_channels
        self.input_size = input_size

        # ---------------- Encoder ----------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # -> 32 x 42 x 42
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 21 x 21
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 128 x 10 x 10
            nn.ReLU(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            enc_out = self.encoder_conv(dummy)
            self.flattened_size = enc_out.numel() // dummy.shape[0]
            self.unflatten_shape = enc_out.shape[1:]  # (C,H,W)

        self.fc_mu = nn.Linear(self.flattened_size, z_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, z_dim)

        # ---------------- Decoder ----------------
        self.decoder_fc = nn.Linear(z_dim, self.flattened_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(self.unflatten_shape[0], 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 20 x 20
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 32 x 40 x 40
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> 3 x 84 x 84
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), *self.unflatten_shape)
        x_recon = self.decoder_conv(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon, x, reduction="sum")
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld
