import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# --- Re-define VAE (Same as before) ---
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 1024)
        self.dec_conv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 1024, 1, 1)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        return torch.sigmoid(self.dec_conv4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- Training Logic ---
def train_vae(data_path="data/rollout_obs.npy", epochs=20, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # Load data
    print("Loading data...")
    data = np.load(data_path)
    # Convert to Tensor, but keep on CPU for now to save GPU VRAM
    # We will normalize (divide by 255) inside the loop
    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for i, (imgs,) in enumerate(loader):
            # Move to device and normalize
            imgs = imgs.float().to(device) / 255.0

            # Forward pass
            recon_imgs, mu, logvar = vae(imgs)

            # Loss Calculation
            # 1. Reconstruction Loss (MSE)
            recon_loss = F.mse_loss(recon_imgs, imgs, reduction='sum')
            # 2. KL Divergence (Regularization)
            # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kld_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(data):.4f}")

    # Save Model
    torch.save(vae.state_dict(), "vae.pth")
    print("VAE saved to vae.pth")

if __name__ == "__main__":
    train_vae()