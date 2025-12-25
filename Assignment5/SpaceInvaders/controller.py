import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, action_dim, hidden_size=128, latent_dim=128, rnn_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + rnn_hidden, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, z, h):
        h_flat = h[0].squeeze(0) if h is not None else torch.zeros((z.size(0), 256), device=z.device)
        return self.net(torch.cat([z, h_flat], dim=-1))
