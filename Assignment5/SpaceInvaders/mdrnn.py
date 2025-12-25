import torch
import torch.nn as nn

class MDRNN(nn.Module):
    def __init__(self, action_dim, hidden_size=256, latent_dim=128):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim + action_dim, hidden_size)
        self.fc_z = nn.Linear(hidden_size, latent_dim)
        self.fc_r = nn.Linear(hidden_size, 1)
        self.fc_done = nn.Linear(hidden_size, 1)

    def forward(self, z, a, h):
        x = torch.cat([z, a], dim=-1).unsqueeze(0)
        out, h = self.rnn(x, h)
        out = out.squeeze(0)
        z_next = self.fc_z(out)
        r = self.fc_r(out)
        done = torch.sigmoid(self.fc_done(out))
        return z_next, r, done, h
