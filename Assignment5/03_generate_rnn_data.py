import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import numpy as np
import cv2
import os

# --- 1. Define VAE & Wrapper (Must match previous step) ---
class WorldModelWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

    def observation(self, obs):
        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
        obs = np.moveaxis(obs, -1, 0)
        return obs

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
        # Decoder (not used here, but needed to load weights)
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

# --- 2. Generation Logic ---
def generate_rnn_data(num_rollouts=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating data using {device}...")

    # Load VAE
    vae = VAE().to(device)
    vae.load_state_dict(torch.load("vae.pth", map_location=device, weights_only=True))
    vae.eval()

    # Setup Env
    gym.register_envs(ale_py)
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = WorldModelWrapper(env)

    # Storage
    # We will store lists of arrays, then concatenate
    obs_list = []
    action_list = []
    done_list = []

    print(f"Collecting {num_rollouts} episodes...")
    
    with torch.no_grad():
        for i in range(num_rollouts):
            obs, _ = env.reset()
            done = False
            
            while not done:
                # 1. Process Frame -> VAE -> Latent z
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device) / 255.0
                mu, logvar = vae.encode(obs_tensor)
                z = vae.reparameterize(mu, logvar) # (1, 32)
                
                # 2. Random Action
                action = env.action_space.sample()
                
                # 3. Store
                obs_list.append(z.cpu().numpy()) # Store z, not image
                action_list.append(action)
                
                # 4. Step
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store done flag (shifted: this action caused this done state)
                done_list.append(done)

            if (i+1) % 100 == 0:
                print(f"Episode {i+1}/{num_rollouts} done.")

    # Convert to numpy
    obs_data = np.concatenate(obs_list, axis=0) # (Total_Steps, 32)
    action_data = np.array(action_list)         # (Total_Steps, )
    done_data = np.array(done_list)             # (Total_Steps, )

    print(f"Final dataset shape: {obs_data.shape}")
    np.savez("rnn_data.npz", obs=obs_data, action=action_data, done=done_data)
    print("Saved to rnn_data.npz")

if __name__ == "__main__":
    generate_rnn_data()