import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import numpy as np
import cv2
import os

# 1. Image Preprocessing Wrapper
class WorldModelWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # World Models typically uses 64x64 input
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

    def observation(self, obs):
        # Resize to 64x64
        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
        # Reorder to Channel-First (C, H, W) for PyTorch
        obs = np.moveaxis(obs, -1, 0) 
        return obs

# 2. VAE Architecture (The Vision Model)
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 3x64x64 -> 32
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        
        # Flatten size: 256 * 2 * 2 = 1024
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # Decoder: 32 -> 3x64x64
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
        return torch.sigmoid(self.dec_conv4(h)) # Output 0-1

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 3. Data Collection Function
def collect_random_rollouts(num_rollouts=100, save_dir="data"):
    gym.register_envs(ale_py)
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = WorldModelWrapper(env)
    
    os.makedirs(save_dir, exist_ok=True)
    all_obs = []
    
    print(f"Collecting {num_rollouts} episodes...")
    for i in range(num_rollouts):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample() # Random policy
            obs, _, terminated, truncated, _ = env.step(action)
            all_obs.append(obs)
            done = terminated or truncated
            
        if (i+1) % 10 == 0:
            print(f"Episode {i+1}/{num_rollouts} done.")

    # Save as numpy array
    all_obs = np.array(all_obs)
    print(f"Saving {len(all_obs)} frames...")
    np.save(os.path.join(save_dir, "rollout_obs.npy"), all_obs)
    print("Data collection complete.")

if __name__ == "__main__":
    collect_random_rollouts(num_rollouts=50) # Generates ~10k-15k frames