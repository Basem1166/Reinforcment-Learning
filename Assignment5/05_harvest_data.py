import torch
import gymnasium as gym
import ale_py
import numpy as np
import cv2
import os

# --- Load Architectures ---
# (Paste VAE, RobustRNN, Controller classes here or import them)
from train_agent import VAE, RobustRNN, Controller, LatentBlockWrapper

def harvest():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Harvesting data on {device}...")

    # Load Models
    vae = VAE().to(device)
    vae.load_state_dict(torch.load("vae.pth", map_location=device, weights_only=True))
    
    rnn = RobustRNN().to(device)
    rnn.load_state_dict(torch.load("rnn.pth", map_location=device, weights_only=True))
    
    # Load YOUR BEST Controller (from the CMA run)
    # The script saves 'controller_cma.pth' periodically.
    controller = Controller(32+256, 4).to("cpu") 
    try:
        controller.load_state_dict(torch.load("controller_cma.pth", map_location="cpu", weights_only=True))
        print("Loaded current best controller.")
    except:
        print("Could not load controller_cma.pth! Make sure it exists.")
        return

    # 2. Prepare Environment
    gym.register_envs(ale_py)
    base_env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = LatentBlockWrapper(base_env, vae, rnn, device)

    # 3. Data Buffers
    obs_data = []
    action_data = []
    
    MAX_EPISODES = 500
    print(f"Collecting {MAX_EPISODES} episodes of 'Pro' data...")

    for i in range(MAX_EPISODES):
        obs, _ = env.reset()
        done = False
        
        # We need raw frames for VAE/RNN training, not the latent z
        # So we hack into the base environment to get raw pixels
        raw_obs = base_env.render() 
        raw_obs = cv2.resize(raw_obs, (64, 64), interpolation=cv2.INTER_AREA)
        
        episode_obs = [raw_obs]
        episode_actions = []

        while not done:
            # Agent decides based on Latent State
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to("cpu")
            action = controller.get_action(obs_t)
            
            # Step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Record Raw Data
            raw_obs = base_env.render()
            raw_obs = cv2.resize(raw_obs, (64, 64), interpolation=cv2.INTER_AREA)
            
            episode_obs.append(raw_obs)
            episode_actions.append(action)

        # Save Episode
        obs_data.append(np.array(episode_obs))
        action_data.append(np.array(episode_actions))
        
        if (i+1) % 50 == 0:
            print(f"Recorded Episode {i+1}/{MAX_EPISODES}")

    # 4. Save to Disk
    print("Saving dataset...")
    np.savez_compressed("data_iter2.npz", obs=np.array(obs_data, dtype=object), action=np.array(action_data, dtype=object))
    print("Saved 'data_iter2.npz'.")

if __name__ == "__main__":
    harvest()