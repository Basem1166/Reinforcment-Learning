import torch
import wandb
import numpy as np
from env import make_env
from vae import VAE, vae_loss
from mdrnn import MDRNN
from controller import Controller
from config import Config
import os

cfg = Config()
device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.VIDEO_DIR, exist_ok=True)

# ---------------- Initialize env & models ----------------
env = make_env(record=cfg.RECORD_VIDEO, video_dir=cfg.VIDEO_DIR)
vae = VAE(z_dim=cfg.LATENT_DIM).to(device)
mdrnn = MDRNN(action_dim=env.action_space.n, hidden_size=cfg.RNN_HIDDEN, latent_dim=cfg.LATENT_DIM).to(device)
controller = Controller(action_dim=env.action_space.n, hidden_size=cfg.CONTROLLER_HIDDEN,
                        latent_dim=cfg.LATENT_DIM, rnn_hidden=cfg.RNN_HIDDEN).to(device)

# ---------------- World Model Training ----------------
wandb.init(project=cfg.WAND_B_PROJECT_WM)
optimizer_wm = torch.optim.Adam(list(vae.parameters()) + list(mdrnn.parameters()), lr=cfg.LR_WM)

print("=== Training World Model ===")
for ep in range(cfg.TOTAL_EPISODES_WM):
    obs, info = env.reset()
    h = None
    done = False
    ep_reward = 0
    steps = 0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) / 255.0  # normalize
        recon, mu, logvar = vae(obs_t)
        z = mu  # use mean as latent

        # Sample action
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # One-hot encode action
        a_onehot = torch.zeros(1, env.action_space.n, device=device)
        a_onehot[0, action] = 1

        # MDRNN forward
        z_next, r_pred, _, h = mdrnn(z, a_onehot, h)

        # VAE loss
        loss = vae_loss(recon, obs_t, mu, logvar)
        optimizer_wm.zero_grad()
        loss.backward()
        optimizer_wm.step()

        obs = obs2
        ep_reward += reward
        steps += 1

    wandb.log({"vae_loss": loss.item(), "reward": ep_reward, "timesteps": steps})
    print(f"[WORLD MODEL] Episode {ep} - Reward: {ep_reward} - Steps: {steps}")

    if ep % cfg.CHECKPOINT_FREQ == 0:
        torch.save({"vae": vae.state_dict(), "mdrnn": mdrnn.state_dict()},
                   f"{cfg.CHECKPOINT_DIR}/world_model_{ep}.pt")

# ---------------- Controller Training ----------------
wandb.init(project=cfg.WAND_B_PROJECT_CONTROLLER)
optimizer_ctrl = torch.optim.Adam(controller.parameters(), lr=cfg.LR_CONTROLLER)

# Load latest world model checkpoint
wm_ckpt = torch.load(f"{cfg.CHECKPOINT_DIR}/world_model_{cfg.TOTAL_EPISODES_WM-10}.pt", map_location=device)
vae.load_state_dict(wm_ckpt["vae"])
mdrnn.load_state_dict(wm_ckpt["mdrnn"])

print("=== Training Controller ===")
for ep in range(cfg.TOTAL_EPISODES_CONTROLLER):
    obs, info = env.reset()
    h = None
    done = False
    ep_reward = 0
    steps = 0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
        z, _ = vae.encode(obs_t)

        logits = controller(z, h)
        action = torch.argmax(logits, dim=-1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1

    wandb.log({"controller_reward": ep_reward, "controller_timesteps": steps})
    print(f"[CONTROLLER] Episode {ep} - Reward: {ep_reward} - Steps: {steps}")

    if ep % cfg.CHECKPOINT_FREQ == 0:
        torch.save(controller.state_dict(), f"{cfg.CHECKPOINT_DIR}/controller_{ep}.pt")

# ---------------- Evaluation ----------------
wandb.init(project="eval-spaceinvaders")
vae.load_state_dict(torch.load(f"{cfg.CHECKPOINT_DIR}/world_model_{cfg.TOTAL_EPISODES_WM-10}.pt", map_location=device)["vae"])
controller.load_state_dict(torch.load(f"{cfg.CHECKPOINT_DIR}/controller_{cfg.TOTAL_EPISODES_CONTROLLER-10}.pt", map_location=device))

eval_env = make_env(record=True, video_dir=cfg.VIDEO_DIR)
print("=== Evaluation ===")
for ep in range(cfg.EVAL_EPISODES):
    obs, info = eval_env.reset()
    h = None
    done = False
    total_reward = 0
    steps = 0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
        z, _ = vae.encode(obs_t)
        logits = controller(z, h)
        action = torch.argmax(logits, dim=-1).item()

        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    wandb.log({"eval_reward": total_reward, "eval_steps": steps})
    print(f"[EVAL] Episode {ep}: Reward={total_reward}, Steps={steps}")
