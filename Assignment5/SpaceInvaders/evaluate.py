import torch
from env import make_env
from vae import VAE
from mdrnn import MDRNN
from controller import Controller
from config import Config
import wandb

cfg = Config()
device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

env = make_env(record=cfg.RECORD_VIDEO, video_dir=cfg.VIDEO_DIR)
vae = VAE().to(device)
mdrnn = MDRNN(action_dim=env.action_space.n, hidden_size=cfg.RNN_HIDDEN).to(device)
controller = Controller(action_dim=env.action_space.n, hidden_size=cfg.CONTROLLER_HIDDEN).to(device)

# Load checkpoints
vae.load_state_dict(torch.load(f"{cfg.CHECKPOINT_DIR}/world_model_{cfg.TOTAL_EPISODES_WM-10}.pt")["vae"])
controller.load_state_dict(torch.load(f"{cfg.CHECKPOINT_DIR}/controller_{cfg.TOTAL_EPISODES_CONTROLLER-10}.pt"))

wandb.init(project="eval-spaceinvaders")

for ep in range(cfg.EVAL_EPISODES):
    obs, _ = env.reset()
    h = None
    done = False
    total_reward = 0
    steps = 0

    while not done:
        obs_t = torch.tensor(obs).permute(2,0,1).unsqueeze(0).float().to(device)
        z, _ = vae.encode(obs_t)

        action = torch.argmax(controller(z, h), dim=-1).item()
        obs, reward, done, _, _ = env.step(action)

        total_reward += reward
        steps += 1

    wandb.log({
        "eval_reward": total_reward,
        "eval_steps": steps
    })
    print(f"[EVALUATION] Episode {ep} - Reward: {total_reward} - Steps: {steps}")