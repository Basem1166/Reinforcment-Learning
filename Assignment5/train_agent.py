import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import numpy as np
import cv2
import cma  # <--- The Evolution Strategy Library
import wandb
import os

# --- 1. Load Pre-trained Models (VAE & RNN) ---
# (Same architecture as before)
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2) 
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
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
        return mu # Deterministic for controller training

class RobustRNN(nn.Module):
    def __init__(self, latent_dim=32, action_size=4, hidden_size=256):
        super(RobustRNN, self).__init__()
        self.lstm = nn.LSTM(latent_dim + action_size, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
    def forward(self, z, action, hidden=None):
        x = torch.cat([z, action], dim=-1)
        output, hidden = self.lstm(x, hidden)
        return output, hidden

# --- 2. The Linear Controller ---
# The original paper uses a simple Linear Layer (w*x + b). 
# It maps [z, h] -> Action Logits
class Controller(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Controller, self).__init__()
        self.fc = nn.Linear(input_dim, action_dim) 
        
    def forward(self, x):
        return self.fc(x)

    def get_action(self, x):
        # x is tensor (1, 288)
        with torch.no_grad():
            logits = self.fc(x)
            return torch.argmax(logits, dim=1).item()
    
    # Helper to load flat parameters from CMA-ES
    def set_parameters(self, flat_params):
        # Copy the flat numpy array into the PyTorch weights
        state_dict = self.state_dict()
        current_idx = 0
        
        # We manually iterate because PyTorch dicts are ordered
        for key in state_dict:
            param = state_dict[key]
            num_elements = param.numel()
            chunk = flat_params[current_idx : current_idx + num_elements]
            
            # Reshape and load
            chunk_tensor = torch.from_numpy(chunk).float().view(param.size())
            state_dict[key].copy_(chunk_tensor)
            current_idx += num_elements

# --- 3. The "Dream" Environment ---
class LatentBlockWrapper(gym.Wrapper):
    def __init__(self, env, vae, rnn, device):
        super().__init__(env)
        self.vae = vae
        self.rnn = rnn
        self.device = device
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(32 + 256,), dtype=np.float32)
        self.rnn_hidden = None
        self.last_z = None

    def _process_frame(self, frame):
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        frame = np.moveaxis(frame, -1, 0)
        t = torch.from_numpy(frame).float().unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            mu, _ = self.vae.encode(t)
            z = mu # Use Mean for deterministic behavior
        return z

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_z = self._process_frame(obs)
        self.rnn_hidden = None 
        h_zero = torch.zeros(1, 1, 256).to(self.device)
        state = torch.cat([self.last_z, h_zero.squeeze(0)], dim=-1)
        return state.cpu().numpy().flatten(), info

    def step(self, action):
        a_tensor = torch.zeros(1, 1, 4).to(self.device)
        a_tensor[0, 0, action] = 1.0
        with torch.no_grad():
            _, self.rnn_hidden = self.rnn(self.last_z.unsqueeze(0), a_tensor, self.rnn_hidden)
            h_curr = self.rnn_hidden[0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_z = self._process_frame(obs)
        state = torch.cat([self.last_z, h_curr.squeeze(0)], dim=-1)
        return state.cpu().numpy().flatten(), reward, terminated, truncated, info

def rollout(env, controller, device):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0  # <--- Initialize counter
    
    # Add "and steps < 2000" to safety check
    while not done and steps < 2000:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        action = controller.get_action(obs_t)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1  # <--- Increment counter
        
        if total_reward > 100: break 
        
    return total_reward

# --- 5. Main Training Loop (CMA-ES) ---
def train_cma():
    wandb.init(project="world-models-breakout", name="cma-es-linear")
    device = torch.device("cpu") # KEEP CONTROLLER ON CPU. VAE/RNN logic handles GPU internally if needed.
    
    print("Loading World Model...")
    # Load VAE/RNN on GPU if available, pass device to wrapper
    wm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae = VAE().to(wm_device)
    vae.load_state_dict(torch.load("vae.pth", map_location=wm_device, weights_only=True))
    vae.eval()
    
    rnn = RobustRNN().to(wm_device)
    rnn.load_state_dict(torch.load("rnn.pth", map_location=wm_device, weights_only=True))
    rnn.eval()
    
    # Environment
    gym.register_envs(ale_py)
    base_env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = LatentBlockWrapper(base_env, vae, rnn, wm_device)
    
    # Controller
    input_dim = 32 + 256 # z + h
    action_dim = 4
    controller = Controller(input_dim, action_dim).to(device) # Controller stays on CPU for CMA
    
    # Count parameters
    num_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller has {num_params} parameters.")
    
    # --- Initialize CMA-ES ---
    # sigma0=0.1 is the initial standard deviation (exploration noise)
    es = cma.CMAEvolutionStrategy(num_params * [0], 0.1, {'popsize': 32})
    
    generation = 0
    
    while not es.stop():
        # 1. Ask for candidate solutions (parameters)
        solutions = es.ask()
        
        # 2. Evaluate each solution
        rewards = []
        for i, solution in enumerate(solutions):
            # Load weights into controller
            controller.set_parameters(np.array(solution))
            
            # Run Episode
            r = rollout(env, controller, device)
            rewards.append(r)
            
            # Print progress every few candidates
            # print(f"  Candidate {i}/{len(solutions)}: Reward {r}")
            
        # 3. Tell CMA-ES the results
        # Note: CMA minimizes, so we pass negative rewards
        es.tell(solutions, [-r for r in rewards])
        
        # 4. Logs
        best_reward = max(rewards)
        mean_reward = sum(rewards) / len(rewards)
        print(f"Generation {generation} | Best: {best_reward} | Mean: {mean_reward:.2f}")
        
        wandb.log({
            "generation": generation,
            "best_reward": best_reward,
            "mean_reward": mean_reward
        })
        
        # Save best model occasionally
        if generation % 10 == 0:
            best_idx = np.argmax(rewards)
            controller.set_parameters(np.array(solutions[best_idx]))
            torch.save(controller.state_dict(), "controller_cma.pth")
            
        generation += 1
        
    print("Optimization finished.")

if __name__ == "__main__":
    train_cma()