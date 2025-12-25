import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gymnasium as gym
import ale_py
import cv2
import cma
import copy
import os
import wandb
from PIL import Image
import io

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ha & Schmidhuber Parameters
LATENT_SIZE = 64
HIDDEN_SIZE = 256

# --- CHANGED FOR SPACE INVADERS ---
# Space Invaders Actions: [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]
ACTION_SIZE = 6 
# ----------------------------------

N_GAUSSIANS = 5

ITERATIONS = 25
HARVEST_EPISODES = 200
VAE_EPOCHS = 20
RNN_EPOCHS = 40
CMA_GENERATIONS = 20
POPULATION = 32

print(f"--- STARTING MDRNN WORLD MODELS (SPACE INVADERS) ON {DEVICE} ---")

# Initialize Weights & Biases
wandb.init(
    project="mdrnn-space-invaders",
    name="auto_train",
    config={
        "latent_size": LATENT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "action_size": ACTION_SIZE,
        "n_gaussians": N_GAUSSIANS,
        "iterations": ITERATIONS,
        "harvest_episodes": HARVEST_EPISODES,
        "vae_epochs": VAE_EPOCHS,
        "rnn_epochs": RNN_EPOCHS,
        "cma_generations": CMA_GENERATIONS,
        "population": POPULATION,
        "env": "SpaceInvadersNoFrameskip-v4"
    }
)

# ==========================================
# 1. VAE MODEL (Encoder/Decoder)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        
        self.fc_mu = nn.Linear(4096, latent_size)
        self.fc_logsigma = nn.Linear(4096, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        return self.fc_mu(x), self.fc_logsigma(x)

class Decoder(nn.Module):
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, 4096)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        return torch.sigmoid(self.deconv4(x))

class VAE(nn.Module):
    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma

# ==========================================
# 2. MDRNN MODELS (Sequence & Cell)
# ==========================================
def gmm_loss(batch, mus, sigmas, logpi, reduce=True):
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob

class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians
        self.gmm_linear = nn.Linear(hiddens, (2 * latents + 1) * gaussians + 2)

class MDRNN(_MDRNNBase):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions, latents): 
        seq_len, bs = actions.size(0), actions.size(1)
        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents
        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        log_sigmas = gmm_outs[:, :, stride:2 * stride]
        log_sigmas = log_sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(torch.clamp(log_sigmas, min=-5, max=5)) 

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]
        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

class MDRNNCell(_MDRNNBase):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden): 
        in_al = torch.cat([action, latent], dim=1)
        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)
        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        log_sigmas = out_full[:, stride:2 * stride]
        log_sigmas = log_sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(torch.clamp(log_sigmas, min=-5, max=5))

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        r = out_full[:, -2]
        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden

# ==========================================
# 3. CONTROLLER & HELPERS
# ==========================================
class Controller(nn.Module):
    def __init__(self, latents, hiddens, actions):
        super().__init__()
        self.fc = nn.Linear(latents + hiddens, actions)

    def forward(self, z, h):
        if isinstance(h, (tuple, list)): h = h[0]
        inp = torch.cat([z, h], dim=1)
        return self.fc(inp)

def get_env():
    gym.register_envs(ale_py)
    # --- CHANGED FOR SPACE INVADERS ---
    return gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")
    # ----------------------------------

def save_checkpoint(vae, mdrnn, mdrnn_cell, controller, opt_vae, opt_rnn, iteration, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'iteration': iteration,
        'vae_state': vae.state_dict(),
        'mdrnn_state': mdrnn.state_dict(),
        'mdrnn_cell_state': mdrnn_cell.state_dict(),
        'controller_state': controller.state_dict(),
        'opt_vae_state': opt_vae.state_dict(),
        'opt_rnn_state': opt_rnn.state_dict(),
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_iter{iteration}.pth")
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint: {path}")
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    torch.save(checkpoint, latest_path)

def load_checkpoint(vae, mdrnn, mdrnn_cell, controller, opt_vae, opt_rnn, checkpoint_dir="checkpoints"):
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    if os.path.exists(latest_path):
        try:
            print(f"Loading checkpoint from {latest_path}...")
            checkpoint = torch.load(latest_path, map_location=DEVICE)
            
            # Simple check to prevent loading Breakout weights (Action Size mismatch)
            saved_controller_weight = checkpoint['controller_state']['fc.weight']
            if saved_controller_weight.shape[0] != ACTION_SIZE:
                print(f"!!! CRITICAL WARNING: Saved checkpoint action size ({saved_controller_weight.shape[0]}) "
                      f"does not match current Space Invaders action size ({ACTION_SIZE}). "
                      f"Starting FRESH.")
                return 0

            vae.load_state_dict(checkpoint['vae_state'])
            mdrnn.load_state_dict(checkpoint['mdrnn_state'])
            mdrnn_cell.load_state_dict(checkpoint['mdrnn_cell_state'])
            controller.load_state_dict(checkpoint['controller_state'])
            opt_vae.load_state_dict(checkpoint['opt_vae_state'])
            opt_rnn.load_state_dict(checkpoint['opt_rnn_state'])
            
            start_iteration = checkpoint['iteration'] + 1
            print(f"✓ Resumed from iteration {checkpoint['iteration']}")
            return start_iteration
        except Exception as e:
            print(f"!! Failed to load checkpoint: {e}. Starting fresh.")
            return 0
    return 0

def process_frame(frame):
    # --- CHANGED FOR SPACE INVADERS ---
    # Breakout crop was [34:194, 8:152]
    # Space Invaders raw is 210x160. Score is top.
    # We crop top 25 (score) and bottom 15 (padding)
    # We keep full width (0:160)
    frame = frame[25:195, :, :] 
    # ----------------------------------
    
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    frame = np.moveaxis(frame, -1, 0) # HWC -> CHW
    return frame / 255.0

def tensor_to_image(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    tensor = tensor.cpu().detach()
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    tensor = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(tensor)

def vae_debug_output(vae, test_frames, device, iteration):
    vae.eval()
    with torch.no_grad():
        test_frames_t = torch.tensor(test_frames[:8], dtype=torch.float32).to(device)
        reconstructed, mu, logsigma = vae(test_frames_t)
        recon_error = F.mse_loss(reconstructed, test_frames_t).item()
        
        fig_images = []
        for i in range(min(4, len(test_frames))):
            orig = tensor_to_image(test_frames_t[i])
            recon = tensor_to_image(reconstructed[i])
            fig_images.append(wandb.Image(orig, caption=f"Original {i}"))
            fig_images.append(wandb.Image(recon, caption=f"Reconstructed {i}"))
        
        wandb.log({
            f"vae_recon_error_iter{iteration}": recon_error,
            f"vae_samples_iter{iteration}": fig_images,
        })
    return recon_error

def rnn_dream_example(vae, mdrnn, mdrnn_cell, test_latents, test_actions, device, iteration):
    vae.eval()
    mdrnn.eval()
    mdrnn_cell.eval()
    with torch.no_grad():
        if isinstance(test_latents, np.ndarray):
            test_latents = torch.tensor(test_latents, dtype=torch.float32).to(device)
        if isinstance(test_actions, np.ndarray):
            test_actions = torch.tensor(test_actions, dtype=torch.float32).to(device)
        
        z = test_latents[0:1]
        a = test_actions
        seq_len = len(a)
        hidden = [torch.zeros(1, HIDDEN_SIZE).to(device), torch.zeros(1, HIDDEN_SIZE).to(device)]
        
        dream_latents = [z.cpu().numpy()[0]]
        dream_rewards = []
        
        for step in range(seq_len):
            action = a[step:step+1]
            mus, sigmas, logpi, r, d, hidden = mdrnn_cell(action, z, hidden)
            pi_sample = torch.multinomial(torch.exp(logpi), 1)
            z_next = mus[torch.arange(1), pi_sample.squeeze(), :]
            dream_latents.append(z_next.cpu().numpy()[0])
            dream_rewards.append(r.item())
            z = z_next
        
        dream_latents_t = torch.tensor(np.array(dream_latents), dtype=torch.float32).to(device)
        dream_frames_t = vae.decoder(dream_latents_t)
        dream_frames = dream_frames_t.cpu().detach().numpy()
        dream_frames = np.clip(dream_frames, 0, 1)
        
        dream_video_frames = []
        for frame in dream_frames:
            frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            dream_video_frames.append(frame_uint8)
        
        dream_video = np.array(dream_video_frames, dtype=np.uint8)
        
        wandb.log({
            f"rnn_dream_video_iter{iteration}": wandb.Video(dream_video, fps=10, format="mp4"),
            f"rnn_dream_total_reward_iter{iteration}": float(np.sum(dream_rewards)),
        })

def controller_play_example(vae, mdrnn_cell, controller, device, iteration, num_steps=600):
    env = get_env()
    obs, info = env.reset()
    current_lives = info.get('lives', 3) # Space Invaders usually starts with 3
    
    # Space Invaders doesn't strictly require Fire to start, but it helps
    obs, _, _, _, info = env.step(1)
    
    frame = process_frame(obs)
    t_frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mu, _ = vae.encoder(t_frame)
    z = mu
    hidden = [torch.zeros(1, HIDDEN_SIZE).to(device), torch.zeros(1, HIDDEN_SIZE).to(device)]
    
    done = False
    total_r = 0
    play_frames_rgb = []
    
    steps = 0
    force_fire_next = False

    while not done and steps < num_steps:
        if force_fire_next:
            action = 1 
            force_fire_next = False
        else:
            with torch.no_grad():
                logits = controller(z, hidden)
                action = torch.argmax(logits, dim=1).item()

        act_t = torch.zeros(1, ACTION_SIZE).to(device)
        act_t[0, action] = 1.0
        with torch.no_grad():
            _, _, _, _, _, hidden = mdrnn_cell(act_t, z, hidden)
        
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        total_r += r
        
        new_lives = info.get('lives', current_lives)
        if new_lives < current_lives:
            if new_lives > 0:
                force_fire_next = True
            current_lives = new_lives

        annotated_frame = obs.copy()
        try:
            import cv2
            cv2.putText(annotated_frame, f"L: {current_lives} R: {total_r:.0f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        except: pass

        play_frames_rgb.append(annotated_frame)
        
        frame = process_frame(obs)
        t_frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = vae.encoder(t_frame)
        z = mu
        steps += 1
    
    env.close()
    
    play_video = np.array(play_frames_rgb, dtype=np.uint8)
    play_video = np.moveaxis(play_video, -1, 1)
    
    wandb.log({
        f"controller_gameplay_video_iter{iteration}": wandb.Video(play_video, fps=20, format="mp4"),
        f"controller_total_reward_iter{iteration}": total_r,
    })
    
    return total_r

def dream_rollout(mdrnn_cell, controller, z_init, max_steps=1000):
    """
    Agent plays inside the MDRNN dream. 
    No VAE, No Real Environment involved.
    """
    # Initialize state
    z = z_init
    hidden = [torch.zeros(1, HIDDEN_SIZE).to(DEVICE), torch.zeros(1, HIDDEN_SIZE).to(DEVICE)]
    
    total_reward = 0
    done = False
    steps = 0
    
    # We add a temperature to make the dream slightly diverse, preventing the 
    # agent from exploiting a specific deterministic glitch in the RNN.
    temperature = 1.0 

    while not done and steps < max_steps:
        # 1. Controller chooses action
        with torch.no_grad():
            logits = controller(z, hidden)
            action = torch.argmax(logits, dim=1).item()

        # 2. MDRNN Predicts Next Step (Dreaming)
        # Prepare action tensor
        act_t = torch.zeros(1, ACTION_SIZE).to(DEVICE)
        act_t[0, action] = 1.0
        
        with torch.no_grad():
            mus, sigmas, logpi, r, d, hidden = mdrnn_cell(act_t, z, hidden)
            
        # 3. Sample next Latent Z from GMM
        # We adjust logpi by temperature to control randomness
        logpi = logpi / temperature
        pi_sample = torch.multinomial(torch.exp(logpi), 1)
        
        # Pick the Gaussian indicated by pi_sample
        # mus shape: [batch, gaussians, latents]
        z_next = mus[0, pi_sample.item(), :].unsqueeze(0)
        
        # 4. Handle Reward and Done
        # r is the predicted reward, d is the predicted terminal state logit
        reward = r.item()
        is_done = torch.sigmoid(d).item() > 0.5
        
        total_reward += reward
        
        # Update loop vars
        z = z_next
        if is_done:
            done = True
            
        steps += 1
        
    return total_reward
# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    vae = VAE(img_channels=3, latent_size=LATENT_SIZE).to(DEVICE)
    mdrnn = MDRNN(latents=LATENT_SIZE, actions=ACTION_SIZE, hiddens=HIDDEN_SIZE, gaussians=N_GAUSSIANS).to(DEVICE)
    mdrnn_cell = MDRNNCell(latents=LATENT_SIZE, actions=ACTION_SIZE, hiddens=HIDDEN_SIZE, gaussians=N_GAUSSIANS).to(DEVICE)
    controller = Controller(latents=LATENT_SIZE, hiddens=HIDDEN_SIZE, actions=ACTION_SIZE).to(DEVICE)

    opt_vae = optim.Adam(vae.parameters(), lr=1e-3)
    opt_rnn = optim.Adam(mdrnn.parameters(), lr=1e-3)

    start_iteration = load_checkpoint(vae, mdrnn, mdrnn_cell, controller, opt_vae, opt_rnn)

    for iteration in range(start_iteration, ITERATIONS):
        print(f"\n=== ITERATION {iteration} ===")
        
        # ---------------------------------------------------------
        # PHASE 1: HARVEST DATA
        # ---------------------------------------------------------
        print("[PHASE 1] Harvesting...")
        env = get_env()
        data_frames = []
        data_actions = []
        data_rewards = []
        data_dones = []
        
        for ep in range(HARVEST_EPISODES):
            obs, _ = env.reset()
            done = False
            ep_frames = []
            ep_actions = []
            ep_rewards = []
            ep_dones = []
            
            if iteration > 0:
                frame = process_frame(obs)
                t_frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): mu, _ = vae.encoder(t_frame)
                z = mu
                hidden = [torch.zeros(1, HIDDEN_SIZE).to(DEVICE), torch.zeros(1, HIDDEN_SIZE).to(DEVICE)]

            steps = 0
            current_lives = 3
            should_fire_next = False

            while not done and steps < 1000:
                frame = process_frame(obs)
                ep_frames.append(frame)
                
                # --- ACTION SELECTION ---
                if should_fire_next:
                    action = 1
                    should_fire_next = False
                elif iteration == 0:
                    if len(ep_frames) == 1: action = 1 
                    else: action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        logits = controller(z, hidden)
                        action = torch.argmax(logits, dim=1).item()
                        # Random fire injection for exploration
                        if np.random.rand() < 0.05: 
                            action = np.random.choice([1, 4, 5]) # Fire, RightFire, LeftFire

                # Update RNN
                if iteration > 0:
                    act_t = torch.zeros(1, ACTION_SIZE).to(DEVICE)
                    act_t[0, action] = 1.0
                    with torch.no_grad():
                        _, _, _, _, _, hidden = mdrnn_cell(act_t, z, hidden)

                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
                
                if 'lives' in info:
                    lives = info['lives']
                    if lives < current_lives and lives > 0:
                        should_fire_next = True 
                        current_lives = lives
                
                if iteration > 0 and not done:
                    frame_next = process_frame(obs)
                    t_frame = torch.tensor(frame_next, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    with torch.no_grad(): mu, _ = vae.encoder(t_frame)
                    z = mu

                a_oh = np.zeros(ACTION_SIZE)
                a_oh[action] = 1
                ep_actions.append(a_oh)
                ep_rewards.append(r)
                ep_dones.append(done)
                steps += 1

            if len(ep_frames) > 10:
                data_frames.append(np.array(ep_frames, dtype=np.float32))
                data_actions.append(np.array(ep_actions, dtype=np.float32))
                data_rewards.append(np.array(ep_rewards, dtype=np.float32))
                data_dones.append(np.array(ep_dones, dtype=np.float32))

        print(f" -> Collected {len(data_frames)} episodes.")
        
        # ---------------------------------------------------------
        # PHASE 2: TRAIN VAE (Weighted Loss)
        # ---------------------------------------------------------
        print("[PHASE 2] Training VAE...")
        train_frames = []
        max_frames = 15000
        frame_count = sum(len(ep) for ep in data_frames)
        
        if frame_count > 0:
            episode_indices = np.random.choice(len(data_frames), size=len(data_frames), replace=False)
            for ep_idx in episode_indices:
                ep_frames = data_frames[ep_idx]
                frame_indices = np.random.choice(len(ep_frames), size=min(len(ep_frames), max_frames - len(train_frames)), replace=False)
                for f_idx in frame_indices:
                    train_frames.append(ep_frames[f_idx])
                    if len(train_frames) >= max_frames: break
                if len(train_frames) >= max_frames: break
            train_frames = np.array(train_frames, dtype=np.float32)
        else:
            print(" -> No frames collected!")
            continue
        
        dset_vae = TensorDataset(torch.tensor(train_frames, dtype=torch.float32).to(DEVICE))
        loader_vae = DataLoader(dset_vae, batch_size=64, shuffle=True)
        
        vae.train()
        for e in range(VAE_EPOCHS):
            t_loss = 0
            for (b_img,) in loader_vae:
                opt_vae.zero_grad()
                recon, mu, logsigma = vae(b_img)
                
                loss_unreduced = F.binary_cross_entropy(recon, b_img, reduction='none')
                # Space Invaders: Weight aliens/bullets (non-black) higher
                weights = torch.ones_like(b_img)
                weights[b_img > 0.05] = 8.0 
                
                bce = (loss_unreduced * weights).sum()
                kld = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
                loss = bce + kld*0.1
                loss.backward()
                opt_vae.step()
                t_loss += loss.item()
            
            if (e+1)%5 == 0:
                print(f" -> VAE Epoch {e+1}: {t_loss/len(loader_vae.dataset):.4f}")
        
        vae_debug_output(vae, train_frames[:8], DEVICE, iteration)

        # ---------------------------------------------------------
        # PHASE 3: ENCODE FOR RNN
        # ---------------------------------------------------------
        print("[PHASE 3] Encoding for RNN...")
        vae.eval()
        rnn_latents, rnn_actions, rnn_rewards, rnn_dones = [], [], [], []
        
        with torch.no_grad():
            for i in range(len(data_frames)):
                frames = torch.tensor(data_frames[i], dtype=torch.float32).to(DEVICE)
                curr_mu = []
                for b in range(0, len(frames), 64):
                    mu, _ = vae.encoder(frames[b:b+64])
                    curr_mu.append(mu.cpu().numpy())
                rnn_latents.append(np.concatenate(curr_mu, axis=0))
                rnn_actions.append(data_actions[i])
                rnn_rewards.append(data_rewards[i])
                rnn_dones.append(data_dones[i])

        # ---------------------------------------------------------
        # PHASE 4: TRAIN MDRNN
        # ---------------------------------------------------------
        print("[PHASE 4] Training MDRNN...")
        SEQ_LEN = 32
        X_lat, X_act, Y_lat, Y_rew, Y_done = [], [], [], [], []
        
        for i in range(len(rnn_latents)):
            lat, act, rew, don = rnn_latents[i], rnn_actions[i], rnn_rewards[i], rnn_dones[i]
            if len(lat) <= SEQ_LEN: continue
            for j in range(0, len(lat) - SEQ_LEN):
                X_lat.append(lat[j:j+SEQ_LEN])
                X_act.append(act[j:j+SEQ_LEN])
                Y_lat.append(lat[j+1:j+SEQ_LEN+1])
                Y_rew.append(rew[j:j+SEQ_LEN])
                Y_done.append(don[j:j+SEQ_LEN])
                
        dset_rnn = TensorDataset(
            torch.tensor(np.array(X_lat), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(X_act), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(Y_lat), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(Y_rew), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(Y_done), dtype=torch.float32).to(DEVICE)
        )
        loader_rnn = DataLoader(dset_rnn, batch_size=32, shuffle=True)
        
        mdrnn.train()
        for e in range(RNN_EPOCHS):
            t_loss = 0
            for b_lat, b_act, t_lat, t_rew, t_done in loader_rnn:
                opt_rnn.zero_grad()
                b_act = b_act.transpose(0, 1)
                b_lat = b_lat.transpose(0, 1)
                t_lat = t_lat.transpose(0, 1)
                
                mus, sigmas, logpi, rs, ds = mdrnn(b_act, b_lat)
                
                gl = gmm_loss(t_lat.transpose(0,1), mus.transpose(0,1), sigmas.transpose(0,1), logpi.transpose(0,1))
                r_loss = F.mse_loss(rs.transpose(0,1), t_rew)
                d_loss = F.binary_cross_entropy_with_logits(ds.transpose(0,1), t_done)
                
                loss = gl + r_loss + d_loss
                loss.backward()
                nn.utils.clip_grad_norm_(mdrnn.parameters(), 5.0)
                opt_rnn.step()
                t_loss += loss.item()
            if (e+1)%5 == 0:
                print(f" -> RNN Epoch {e+1}: {t_loss/len(loader_rnn):.4f}")

        # SYNC WEIGHTS TO CELL
        mdrnn_cell.gmm_linear.load_state_dict(mdrnn.gmm_linear.state_dict())
        mdrnn_cell.rnn.weight_ih.data = mdrnn.rnn.weight_ih_l0.data
        mdrnn_cell.rnn.weight_hh.data = mdrnn.rnn.weight_hh_l0.data
        mdrnn_cell.rnn.bias_ih.data = mdrnn.rnn.bias_ih_l0.data
        mdrnn_cell.rnn.bias_hh.data = mdrnn.rnn.bias_hh_l0.data
        
        # ---------------------------------------------------------
        # PHASE 5: EVOLVE CONTROLLER (IN DREAM)
        # ---------------------------------------------------------
        print("[PHASE 5] Evolving Controller in the Dream...")
        
        # 1. Prepare Seed Zs (Starting points for dreams)
        # We need real Zs to start the dream so the RNN knows where the game begins.
        # We take the first Z from the harvested data.
        dream_seeds = []
        if len(rnn_latents) > 0:
            # Take the first latent from up to 10 harvested episodes
            for i in range(min(10, len(rnn_latents))):
                seed_z = torch.tensor(rnn_latents[i][0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                dream_seeds.append(seed_z)
        else:
            # Fallback if no data (shouldn't happen if harvest ran)
            dream_seeds.append(torch.zeros(1, LATENT_SIZE).to(DEVICE))

        flat_params = torch.cat([p.flatten() for p in controller.parameters()]).cpu().detach().numpy()
        es = cma.CMAEvolutionStrategy(flat_params, 0.5, {'popsize': POPULATION})
        
        for g in range(CMA_GENERATIONS):
            solutions = es.ask()
            dream_rewards = []
            
            # --- FAST DREAM TRAINING LOOP ---
            for s in solutions:
                # Load params into controller
                idx = 0
                state_dict = controller.state_dict()
                for k, v in state_dict.items():
                    sz = v.numel()
                    p = torch.from_numpy(s[idx:idx+sz]).float().to(DEVICE)
                    state_dict[k].copy_(p.view(v.shape))
                    idx += sz
                
                # Pick a random start state from real games
                z_init = dream_seeds[np.random.randint(len(dream_seeds))]
                
                # Run in Dream
                r_dream = dream_rollout(mdrnn_cell, controller, z_init)
                dream_rewards.append(r_dream)

            # Update CMA-ES based on DREAM scores
            es.tell(solutions, [-r for r in dream_rewards])
            
            # --- REAL WORLD VALIDATION ---
            # We must verify if the dream score matches reality.
            # Otherwise, the agent might hallucinate infinite points.
            best_idx = np.argmax(dream_rewards)
            best_params = solutions[best_idx]
            
            # Load best params
            idx = 0
            state_dict = controller.state_dict()
            for k, v in state_dict.items():
                sz = v.numel()
                p = torch.from_numpy(best_params[idx:idx+sz]).float().to(DEVICE)
                state_dict[k].copy_(p.view(v.shape))
                idx += sz
            
            # Run ONE real episode to validate
            real_reward = controller_play_example(vae, mdrnn_cell, controller, DEVICE, iteration, num_steps=1000)
            
            print(f" -> Gen {g}: Dream Mean={np.mean(dream_rewards):.1f}, Dream Best={np.max(dream_rewards):.1f}, >> Real Score={real_reward}")
            
            wandb.log({
                "cma_generation": g,
                "dream_mean_reward": np.mean(dream_rewards),
                "dream_best_reward": np.max(dream_rewards),
                "real_validation_reward": real_reward,
                "cma_iteration": iteration,
            })
            
        
        controller_play_reward = controller_play_example(vae, mdrnn_cell, controller, DEVICE, iteration, num_steps=1024)
        save_checkpoint(vae, mdrnn, mdrnn_cell, controller, opt_vae, opt_rnn, iteration)

if __name__ == "__main__":
    main()
    wandb.finish()