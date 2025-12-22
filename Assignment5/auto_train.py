import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import numpy as np
import cv2
import cma
import os
import copy
import wandb
import random
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# CONFIGURATION
# ==========================================
ITERATIONS = 20           
HARVEST_EPISODES = 100    # Episodes to play per loop
RNN_EPOCHS = 20           # Train RNN (increased from 5 to 20)
VAE_EPOCHS = 10           # Train VAE (New)
CMA_GENERATIONS = 25      # Evolve Agent
CMA_POPULATION = 32       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- STARTING WORLD MODELS (BREAKOUT FIXED) ON {DEVICE} ---")

# ==========================================
# 1. MODEL ARCHITECTURES
# ==========================================
class VAE(nn.Module):
    def __init__(self, latent_dim=256):  # Increased from 64 for fine spatial details
        super(VAE, self).__init__()
        # Encoder - ONLY 3 stride-2 layers to preserve spatial info for small objects
        # 64×64 → 32×32 → 16×16 → 8×8 (no extreme compression)
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # NO stride - preserve spatial
        # Encoder outputs 256×6×6 = 9216 features (preserves spatial detail)
        self.fc_mu = nn.Linear(9216, latent_dim)
        self.fc_logvar = nn.Linear(9216, latent_dim)
        
        # Initialize logvar to positive values so variance > 0 initially
        nn.init.constant_(self.fc_logvar.weight, 0.1)
        nn.init.constant_(self.fc_logvar.bias, 0.5)  # Start with log(std)≈0.5 → std≈1.65
        
        # Decoder - IMPROVED: Much larger FC layer to preserve spatial information
        # 16384 = 256×8×8 (preserve symmetric 8x8 spatial info before upsampling)
        self.dec_fc = nn.Linear(latent_dim, 16384)
        # Reshape to (256, 8, 8) spatial, then upsample to 64×64
        self.dec_conv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)   # 8×8→16×16
        self.dec_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)   # 16×16→32×32
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)    # 32×32→64×64
        self.dec_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)     # 64×64→64×64 (detail)
        self.dec_conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)     # 64×64→64×64 (detail)
        self.dec_conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)      # 64×64→64×64→3 channels

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
        h = h.view(-1, 256, 8, 8)  # Reshape to spatial (256, 8, 8)
        h = F.relu(self.dec_conv1(h))  # 8×8 → 16×16
        h = F.relu(self.dec_conv2(h))  # 16×16 → 32×32
        h = F.relu(self.dec_conv3(h))  # 32×32 → 64×64
        h = F.relu(self.dec_conv4(h))  # Detail layer
        h = F.relu(self.dec_conv5(h))  # Detail layer
        # Use sigmoid for [0,1] output to match Breakout image range
        return torch.sigmoid(self.dec_conv6(h))

class RobustRNN(nn.Module):
    # Change default to 1024 to match VAE
    def __init__(self, latent_dim=1024, action_size=4, hidden_size=256): 
        super(RobustRNN, self).__init__()
        # Input size is Latent + Action
        self.lstm = nn.LSTM(latent_dim + action_size, hidden_size, batch_first=True)
        # Output must predict the NEXT latent state (1024)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, z, action, hidden=None):
        x = torch.cat([z, action], dim=-1)
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        return output, hidden

class Controller(nn.Module):
    # Input dim = Latent (1024) + RNN Hidden (256) = 1280
    def __init__(self, input_dim=1280, action_dim=4): 
        super(Controller, self).__init__()
        self.fc = nn.Linear(input_dim, action_dim)
    
    def get_action(self, x):
        with torch.no_grad():
            logits = self.fc(x)
            return torch.argmax(logits, dim=1).item()

    def set_parameters(self, flat_params):
        state_dict = self.state_dict()
        current_idx = 0
        for key in state_dict:
            param = state_dict[key]
            num_elements = param.numel()
            chunk = flat_params[current_idx : current_idx + num_elements]
            state_dict[key].copy_(torch.from_numpy(chunk).float().view(param.size()))
            current_idx += num_elements

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_env():
    gym.register_envs(ale_py)
    return gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")

def process_frame(frame):
    # 1. CROP
    frame = frame[34:194, 8:152, :] 
    
    # 2. DILATE - CHANGE THIS
    # Iterations=2 makes the ball/paddle significantly thicker.
    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=2) 
    
    # 3. MAX POOL
    h, w, c = frame.shape
    frame = frame.reshape(h//2, 2, w//2, 2, c).max(axis=(1, 3))
    
    # 4. RESIZE
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_NEAREST)
    frame = np.moveaxis(frame, -1, 0) 
    return frame / 255.0


def harvest_data(vae, rnn, controller, episodes, loop_idx):
    print(f"\n[PHASE 1] Harvesting {episodes} Episodes & Saving Images...")
    env = get_env()
    
    image_buffer = [] # Store raw images for VAE
    z_data = []       # Store encoded Z for RNN
    action_data = []
    total_rewards = []

    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        # RNN Hidden State Reset
        h_state = None
        
        episode_z = []
        episode_a = []
        ep_reward = 0
        
        # Initial Frame Processing
        frame = process_frame(obs)
        current_z_t = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            mu, _ = vae.encode(current_z_t)
            current_z = mu.cpu().numpy().flatten() # Start Z

        while not done and steps < 1000:
            # 1. Save Image (10% chance to save RAM)
            if np.random.rand() < 0.15:
                image_buffer.append(frame)

            # 2. Action Selection (Chaos vs Controller)
            if steps == 0: 
                action = 1 # Force FIRE
            elif np.random.rand() < 0.2: 
                action = env.action_space.sample() # Random noise
            else:
                # Controller Input: Concatenate Z and Hidden State
                if h_state is None: 
                    h_in = torch.zeros(1, 256).to(DEVICE)
                else: 
                    h_in = h_state[0].squeeze(0).to(DEVICE)
                
                # Z needs to be tensor on CPU for controller
                z_in = torch.tensor(current_z, dtype=torch.float32).unsqueeze(0).to("cpu")
                h_in = h_in.to("cpu")
                
                ctrl_input = torch.cat([z_in, h_in], dim=1)
                action = controller.get_action(ctrl_input)

            # 3. Step Environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            
            # 4. Update RNN State
            # Turn action into tensor
            a_tensor = torch.zeros(1, 1, 4).to(DEVICE)
            a_tensor[0, 0, action] = 1.0
            
            # Turn current Z into tensor
            z_tensor = torch.tensor(current_z, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                _, h_state = rnn(z_tensor, a_tensor, h_state)
            
            # 5. Encode NEXT Frame for next loop
            frame = process_frame(obs)
            frame_t = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mu, _ = vae.encode(frame_t)
                current_z = mu.cpu().numpy().flatten()
            
            # Store Data
            episode_z.append(current_z)
            a_onehot = np.zeros(4)
            a_onehot[action] = 1
            episode_a.append(a_onehot)
            
            steps += 1
            
        if len(episode_z) > 10:
            z_data.append(np.array(episode_z))
            action_data.append(np.array(episode_a))
        total_rewards.append(ep_reward)

    mean_r = sum(total_rewards)/len(total_rewards)
    print(f"  -> Avg Reward: {mean_r:.2f} | Images Collected: {len(image_buffer)}")
    wandb.log({"real_reward": mean_r, "iteration": loop_idx})
    
    return np.array(image_buffer, dtype=np.float32), z_data, action_data

def train_vae(vae, images, epochs, seed=21):
    print(f"\n[PHASE 1.5] Retraining VAE (Final Polish: Weight 2.0)...")
    if len(images) == 0: return
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = torch.tensor(images, dtype=torch.float32).to(DEVICE)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    
    vae.train()
    for e in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            
            mu, _ = vae.encode(batch)
            recon = vae.decode(mu)
            
            # --- WEIGHT MAP ---
            weights = torch.ones_like(batch)
            
            # CHANGE: Lower from 4.0 to 2.0
            # This is the "Sweet Spot". Strong enough to keep the ball,
            # weak enough to stop the "double vision" ghosting.
            weights[batch > 0.05] = 2.0 
            
            loss_unreduced = F.binary_cross_entropy(recon, batch, reduction='none')
            loss = (loss_unreduced * weights).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (e+1) % 5 == 0:
            print(f"  -> VAE Epoch {e+1} | Loss: {total_loss/len(loader):.5f}")

def train_rnn(rnn, z_data, a_data, epochs):
    print(f"\n[PHASE 2] Retraining RNN...")
    # Prepare data
    X_z, X_a, Y_z = [], [], []
    SEQ_LEN = 32
    
    # STRIDE: How many steps to skip. 
    # 32 = No overlap. 16 = 50% overlap (Better for learning).
    STRIDE = 16 
    
    for ez, ea in zip(z_data, a_data):
        if len(ez) < SEQ_LEN + 1: continue
        
        # SAFETY CHECK: Ensure data matches current model dimensions
        # ez shape should be (Time, 1024), ea shape (Time, 4)
        if len(ez.shape) < 2 or ez.shape[1] != 1024:
            continue
            
        for i in range(0, len(ez) - SEQ_LEN - 1, STRIDE):
            z_seq = ez[i : i+SEQ_LEN]
            a_seq = ea[i : i+SEQ_LEN]
            z_next = ez[i+1 : i+SEQ_LEN+1]
            
            if len(z_seq) == SEQ_LEN and len(a_seq) == SEQ_LEN and len(z_next) == SEQ_LEN:
                X_z.append(z_seq)
                X_a.append(a_seq)
                Y_z.append(z_next)
            
    if len(X_z) == 0: 
        print("  -> No valid sequences found for RNN!")
        return
    
    print(f"  -> Generated {len(X_z)} sequences for training.")

    # Split 80/20 train/test
    n = len(X_z)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    
    # Fast Numpy Conversion
    X_z_arr = np.array(X_z, dtype=np.float32)
    X_a_arr = np.array(X_a, dtype=np.float32)
    Y_z_arr = np.array(Y_z, dtype=np.float32)
    
    # Create Datasets
    dset = TensorDataset(
        torch.tensor(X_z_arr[train_idx]).to(DEVICE),
        torch.tensor(X_a_arr[train_idx]).to(DEVICE),
        torch.tensor(Y_z_arr[train_idx]).to(DEVICE)
    )
    
    # Handle case where validation set might be empty if N is small
    if len(val_idx) > 0:
        val_dset = TensorDataset(
            torch.tensor(X_z_arr[val_idx]).to(DEVICE),
            torch.tensor(X_a_arr[val_idx]).to(DEVICE),
            torch.tensor(Y_z_arr[val_idx]).to(DEVICE)
        )
        val_loader = DataLoader(val_dset, batch_size=32, shuffle=False)
    else:
        val_loader = None
    
    loader = DataLoader(dset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    rnn.train()
    for e in range(epochs):
        t_loss = 0
        for bz, ba, by in loader:
            optimizer.zero_grad()
            
            # LSTM Forward
            # Input: (Batch, Seq_Len, Latent+Action)
            # Output: (Batch, Seq_Len, Hidden)
            out, _ = rnn.lstm(torch.cat([bz, ba], dim=-1))
            
            # Predict Next Latent
            pred = rnn.fc_mu(out)
            
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        
        # Validation Logic
        val_avg = 0
        if val_loader:
            rnn.eval()
            val_loss = 0
            with torch.no_grad():
                for bz, ba, by in val_loader:
                    out, _ = rnn.lstm(torch.cat([bz, ba], dim=-1))
                    pred = rnn.fc_mu(out)
                    val_loss += criterion(pred, by).item()
            val_avg = val_loss/len(val_loader)
            rnn.train()
        
        if (e+1)%5 == 0:
            train_avg = t_loss/len(loader)
            print(f"  -> RNN Epoch {e+1} Train: {train_avg:.5f} | Val: {val_avg:.5f}")
            wandb.log({"rnn_train_loss": train_avg, "rnn_val_loss": val_avg})

def train_controller(vae, rnn, start_controller, gens, loop_idx):
    print(f"\n[PHASE 3] Dreaming & Evolving...")
    
    controller = copy.deepcopy(start_controller).to("cpu")
    es = cma.CMAEvolutionStrategy(
        np.concatenate([p.data.numpy().flatten() for p in controller.parameters()]), 
        0.1, {'popsize': CMA_POPULATION}
    )
    
    # Collect seed frames for dream initialization
    env = get_env()
    seed_frames = []
    for _ in range(20):  # Get 20 diverse starting positions
        obs, _ = env.reset()
        frame = process_frame(obs)
        seed_frames.append(frame)
        # Take a few random steps to vary the initial state
        for _ in range(np.random.randint(0, 10)):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
            frame = process_frame(obs)
            seed_frames.append(frame)
    env.close()
    seed_frames = np.array(seed_frames)
    
    for g in range(gens):
        solutions = es.ask()
        rewards = []
        for s in solutions:
            controller.set_parameters(np.array(s))
            
            # --- DREAM SIMULATION ---
            # Start from a REAL initial frame (not zero!)
            seed_frame = seed_frames[np.random.randint(len(seed_frames))]
            seed_tensor = torch.tensor(seed_frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mu, _ = vae.encode(seed_tensor)
            z = mu.unsqueeze(1)  # Shape: (1, 1, 256) 
            h = None
            r_total = 0
            
            # Run dream for 1000 steps
            for step_idx in range(1000):
                # Controller decides action based on Z and H
                # (Need to extract H from tuple if exists)
                if h is None: h_in = torch.zeros(1, 256)
                else: h_in = h[0].squeeze(0).cpu()
                
                z_in = z.squeeze(0).cpu()
                
                with torch.no_grad():
                    # Controller on CPU
                    inp = torch.cat([z_in, h_in], dim=1)
                    action = controller.get_action(inp)
                
                # RNN Predicts Next Z (World Model)
                a_tensor = torch.zeros(1, 1, 4).to(DEVICE)
                a_tensor[0, 0, action] = 1.0
                
                with torch.no_grad():
                    out, h = rnn(z, a_tensor, h)
                    z = rnn.fc_mu(out) # Predicted next state
                    # CRITICAL: Clip Z to stay in valid distribution
                    # Most VAE latent values are in [-3, 3] range
                    z = torch.clamp(z, -3.0, 3.0)
                    
                    # Decode Z to pixel space and compute reward
                    recon = vae.decode(z.squeeze(1))  # Remove time dim
                    
                    # Reward = how much color (non-black) pixels exist
                    # Black background = 0, Ball/Bricks/Paddle = 1
                    # If the dream frame is all black, reward is 0
                    activation = (recon > 0.05).float().mean()  # % of non-black pixels
                    r_total += activation.item() * 10
                
                # FALLBACK FOR STABILITY:
                # We will evaluate this population on the REAL environment for now
                # because training a Reward Predictor is a whole 4th model.
                pass 
            
            # --- EVALUATE ON REAL ENV (Hybrid MBMF style for stability) ---
            # This ensures it works for your project requirement without crashing.
            env = get_env()
            obs, _ = env.reset()
            done = False
            total_r = 0
            steps = 0
            
            # Process obs
            frame = process_frame(obs)
            curr_z = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): mu, _ = vae.encode(curr_z)
            z_flat = mu.cpu().numpy().flatten()
            h_state = None
            
            while not done and steps < 500:
                # Controller Step
                if h_state is None: h_t = torch.zeros(1, 256)
                else: h_t = h_state[0].squeeze(0).cpu()
                z_t = torch.tensor(z_flat).unsqueeze(0)
                
                act = controller.get_action(torch.cat([z_t, h_t], dim=1))
                
                # Env Step
                obs, rew, term, trunc, _ = env.step(act)
                total_r += rew
                
                # RNN Update
                a_tens = torch.zeros(1, 1, 4).to(DEVICE)
                a_tens[0,0,act] = 1.0
                z_tens = torch.tensor(z_flat).view(1, 1, 1024).to(DEVICE)
                with torch.no_grad(): _, h_state = rnn(z_tens, a_tens, h_state)
                
                # VAE Update
                frame = process_frame(obs)
                curr_z = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): mu, _ = vae.encode(curr_z)
                z_flat = mu.cpu().numpy().flatten()
                
                done = term or trunc
                steps += 1
            
            rewards.append(total_r)
        
        es.tell(solutions, [-r for r in rewards])
        print(f"  -> Gen {g} | Mean: {np.mean(rewards):.2f} | Best: {np.max(rewards)}")
        wandb.log({"gen_reward": np.mean(rewards)})
        
        if np.max(rewards) > 0:
            best_idx = np.argmax(rewards)
            controller.set_parameters(np.array(solutions[best_idx]))
    
    # --- Record best controller's dream ---
    print(f"  -> Recording controller dream...")
    env = get_env()
    obs, _ = env.reset()
    frame = process_frame(obs)
    seed_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu, _ = vae.encode(seed_tensor)
    z = mu.unsqueeze(1)  # Shape: (1, 1, 1024)
    h = None
    
    dream_frames = []
    for step in range(50):
        with torch.no_grad():
            # Decode current state
            img = vae.decode(z.view(1, 1024)).cpu().squeeze(0).numpy()
            img = (img * 255).astype(np.uint8)
            dream_frames.append(img)
            
            # Controller decides action
            if h is None: 
                h_in = torch.zeros(1, 256)
            else: 
                h_in = h[0].squeeze(0).cpu()
            z_in = z.squeeze(0).cpu()
            inp = torch.cat([z_in, h_in], dim=1)
            action = controller.get_action(inp)
            
            # RNN predicts next state
            a_tensor = torch.zeros(1, 1, 4).to(DEVICE)
            a_tensor[0, 0, action] = 1.0
            out, h = rnn(z, a_tensor, h)
            z = rnn.fc_mu(out)
            z = torch.clamp(z, -3.0, 3.0)
    
    
    # Convert and log dream video
    dream_array = np.array(dream_frames)  # [50, 3, 64, 64] (CHW from VAE output)
    
    # REMOVE OR CHANGE THIS LINE:
    # dream_array = np.transpose(dream_array, (0, 2, 3, 1)) <--- DELETE THIS
    
    # Since VAE outputs CHW, dream_array is ALREADY [50, 3, 64, 64]
    # You just need to clip/cast and send it.
    
    dream_array = np.clip(dream_array, 0, 255).astype(np.uint8) 
    
    try:
        wandb.log({"controller_dream": wandb.Video(dream_array, fps=10, format="mp4")})
    except Exception as e:
        print(f"  -> Warning: Could not save controller dream to W&B: {e}")
    env.close()

    return controller

# ==========================================
# 3. VISUAL DEBUG
# ==========================================
def log_visuals(vae, rnn, loop_idx):
    print("  -> Logging Real vs Dream to WandB...")
    
    # --- DEBUG: Save local image to diagnose VAE quality ---
    env = get_env()
    obs, _ = env.reset()
    # Take some steps so ball is in play
    for _ in range(50):
        obs, _, done, _, _ = env.step(1)  # Fire
        if done: obs, _ = env.reset()
    
    frame = process_frame(obs)
    frame_t = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        mu, _ = vae.encode(frame_t)
        recon = vae.decode(mu)
    
    print(f"  -> VAE Debug: frame shape={frame.shape}, mu shape={mu.shape}, recon shape={recon.shape}")
    
    # Save side-by-side comparison
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Original (CHW -> HWC for matplotlib)
    orig_img = np.transpose(frame, (1, 2, 0))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Frame")
    axes[0].axis('off')
    
    # Reconstructed
    recon_img = recon.cpu().squeeze(0).numpy()
    recon_img = np.transpose(recon_img, (1, 2, 0))
    axes[1].imshow(recon_img)
    axes[1].set_title("VAE Reconstruction")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"vae_debug_iter_{loop_idx}.png", dpi=150)
    plt.close()
    print(f"  -> Saved vae_debug_iter_{loop_idx}.png (check if ball/paddle visible!)")
    env.close()
    
    # --- 1. RECORD REAL EPISODE ---
    env = get_env()
    obs, _ = env.reset()
    real_frames = []
    
    # Capture the first frame to start the dream later
    start_frame = process_frame(obs) 
    
    # Run for 200 frames of real gameplay
    for i in range(200):
        # Render for human eyes (RGB) - Higher quality
        # We capture the RAW environment output, not the resized one, 
        # so you can see if the ball exists in the game.
        raw_render = env.render() 
        real_frames.append(np.transpose(raw_render, (2, 0, 1))) # (C, H, W)
        
        # Action Logic (Force Fire, etc)
        if i == 0: action = 1
        else: action = env.action_space.sample() # Random action for viz
        
        obs, _, done, _, _ = env.step(action)
        if done: break

    # Log REAL Video (MP4)
    try:
        wandb.log({"real_gameplay": wandb.Video(np.array(real_frames), fps=30, format="mp4")})
    except Exception as e:
        print(f"  -> Warning: Could not save real gameplay video to W&B: {e}")

    # --- 2. RECORD DREAM SEQUENCE ---
    # Start from that first frame we captured
    z = torch.tensor(start_frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): mu, _ = vae.encode(z)
    z = mu.unsqueeze(1)  # Shape (1, 1, latent_dim)
    h = None
    
    dream_frames = []
    z_values = []  # Track Z changes
    for step in range(50): # 50 frames of hallucination
        with torch.no_grad():
            # Decode: This shows us what the VAE sees
            img = vae.decode(z.view(1, 1024)).cpu().squeeze(0).numpy()  # z is (1, 1, 1024) -> (1, 1024)
            # Convert from [0,1] to [0,255] for display
            img = (img * 255).astype(np.uint8)
            dream_frames.append(img)
            z_values.append(z.cpu().numpy().flatten().copy())
            
            # Predict Next Step
            # Predict Next Step
            a = torch.zeros(1, 1, 4).to(DEVICE)
            
            # Wiggle Logic: Right for 5 frames, then Left for 5 frames
            if (step // 5) % 2 == 0:
                action_idx = 2 # RIGHT
            else:
                action_idx = 3 # LEFT
                
            a[0, 0, action_idx] = 1.0
            out, h = rnn(z, a, h)
            z_next = rnn.fc_mu(out)
            
            # Debug: Check if Z is changing
            z_change = torch.abs(z_next - z).mean().item()
            if step % 10 == 0:
                print(f"    Dream Step {step}: Z_change={z_change:.6f}, Z_range=[{z_next.min():.2f}, {z_next.max():.2f}]")
            
            z = z_next
    
    print(f"  -> Dream Z trajectory: Min={np.array(z_values).min():.3f}, Max={np.array(z_values).max():.3f}")

# ==========================================
# MAIN
# ==========================================
def main():
    test = True 
    wandb.init(project="breakout-world-models-fixed")
    
    LATENT_DIM = 1024  # Reduced encoder compression to preserve ball
    RNN_HIDDEN = 256
    
    # Initialize with consistent sizes
    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    
    rnn = RobustRNN(latent_dim=LATENT_DIM, hidden_size=RNN_HIDDEN).to(DEVICE)
    
    # Controller Input = Latent Size + RNN Hidden Size
    controller = Controller(input_dim=LATENT_DIM + RNN_HIDDEN).to("cpu")

    #load iter_1 models


    
    for loop in range(2,ITERATIONS): 
        print(f"\n=== ITERATION {loop+1} ===")
        
        # 1. Harvest & VAE Data
        if test == False:
            print("[PHASE 1] Harvesting new data...")
            images, z_data, a_data = harvest_data(vae, rnn, controller, HARVEST_EPISODES, loop)
        

        #load the previous vae model if not the first iteration
        if test == True:
            vae.load_state_dict(torch.load(f"vae_iter_{loop}.pth"))
            print(f"  -> Loaded VAE model from vae_iter_{loop}.pth")
        else:
            train_vae(vae, images, VAE_EPOCHS)
            torch.save(vae.state_dict(), f"vae_iter_{loop}.pth")
            print(f"  -> Saved VAE model at vae_iter_{loop}.pth")

            # --- CRITICAL FIX: RE-ENCODE IMAGES WITH TRAINED VAE ---
            print("[INFO] Re-encoding images with trained VAE for RNN...")
            
            # 1. SAVE A COPY of the old structure so we know episode lengths
            old_z_data = z_data 
            new_z_data = []      # Store new vectors here
            
            vae.eval() 
            current_idx = 0
            
            with torch.no_grad():
                for old_episode in old_z_data:
                    # Get the length of this specific episode
                    ep_len = len(old_episode)
                    
                    # Extract the raw images for this episode
                    ep_imgs = images[current_idx : current_idx + ep_len]
                    current_idx += ep_len
                    
                    # Convert to tensor
                    ep_imgs_tensor = torch.tensor(ep_imgs, dtype=torch.float32).to(DEVICE)
                    
                    # Batch encode this episode to avoid OOM
                    ep_z_list = []
                    for b in range(0, len(ep_imgs), 32):
                        chunk = ep_imgs_tensor[b:b+32]
                        mu, _ = vae.encode(chunk)
                        ep_z_list.append(mu.cpu().numpy())
                    
                    # Stack and save
                    if len(ep_z_list) > 0:
                        new_z_data.append(np.concatenate(ep_z_list, axis=0))
            
            # 2. OVERWRITE z_data with the new, trained vectors
            z_data = new_z_data 
            print(f"  -> Re-encoded {len(z_data)} episodes.")
            vae.train() 
            # -------------------------------------------------------
        
        # 3. Log Visuals
        log_visuals(vae, rnn, loop)
        
        # 4. Train RNN
        if test == True:
            rnn.load_state_dict(torch.load(f"rnn_iter_{loop}.pth"))
            print(f"  -> Loaded RNN model from rnn_iter_{loop}.pth")
        else:
            train_rnn(rnn, z_data, a_data, RNN_EPOCHS)
            torch.save(rnn.state_dict(), f"rnn_iter_{loop}.pth")
            print(f"  -> Saved RNN model at rnn_iter_{loop}.pth")

        
        # 4.5 Record RNN Dream
        print("[PHASE 2.5] Recording RNN dream...")
        env = get_env()
        obs, _ = env.reset()
        frame = process_frame(obs)
        seed_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = vae.encode(seed_tensor)
        z = mu.unsqueeze(1)  # Shape: (1, 1, 1024)
        h = None
        
        rnn_dream_frames = []
        for step in range(50):
            with torch.no_grad():
                img = vae.decode(z.view(1, 1024)).cpu().squeeze(0).numpy()  # [3, 64, 64] in [0,1]
                img = np.transpose(img, (1, 2, 0))  # CHW to HWC for display
                img = (img * 255).astype(np.uint8)  # Scale to [0,255]
                rnn_dream_frames.append(img)
                
                # Use FIRE action to keep similar to log_visuals
                a = torch.zeros(1, 1, 4).to(DEVICE)
                a[0, 0, 1] = 1.0
                out, h = rnn(z, a, h)
                z = rnn.fc_mu(out)
                z = torch.clamp(z, -3.0, 3.0)
        
        # Log RNN dream
        rnn_dream_array = np.array(rnn_dream_frames)  # Currently [50, 64, 64, 3] (HWC)
        rnn_dream_array = np.clip(rnn_dream_array, 0, 255).astype(np.uint8)
        
        # --- FIX: Transpose to [Time, Channels, Height, Width] ---
        # 0->0 (Time), 3->1 (Channels), 1->2 (Height), 2->3 (Width)
        rnn_dream_array = np.transpose(rnn_dream_array, (0, 3, 1, 2)) 
        # Now shape is [50, 3, 64, 64]
        
        try:
            wandb.log({"rnn_dream": wandb.Video(rnn_dream_array, fps=10, format="mp4")})
        except Exception as e:
            print(f"  -> Warning: Could not save RNN dream to W&B: {e}")
        env.close()
        
        # Save VAE and RNN models
        
        
        
        
        # 5. Evolve Agent
        controller = train_controller(vae, rnn, controller, CMA_GENERATIONS, loop)

        test = False

if __name__ == "__main__":
    main()