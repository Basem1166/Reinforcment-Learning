"""
SAC training for CarRacing with GRAYSCALE STACKS (no CNN encoder).
This script bypasses the encoder and uses downscaled, flattened grayscale frames.
Purpose: Diagnose if the CNN encoder is the learning bottleneck.
"""
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import argparse
import os
from sac_agent import SACAgent
from buffer import PrioritizedReplayBuffer
import config

# Grayscale preprocessing: smaller resolution for direct MLP input
IMG_SIZE = 42  # Downscale 96x96 -> 42x42 for tractable MLP input
NUM_FRAMES = 4
OBS_DIM = IMG_SIZE * IMG_SIZE * NUM_FRAMES  # 42*42*4 = 7056

def preprocess_frame(frame):
    """Convert RGB frame to grayscale and downscale using proper luminance weights."""
    # Proper luminance-based grayscale (ITU-R BT.601)
    gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
    # Downscale using tensor operations (faster)
    gray_tensor = torch.as_tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    downscaled = F.interpolate(gray_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    # Normalize to [0, 1] - CarRacing-v3 returns float32 in [0, 255]
    normalized = downscaled.squeeze().numpy() / 255.0
    return normalized

def stack_to_obs(frame_stack):
    """Convert frame stack to flat observation vector."""
    stacked = np.stack([preprocess_frame(f) for f in frame_stack], axis=0)  # (4, 42, 42)
    return stacked.flatten()  # (7056,)

def train_grayscale(use_wandb=True):
    """Train SAC with grayscale stacks (no encoder)."""
    cfg = config.SAC_CARRACING
    hyperparams = cfg['hyperparameters']
    training_cfg = cfg['training']
    wandb_cfg = cfg['wandb']
    
    env = gym.make('CarRacing-v3', continuous=True)
    eval_env = gym.make('CarRacing-v3', continuous=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GRAYSCALE MODE: {NUM_FRAMES} frames x {IMG_SIZE}x{IMG_SIZE} = {OBS_DIM} dims (no encoder)")
    
    # Initialize wandb
    if use_wandb and wandb_cfg['enabled']:
        wandb.init(
            project=wandb_cfg['project'],
            entity=wandb_cfg['entity'],
            name="SAC-CarRacing-Grayscale",
            config={**hyperparams, **training_cfg, 'mode': 'grayscale', 'obs_dim': OBS_DIM}
        )
    
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    
    # Create agent with flattened grayscale input
    agent = SACAgent(OBS_DIM, action_dim, action_scale, hyperparams, device)
    
    # Prioritized replay buffer for better sample efficiency
    replay_buffer = PrioritizedReplayBuffer(
        obs_dim=OBS_DIM,
        action_dim=action_dim,
        max_size=hyperparams['buffer_size'],
        alpha=0.6,
        beta_start=0.4,
        beta_frames=training_cfg['total_timesteps']
    )
    
    state, _ = env.reset()
    frame_stack = [state] * NUM_FRAMES
    episode_reward = 0
    episode_num = 0
    best_reward = -float('inf')
    
    os.makedirs('models', exist_ok=True)
    print(f"Starting GRAYSCALE training for {training_cfg['total_timesteps']} timesteps...")
    
    for t in range(training_cfg['total_timesteps']):
        obs = stack_to_obs(frame_stack)
        
        if t < training_cfg['start_timesteps']:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, eval_mode=False)
        
        action = np.clip(action, env.action_space.low, env.action_space.high)
        if np.any(np.isnan(action)):
            action = env.action_space.sample()
        
        # Action repeat
        action_repeat = training_cfg.get('action_repeat', 4)
        total_reward = 0
        for _ in range(action_repeat):
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        next_frame_stack = frame_stack[1:] + [next_state]
        next_obs = stack_to_obs(next_frame_stack)
        
        replay_buffer.add(obs, action, total_reward, next_obs, float(done))
        
        frame_stack = next_frame_stack
        state = next_state
        episode_reward += total_reward
        
        # Update
        if t >= training_cfg['start_timesteps']:
            # PER returns: states, actions, rewards, next_states, dones, weights, indices
            states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(
                hyperparams['batch_size'], device
            )
            batch = (states, actions, rewards, next_states, dones)
            metrics = agent.update(batch, weights=weights)
            
            # Update priorities based on TD errors
            replay_buffer.update_priorities(indices, metrics['td_errors'])
            
            # Log to wandb
            if use_wandb and wandb_cfg['enabled'] and t % 1000 == 0:
                wandb.log({
                    'timestep': t,
                    'q1_loss': metrics['q1_loss'],
                    'q2_loss': metrics['q2_loss'],
                    'policy_loss': metrics['policy_loss'],
                    'alpha': metrics['alpha'],
                })
        
        if done:
            print(f"T={t+1}, Ep={episode_num+1}, R={episode_reward:.1f}")
            if use_wandb and wandb_cfg['enabled']:
                wandb.log({
                    'timestep': t,
                    'episode_reward': episode_reward,
                    'episode': episode_num + 1,
                })
            state, _ = env.reset()
            frame_stack = [state] * NUM_FRAMES
            episode_reward = 0
            episode_num += 1
        
        if (t + 1) % training_cfg['eval_frequency'] == 0:
            eval_r = evaluate(agent, eval_env)
            print(f">>> Eval at {t+1}: {eval_r:.1f}")
            if use_wandb and wandb_cfg['enabled']:
                wandb.log({'timestep': t, 'eval_reward': eval_r})
            if eval_r > best_reward:
                best_reward = eval_r
                torch.save(agent.policy.state_dict(), 'models/SAC_CarRacing_grayscale_best.pth')
    
    env.close()
    eval_env.close()
    if use_wandb and wandb_cfg['enabled']:
        wandb.finish()
    print("Training complete!")

def evaluate(agent, env, n_episodes=3):
    """Evaluate agent."""
    total = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        frame_stack = [state] * NUM_FRAMES
        done = False
        while not done:
            obs = stack_to_obs(frame_stack)
            action = np.clip(agent.select_action(obs, eval_mode=True),
                           env.action_space.low, env.action_space.high)
            for _ in range(4):  # Action repeat
                state, r, term, trunc, _ = env.step(action)
                total += r
                done = term or trunc
                if done:
                    break
            frame_stack = frame_stack[1:] + [state]
    return total / n_episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    train_grayscale(use_wandb=not args.no_wandb)
