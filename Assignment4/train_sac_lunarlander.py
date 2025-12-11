"""
SAC training for LunarLander-v3 (Continuous).
Simple vector observation environment - no preprocessing needed.
"""
import gymnasium as gym
import torch
import numpy as np
import wandb
import argparse
import os
from sac_agent import SACAgent
from buffer import ReplayBuffer
import config


def train_lunarlander(use_wandb=True):
    """Train SAC on LunarLander-v3 continuous."""
    cfg = config.SAC_LUNARLANDER
    hyperparams = cfg['hyperparameters']
    training_cfg = cfg['training']
    wandb_cfg = cfg['wandb']
    
    env = gym.make('LunarLander-v3', continuous=True)
    eval_env = gym.make('LunarLander-v3', continuous=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    obs_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.shape[0]    # 2
    action_scale = float(env.action_space.high[0])
    
    print(f"LunarLander-v3: obs_dim={obs_dim}, action_dim={action_dim}, action_scale={action_scale}")
    
    # Initialize wandb
    if use_wandb and wandb_cfg['enabled']:
        wandb.init(
            project=wandb_cfg['project'],
            entity=wandb_cfg['entity'],
            name="SAC-LunarLander",
            config={**hyperparams, **training_cfg, 'env': 'LunarLander-v3'}
        )
    
    # Create agent
    agent = SACAgent(obs_dim, action_dim, action_scale, hyperparams, device)
    
    # Simple replay buffer (LunarLander is easy, no need for PER)
    replay_buffer = ReplayBuffer(
        obs_shape=obs_dim,
        action_dim=action_dim,
        max_size=hyperparams['buffer_size']
    )
    
    state, _ = env.reset()
    episode_reward = 0
    episode_num = 0
    best_reward = -float('inf')
    
    os.makedirs('models', exist_ok=True)
    print(f"Starting SAC training on LunarLander for {training_cfg['total_timesteps']} timesteps...")
    
    for t in range(training_cfg['total_timesteps']):
        # Select action
        if t < training_cfg['start_timesteps']:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, eval_mode=False)
        
        # Clip and validate action
        action = np.clip(action, env.action_space.low, env.action_space.high)
        if np.any(np.isnan(action)):
            action = env.action_space.sample()
        
        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        replay_buffer.add(state, action, reward, next_state, float(done))
        
        state = next_state
        episode_reward += reward
        
        # Update agent
        if t >= training_cfg['start_timesteps']:
            batch = replay_buffer.sample(hyperparams['batch_size'], device)
            metrics = agent.update(batch)
            
            # Log to wandb
            if use_wandb and wandb_cfg['enabled'] and t % 1000 == 0:
                wandb.log({
                    'timestep': t,
                    'q1_loss': metrics['q1_loss'],
                    'q2_loss': metrics['q2_loss'],
                    'policy_loss': metrics['policy_loss'],
                    'alpha': metrics['alpha'],
                })
        
        # Episode done
        if done:
            print(f"T={t+1}, Ep={episode_num+1}, R={episode_reward:.1f}")
            if use_wandb and wandb_cfg['enabled']:
                wandb.log({
                    'timestep': t,
                    'episode_reward': episode_reward,
                    'episode': episode_num + 1,
                })
            state, _ = env.reset()
            episode_reward = 0
            episode_num += 1
        
        # Evaluate
        if (t + 1) % training_cfg['eval_frequency'] == 0:
            eval_r = evaluate(agent, eval_env)
            print(f">>> Eval at {t+1}: {eval_r:.1f}")
            if use_wandb and wandb_cfg['enabled']:
                wandb.log({'timestep': t, 'eval_reward': eval_r})
            if eval_r > best_reward:
                best_reward = eval_r
                torch.save(agent.policy.state_dict(), 'models/SAC_LunarLander_best.pth')
                print(f"    New best model saved! ({eval_r:.1f})")
    
    # Save final model
    torch.save(agent.policy.state_dict(), 'models/SAC_LunarLander_final.pth')
    
    env.close()
    eval_env.close()
    if use_wandb and wandb_cfg['enabled']:
        wandb.finish()
    print(f"Training complete! Best eval reward: {best_reward:.1f}")


def evaluate(agent, env, n_episodes=10):
    """Evaluate agent over n episodes."""
    total_reward = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state, eval_mode=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_reward += episode_reward
    return total_reward / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    train_lunarlander(use_wandb=not args.no_wandb)