import gymnasium as gym
import torch
import numpy as np
import wandb
import argparse
import os
from sac_agent import SACAgent
from buffer import ReplayBuffer
from models import CarRacingEncoder
import config

def train_sac(env_name='lunarlander', use_wandb=True):
    """Train SAC agent on specified environment"""
    
    # Get configuration
    if env_name == 'lunarlander':
        cfg = config.SAC_LUNARLANDER
    elif env_name == 'carracing':
        cfg = config.SAC_CARRACING
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    env_cfg = cfg['env']
    hyperparams = cfg['hyperparameters']
    training_cfg = cfg['training']
    wandb_cfg = cfg['wandb']
    
    # Initialize environment
    env = gym.make(env_cfg['name'], continuous=env_cfg['continuous'])
    eval_env = gym.make(env_cfg['name'], continuous=env_cfg['continuous'])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CarRacing uses image observations; others use vectors
    use_cnn_encoder = env_cfg['name'].lower().startswith('carracing')

    if use_cnn_encoder:
        # Image shape e.g. (96, 96, 3)
        obs_shape = env.observation_space.shape  # (H, W, C)
        action_dim = env.action_space.shape[0]
        action_scale = float(env.action_space.high[0])

        # CNN encoder + SAC agent with feature_dim observations
        encoder = CarRacingEncoder(feature_dim=hyperparams.get('encoder_feature_dim', 256)).to(device)
        feature_dim = encoder.fc.out_features

        agent = SACAgent(feature_dim, action_dim, action_scale, hyperparams, device)
        replay_buffer = ReplayBuffer(obs_shape, action_dim, hyperparams['buffer_size'])
    else:
        # Vector observations
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_scale = float(env.action_space.high[0])

        encoder = None
        agent = SACAgent(obs_dim, action_dim, action_scale, hyperparams, device)
        replay_buffer = ReplayBuffer(obs_dim, action_dim, hyperparams['buffer_size'])
    
    # Initialize wandb
    if use_wandb and wandb_cfg['enabled']:
        wandb.init(
            project=wandb_cfg['project'],
            entity=wandb_cfg['entity'],
            name=f"SAC-{env_cfg['name']}",
            config={
                'algorithm': 'SAC',
                'environment': env_cfg['name'],
                **hyperparams,
                **training_cfg
            }
        )
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    best_eval_reward = -float('inf')
    
    print(f"Starting SAC training on {env_cfg['name']}...")
    print(f"Total timesteps: {training_cfg['total_timesteps']}")
    
    for t in range(training_cfg['total_timesteps']):
        episode_timesteps += 1
        
        # Select action
        if t < training_cfg['start_timesteps']:
            # Random exploration at the beginning
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, eval_mode=False)
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition (raw states: images or vectors)
        replay_buffer.add(state, action, reward, next_state, float(done))
        
        state = next_state
        episode_reward += reward
        
        # Update agent
        if t >= training_cfg['start_timesteps']:
            batch = replay_buffer.sample(hyperparams['batch_size'], device)

            if use_cnn_encoder:
                # Unpack and encode image observations before SAC update
                states, actions, rewards, next_states, dones = batch
                with torch.no_grad():
                    states_feat = encoder(states)
                    next_states_feat = encoder(next_states)
                encoded_batch = (states_feat, actions, rewards, next_states_feat, dones)
                metrics = agent.update(encoded_batch)
            else:
                metrics = agent.update(batch)
            
            # Log training metrics
            if use_wandb and wandb_cfg['enabled'] and t % 1000 == 0:
                wandb.log({
                    'timestep': t,
                    'q1_loss': metrics['q1_loss'],
                    'q2_loss': metrics['q2_loss'],
                    'policy_loss': metrics['policy_loss'],
                    'alpha': metrics['alpha'],
                })
        
        # End of episode
        if done:
            print(f"Timestep: {t+1}, Episode: {episode_num+1}, Reward: {episode_reward:.2f}, Length: {episode_timesteps}")
            
            if use_wandb and wandb_cfg['enabled']:
                wandb.log({
                    'timestep': t,
                    'episode_reward': episode_reward,
                    'episode_length': episode_timesteps,
                })
            
            # Reset
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Evaluation
        if (t + 1) % training_cfg['eval_frequency'] == 0:
            eval_reward = evaluate(agent, eval_env, training_cfg['eval_episodes'], encoder if use_cnn_encoder else None)
            print(f"Evaluation at timestep {t+1}: Avg Reward: {eval_reward:.2f}")
            
            if use_wandb and wandb_cfg['enabled']:
                wandb.log({
                    'timestep': t,
                    'eval_reward': eval_reward,
                })
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(f"models/SAC_{env_cfg['name']}_best.pth")
                print(f"New best model saved with reward: {eval_reward:.2f}")
        
        # Save checkpoint
        if (t + 1) % training_cfg['save_frequency'] == 0:
            agent.save(f"models/SAC_{env_cfg['name']}_checkpoint_{t+1}.pth")
    
    # Save final model
    agent.save(f"models/SAC_{env_cfg['name']}_final.pth")
    print("Training completed!")
    
    if use_wandb and wandb_cfg['enabled']:
        wandb.finish()
    
    env.close()
    eval_env.close()


def evaluate(agent, env, num_episodes=10, encoder=None):
    """Evaluate agent performance.

    If encoder is provided (CarRacing), it will be used to encode observations
    before passing them to the agent.
    """
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if encoder is not None:
                # Encode single observation for policy
                with torch.no_grad():
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
                    # encoder expects (H,W,C) or (B,H,W,C)
                    feat = encoder(state_tensor).cpu().numpy()[0]
                action = agent.select_action(feat, eval_mode=True)
            else:
                action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAC agent')
    parser.add_argument('--env', type=str, default='lunarlander', 
                        choices=['lunarlander', 'carracing'],
                        help='Environment to train on')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    train_sac(env_name=args.env, use_wandb=not args.no_wandb)
