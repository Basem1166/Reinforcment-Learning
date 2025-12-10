import gymnasium as gym
import torch
import numpy as np
import wandb
import argparse
import os
from ppo_agent import PPOAgent
from models import CarRacingEncoder
import config

def train_ppo(env_name='lunarlander', use_wandb=True):
    """Train PPO agent on specified environment"""
    
    # Get configuration
    if env_name == 'lunarlander':
        cfg = config.PPO_LUNARLANDER
    elif env_name == 'carracing':
        cfg = config.PPO_CARRACING
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

    # Determine if we are using image observations (CarRacing)
    use_cnn_encoder = env_cfg['name'].lower().startswith('carracing')

    # Get environment info
    if use_cnn_encoder:
        obs_shape = env.observation_space.shape  # (H, W, C)
        action_dim = env.action_space.shape[0]

        # Initialize encoder and PPO on encoded features
        encoder_feature_dim = hyperparams.get('encoder_feature_dim', 256)
        encoder = CarRacingEncoder(feature_dim=encoder_feature_dim).to(device)
        obs_dim = encoder_feature_dim
    else:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        encoder = None
    
    # Initialize agent
    agent = PPOAgent(obs_dim, action_dim, hyperparams, device)
    
    # Initialize wandb
    if use_wandb and wandb_cfg['enabled']:
        wandb.init(
            project=wandb_cfg['project'],
            entity=wandb_cfg['entity'],
            name=f"PPO-{env_cfg['name']}",
            config={
                'algorithm': 'PPO',
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
    rollout_length = hyperparams['rollout_length']
    
    print(f"Starting PPO training on {env_cfg['name']}...")
    print(f"Total timesteps: {training_cfg['total_timesteps']}")
    
    for t in range(training_cfg['total_timesteps']):
        episode_timesteps += 1
        
        # Select action
        if use_cnn_encoder:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                feat = encoder(state_tensor).cpu().numpy()[0]
            action, log_prob, value = agent.select_action(feat, eval_mode=False)
        else:
            action, log_prob, value = agent.select_action(state, eval_mode=False)
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition in rollout buffer
        agent.store_transition(state, action, reward, done, log_prob, value)
        
        state = next_state
        episode_reward += reward
        
        # Update agent after collecting rollout
        if len(agent.buffer) >= rollout_length:
            # For CarRacing, pass encoded next_state into update for next_value
            if use_cnn_encoder:
                with torch.no_grad():
                    next_state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                    next_feat = encoder(next_state_tensor).cpu().numpy()[0]
                metrics = agent.update(next_feat)
            else:
                metrics = agent.update(state)
            
            # Log training metrics
            if use_wandb and wandb_cfg['enabled']:
                wandb.log({
                    'timestep': t,
                    'policy_loss': metrics['policy_loss'],
                    'value_loss': metrics['value_loss'],
                    'entropy': metrics['entropy'],
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
            eval_reward = evaluate(agent, eval_env, training_cfg['eval_episodes'],
                                   encoder if use_cnn_encoder else None,
                                   device)
            print(f"Evaluation at timestep {t+1}: Avg Reward: {eval_reward:.2f}")
            
            if use_wandb and wandb_cfg['enabled']:
                wandb.log({
                    'timestep': t,
                    'eval_reward': eval_reward,
                })
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(f"models/PPO_{env_cfg['name']}_best.pth")
                print(f"New best model saved with reward: {eval_reward:.2f}")
        
        # Save checkpoint
        if (t + 1) % training_cfg['save_frequency'] == 0:
            agent.save(f"models/PPO_{env_cfg['name']}_checkpoint_{t+1}.pth")
    
    # Save final model
    agent.save(f"models/PPO_{env_cfg['name']}_final.pth")
    print("Training completed!")
    
    if use_wandb and wandb_cfg['enabled']:
        wandb.finish()
    
    env.close()
    eval_env.close()


def evaluate(agent, env, num_episodes=10, encoder=None, device=None):
    """Evaluate agent performance"""
    total_reward = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if encoder is not None:
                with torch.no_grad():
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
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
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--env', type=str, default='lunarlander', 
                        choices=['lunarlander', 'carracing'],
                        help='Environment to train on')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    train_ppo(env_name=args.env, use_wandb=not args.no_wandb)
