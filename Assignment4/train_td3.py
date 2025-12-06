import gymnasium as gym
import torch
import numpy as np
import wandb
import argparse
import os
from td3_agent import TD3Agent
from buffer import ReplayBuffer
import config

def train_td3(env_name='lunarlander', use_wandb=True):
    """Train TD3 agent on specified environment"""
    
    # Get configuration
    if env_name == 'lunarlander':
        cfg = config.TD3_LUNARLANDER
    elif env_name == 'carracing':
        cfg = config.TD3_CARRACING
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    env_cfg = cfg['env']
    hyperparams = cfg['hyperparameters']
    training_cfg = cfg['training']
    wandb_cfg = cfg['wandb']
    
    # Initialize environment
    env = gym.make(env_cfg['name'], continuous=env_cfg['continuous'])
    eval_env = gym.make(env_cfg['name'], continuous=env_cfg['continuous'])
    
    # Get environment info
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    agent = TD3Agent(obs_dim, action_dim, max_action, hyperparams, device)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(obs_dim, action_dim, hyperparams['buffer_size'])
    
    # Initialize wandb
    if use_wandb and wandb_cfg['enabled']:
        wandb.init(
            project=wandb_cfg['project'],
            entity=wandb_cfg['entity'],
            name=f"TD3-{env_cfg['name']}",
            config={
                'algorithm': 'TD3',
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
    
    print(f"Starting TD3 training on {env_cfg['name']}...")
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
        
        # Store transition
        replay_buffer.add(state, action, reward, next_state, float(done))
        
        state = next_state
        episode_reward += reward
        
        # Update agent
        if t >= training_cfg['start_timesteps']:
            batch = replay_buffer.sample(hyperparams['batch_size'], device)
            metrics = agent.update(batch)
            
            # Log training metrics
            if use_wandb and wandb_cfg['enabled'] and t % 1000 == 0:
                wandb.log({
                    'timestep': t,
                    'critic_loss': metrics['critic_loss'],
                    'policy_loss': metrics['policy_loss'],
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
            eval_reward = evaluate(agent, eval_env, training_cfg['eval_episodes'])
            print(f"Evaluation at timestep {t+1}: Avg Reward: {eval_reward:.2f}")
            
            if use_wandb and wandb_cfg['enabled']:
                wandb.log({
                    'timestep': t,
                    'eval_reward': eval_reward,
                })
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(f"models/TD3_{env_cfg['name']}_best.pth")
                print(f"New best model saved with reward: {eval_reward:.2f}")
        
        # Save checkpoint
        if (t + 1) % training_cfg['save_frequency'] == 0:
            agent.save(f"models/TD3_{env_cfg['name']}_checkpoint_{t+1}.pth")
    
    # Save final model
    agent.save(f"models/TD3_{env_cfg['name']}_final.pth")
    print("Training completed!")
    
    if use_wandb and wandb_cfg['enabled']:
        wandb.finish()
    
    env.close()
    eval_env.close()


def evaluate(agent, env, num_episodes=10):
    """Evaluate agent performance"""
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TD3 agent')
    parser.add_argument('--env', type=str, default='lunarlander', 
                        choices=['lunarlander', 'carracing'],
                        help='Environment to train on')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    train_td3(env_name=args.env, use_wandb=not args.no_wandb)
