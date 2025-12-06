import gymnasium as gym
import torch
import numpy as np
import argparse
import os
import csv
from sac_agent import SACAgent
from ppo_agent import PPOAgent
from td3_agent import TD3Agent
import config

def test_and_record(algorithm, env_name, model_path, num_episodes=100, record_video=True):
    """Test trained agent and optionally record video"""
    
    # Get configuration
    config_map = {
        ('sac', 'lunarlander'): config.SAC_LUNARLANDER,
        ('sac', 'carracing'): config.SAC_CARRACING,
        ('ppo', 'lunarlander'): config.PPO_LUNARLANDER,
        ('ppo', 'carracing'): config.PPO_CARRACING,
        ('td3', 'lunarlander'): config.TD3_LUNARLANDER,
        ('td3', 'carracing'): config.TD3_CARRACING,
    }
    
    cfg = config_map[(algorithm.lower(), env_name.lower())]
    env_cfg = cfg['env']
    hyperparams = cfg['hyperparameters']
    
    # Create environment
    if record_video:
        os.makedirs('videos', exist_ok=True)
        env = gym.make(
            env_cfg['name'], 
            continuous=env_cfg['continuous'],
            render_mode='rgb_array'
        )
        env = gym.wrappers.RecordVideo(
            env, 
            f"videos/{algorithm}_{env_cfg['name']}_test",
            episode_trigger=lambda x: x % 10 == 0  # Record every 10th episode
        )
    else:
        env = gym.make(env_cfg['name'], continuous=env_cfg['continuous'])
    
    # Get environment info
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent based on algorithm
    if algorithm.lower() == 'sac':
        action_scale = float(env.action_space.high[0])
        agent = SACAgent(obs_dim, action_dim, action_scale, hyperparams, device)
    elif algorithm.lower() == 'ppo':
        agent = PPOAgent(obs_dim, action_dim, hyperparams, device)
    elif algorithm.lower() == 'td3':
        max_action = float(env.action_space.high[0])
        agent = TD3Agent(obs_dim, action_dim, max_action, hyperparams, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    agent.load(model_path)
    
    # Test agent
    print(f"Testing {algorithm.upper()} on {env_cfg['name']} for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 10): {avg_reward:.2f}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print("\n" + "="*50)
    print(f"Test Results for {algorithm.upper()} on {env_cfg['name']}")
    print("="*50)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print("="*50)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_file = f"results/{algorithm}_{env_cfg['name']}_test_results.csv"
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Length'])
        for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
            writer.writerow([i+1, reward, length])
    
    print(f"\nResults saved to {results_file}")
    
    # Save summary
    summary_file = f"results/{algorithm}_{env_cfg['name']}_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Test Results for {algorithm.upper()} on {env_cfg['name']}\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Number of Episodes: {num_episodes}\n")
        f.write(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Min Reward: {min_reward:.2f}\n")
        f.write(f"Max Reward: {max_reward:.2f}\n")
        f.write(f"Mean Episode Length: {mean_length:.2f}\n")
    
    print(f"Summary saved to {summary_file}")
    
    env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'mean_length': mean_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained agent')
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['sac', 'ppo', 'td3'],
                        help='Algorithm to test')
    parser.add_argument('--env', type=str, required=True,
                        choices=['lunarlander', 'carracing'],
                        help='Environment to test on')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of test episodes')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    
    args = parser.parse_args()
    
    test_and_record(
        algorithm=args.algorithm,
        env_name=args.env,
        model_path=args.model,
        num_episodes=args.episodes,
        record_video=not args.no_video
    )
