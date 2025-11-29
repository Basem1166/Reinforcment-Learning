import gymnasium as gym
import numpy as np
import torch
import wandb
from ppo_agent import PPOAgent
from config import HYPERPARAMETERS, ENV_CONFIG, WANDB_CONFIG

def train():
    # 1. Initialize W&B
    wandb.init(
        project=WANDB_CONFIG["project_name"],
        name=f"PPO_{ENV_CONFIG['env_name']}",
        config=HYPERPARAMETERS,
        tags=["PPO", "Discrete", "On-Policy"],
        monitor_gym=True,
        save_code=True
    )
    
    env_name = ENV_CONFIG["env_name"]
    env = gym.make(env_name)
    agent = PPOAgent(env, HYPERPARAMETERS)
    
    # PPO Parameters
    total_timesteps = ENV_CONFIG["total_timesteps"]
    batch_size = 2048 # PPO collects a large buffer before updating
    
    step = 0
    episode_count = 0
    state, _ = env.reset()
    episode_reward = 0
    
    print(f"--- Starting PPO Training on {env_name} ---")
    
    while step < total_timesteps:
        
        # 1. Collection Phase (Algorithm Line 3)
        # Run policy for T steps (batch_size)
        for _ in range(batch_size):
            # Get Action & Log Prob
            action, log_prob = agent.get_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            # We need log_prob for the PPO ratio calculation
            agent.store_transition((state, action, log_prob, reward, done, next_state))
            
            state = next_state
            episode_reward += reward
            step += 1
            
            if done:
                episode_count += 1
                wandb.log({"train/episode_reward": episode_reward, "global_step": step})
                print(f"Step: {step}, Episode: {episode_count}, Reward: {episode_reward}")
                
                state, _ = env.reset()
                episode_reward = 0
                
                if step >= total_timesteps:
                    break
        
        # 2. Update Phase (Algorithm Line 6-7)
        # Agent computes advantages and optimizes surrogate loss
        loss = agent.update()
        wandb.log({"train/loss": loss, "global_step": step})
        
    # 3. Save Model
    torch.save(agent.actor.state_dict(), "ppo_actor.pth")
    torch.save(agent.critic.state_dict(), "ppo_critic.pth")
    
    # Upload to W&B
    artifact = wandb.Artifact('ppo_model', type='model')
    artifact.add_file('ppo_actor.pth')
    artifact.add_file('ppo_critic.pth')
    wandb.log_artifact(artifact)

    env.close()
    wandb.finish()
    print("PPO Training Complete.")

if __name__ == "__main__":
    train()