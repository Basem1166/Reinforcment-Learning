import gymnasium as gym
import numpy as np
import torch
import wandb
from sac_agent import SACAgent
from buffer import ReplayBuffer
from config import HYPERPARAMETERS, ENV_CONFIG, WANDB_CONFIG

def train():
    # 1. Initialize W&B
    wandb.init(
        project=WANDB_CONFIG["project_name"],
        name=f"SAC_{ENV_CONFIG['env_name']}",
        config=HYPERPARAMETERS,
        tags=["SAC", "Continuous", "Off-Policy"],
        monitor_gym=True,
        save_code=True
    )
    
    # 2. Setup Environment
    # SAC is designed for Continuous actions (e.g., Pendulum-v1, MountainCarContinuous-v0)
    env_name = ENV_CONFIG["env_name"]
    env = gym.make(env_name)
    
    # 3. Initialize Agent & Buffer
    agent = SACAgent(env, HYPERPARAMETERS)
    memory = ReplayBuffer(HYPERPARAMETERS["buffer_size"])
    
    # 4. Training Loop
    batch_size = HYPERPARAMETERS["batch_size"]
    total_timesteps = ENV_CONFIG["total_timesteps"]
    
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    
    print(f"--- Starting SAC Training on {env_name} ---")
    
    for step in range(total_timesteps):
        
        # SAC policies inherently handle exploration via the "reparameterization trick"
        # so we don't need epsilon-greedy here.
        action = agent.get_action(state)
        
        # Step Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store in Replay Buffer
        # For SAC, 'done' usually shouldn't include TimeLimit (truncated) 
        # but for simple assignments, storing 'done' as is is acceptable.
        mask = 1 if terminated else 0 
        memory.push(state, action, reward, next_state, mask)
        
        state = next_state
        episode_reward += reward
        
        # Update Agent (only if we have enough samples)
        if len(memory) > batch_size:
            agent.update(memory, batch_size)
            
        # Logging & Reset
        if done:
            episode_count += 1
            wandb.log({
                "train/episode_reward": episode_reward,
                "global_step": step
            })
            print(f"Step: {step}, Episode: {episode_count}, Reward: {episode_reward:.2f}")
            
            state, _ = env.reset()
            episode_reward = 0
            
    # 5. Save Final Model
    # We save the policy (Actor) for testing
    torch.save(agent.policy.state_dict(), "sac_policy.pth")
    # We can save critics too if we want to resume training later
    torch.save(agent.q1.state_dict(), "sac_q1.pth")
    torch.save(agent.q2.state_dict(), "sac_q2.pth")
    
    # Upload to W&B
    artifact = wandb.Artifact('sac_model', type='model')
    artifact.add_file('sac_policy.pth')
    wandb.log_artifact(artifact)

    env.close()
    wandb.finish()
    print("SAC Training Complete.")

if __name__ == "__main__":
    train()