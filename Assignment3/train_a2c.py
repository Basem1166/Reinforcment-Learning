import gymnasium as gym
import numpy as np
import torch
import wandb  
from a2c_agent import A2CAgent
from config import HYPERPARAMETERS, ENV_CONFIG, WANDB_CONFIG

def train():
    # 1. Initialize Weights & Biases
    wandb.init(
        project=WANDB_CONFIG["project_name"],
        config=HYPERPARAMETERS, # Pass params so W&B tracks them
        tags=WANDB_CONFIG["tags"],
        name=WANDB_CONFIG["run_name"],
        monitor_gym=True,       # Automatically log video of agent (if supported)
        save_code=True          # Saves your code to W&B
    )
    
    # 2. Setup Environment
    env_name = ENV_CONFIG["env_name"]
    env = gym.make(env_name) 
    
    # 3. Initialize Agent
    agent = A2CAgent(env, HYPERPARAMETERS)
    
    # Watch the model: gradients and topology
    wandb.watch(agent.actor, log="all", log_freq=100)
    wandb.watch(agent.critic, log="all", log_freq=100)

    # 4. Training Loop
    batch_size = HYPERPARAMETERS["batch_size"]
    total_timesteps = ENV_CONFIG["total_timesteps"]
    
    state, _ = env.reset()
    rollout_buffer = []
    episode_reward = 0
    episode_count = 0
    loss = 0
    
    print(f"--- Starting Training on {env_name} with W&B ---")
    
    for step in range(total_timesteps):
        
        # Select Action
        action = agent.get_action(state)
        
        # Step Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store in buffer
        rollout_buffer.append((state, action, reward, next_state, done))
        
        state = next_state
        episode_reward += reward
        
        # Update when buffer reaches batch_size OR episode ends
        if len(rollout_buffer) >= batch_size or done:
            loss = agent.update(rollout_buffer)
            rollout_buffer = [] 
            
            # Log Loss (Continuous logging)
            if loss is not None:
                wandb.log({"train/loss": loss, "global_step": step})
            
        if done:
            episode_count += 1
            # Log Episode Metrics to W&B
            wandb.log({
                "train/episode_reward": episode_reward,
                "train/episode_length": step, # Approximate relative step
                "global_step": step
            })
            
            print(f"Step: {step}, Episode: {episode_count}, Reward: {episode_reward}, Loss: {loss:.4f}")
            
            state, _ = env.reset()
            episode_reward = 0
            
    # 5. Save Final Model
    torch.save(agent.actor.state_dict(), "a2c_actor.pth")
    torch.save(agent.critic.state_dict(), "a2c_critic.pth")
    
    # Upload model files to W&B Artifacts
    artifact = wandb.Artifact('a2c_model', type='model')
    artifact.add_file('a2c_actor.pth')
    artifact.add_file('a2c_critic.pth')
    wandb.log_artifact(artifact)

    env.close()
    wandb.finish()
    print("Training Complete. Metrics logged to W&B.")

if __name__ == "__main__":
    train()