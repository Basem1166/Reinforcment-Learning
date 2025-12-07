import gymnasium as gym
import numpy as np
import os
import torch
import wandb
from sac_agent import SACAgent
from buffer import ReplayBuffer, PrioritizedReplayBuffer
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
    # SAC now supports both Continuous and Discrete action spaces
    env_name = ENV_CONFIG["env_name"]
    env = gym.make(env_name)
    
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_type = "Discrete" if is_discrete else "Continuous"
    
    # 3. Initialize Agent & Buffer
    agent = SACAgent(env, HYPERPARAMETERS)
    
    # Use Prioritized Replay for sparse reward envs like MountainCar
    use_per = HYPERPARAMETERS.get("use_prioritized_replay", False)
    if use_per:
        memory = PrioritizedReplayBuffer(
            capacity=HYPERPARAMETERS["buffer_size"],
            alpha=HYPERPARAMETERS.get("per_alpha", 0.6),
            beta=HYPERPARAMETERS.get("per_beta", 0.4),
            beta_increment=HYPERPARAMETERS.get("per_beta_increment", 0.001),
            reward_priority_weight=HYPERPARAMETERS.get("reward_priority_weight", 0.5),
            success_bonus=HYPERPARAMETERS.get("success_bonus", 10.0)
        )
        print("Using Prioritized Experience Replay (PER)")
    else:
        memory = ReplayBuffer(HYPERPARAMETERS["buffer_size"])
        print("Using standard Replay Buffer")
    
    # 4. Training Loop
    batch_size = HYPERPARAMETERS["batch_size"]
    total_timesteps = ENV_CONFIG["total_timesteps"]
    
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    
    print(f"--- Starting SAC-{action_type} Training on {env_name} ---")
    
    try:
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
                log_dict = {
                    "train/episode_reward": episode_reward,
                    "train/alpha": agent.alpha,
                    "global_step": step
                }
                wandb.log(log_dict)
                print(f"Step: {step}, Episode: {episode_count}, Reward: {episode_reward:.2f}, Alpha: {agent.alpha:.4f}")
                
                state, _ = env.reset()
                episode_reward = 0
    except Exception as ex:
        print(f"Training interrupted due to error: {ex}")
    finally:
        # 5. Save Final Model
        # We save the policy (Actor) for testing
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)

        policy_path = os.path.join(models_dir, f"sac_policy_{ENV_CONFIG['env_name']}.pth")
        q1_path = os.path.join(models_dir, f"sac_q1_{ENV_CONFIG['env_name']}.pth")
        q2_path = os.path.join(models_dir, f"sac_q2_{ENV_CONFIG['env_name']}.pth")

        torch.save(agent.policy.state_dict(), policy_path)
        torch.save(agent.q1.state_dict(), q1_path)
        torch.save(agent.q2.state_dict(), q2_path)

        artifact = wandb.Artifact('sac_model', type='model')
        artifact.add_file(policy_path)
        wandb.log_artifact(artifact)

        env.close()
        wandb.finish()
        print(f"SAC Training Complete. Saved:\n  {policy_path}\n  {q1_path}\n  {q2_path}")
            
    # 5. Save Final Model
    # We save the policy (Actor) for testing
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    policy_path = os.path.join(models_dir, f"sac_policy_{ENV_CONFIG['env_name']}.pth")
    q1_path = os.path.join(models_dir, f"sac_q1_{ENV_CONFIG['env_name']}.pth")
    q2_path = os.path.join(models_dir, f"sac_q2_{ENV_CONFIG['env_name']}.pth")

    torch.save(agent.policy.state_dict(), policy_path)
    torch.save(agent.q1.state_dict(), q1_path)
    torch.save(agent.q2.state_dict(), q2_path)

    artifact = wandb.Artifact('sac_model', type='model')
    artifact.add_file(policy_path)
    wandb.log_artifact(artifact)

    env.close()
    wandb.finish()
    print(f"SAC Training Complete. Saved:\n  {policy_path}\n  {q1_path}\n  {q2_path}")

if __name__ == "__main__":
    train()