import gymnasium as gym
import torch
import numpy as np
import time
import wandb
from gym.wrappers import RecordVideo
from config import HYPERPARAMETERS, ENV_CONFIG, WANDB_CONFIG, MODEL_TYPE

# Dynamic Imports based on configuration
if MODEL_TYPE == "A2C":
    from a2c_agent import A2CAgent as AgentClass
elif MODEL_TYPE == "PPO":
    # We will implement this next
    from ppo_agent import PPOAgent as AgentClass
elif MODEL_TYPE == "SAC":
    # We will implement the wrapper for this next
    from sac_agent import SACAgent as AgentClass
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

def evaluate():
    run_name = f"Test_{MODEL_TYPE}_{ENV_CONFIG['env_name']}"
    
    # 1. Initialize W&B
    wandb.init(
        project=WANDB_CONFIG["project_name"],
        name=run_name,
        job_type="evaluation",
        config=HYPERPARAMETERS,
        tags=["Evaluation", MODEL_TYPE, "Video"]
    )

    env_name = ENV_CONFIG["env_name"]
    print(f"--- Setting up Generic Test for {MODEL_TYPE} on {env_name} ---")

    # 2. Setup Environment with Video
    env = gym.make(env_name, render_mode='rgb_array')
    env = RecordVideo(
        env, 
        video_folder=f"./videos/{MODEL_TYPE}/{env_name}", 
        name_prefix="test-agent",
        episode_trigger=lambda x: x == 0 
    )

    # 3. Initialize Agent
    agent = AgentClass(env, HYPERPARAMETERS)
    
    # 4. Load Weights Dynamically
    # We assume you save files as "{model_type}_actor.pth" and "{model_type}_critic.pth"
    actor_path = f"{MODEL_TYPE.lower()}_actor.pth"
    critic_path = f"{MODEL_TYPE.lower()}_critic.pth"
    
    try:
        # SAC usually has 'policy' instead of 'actor', but we can standardize naming in the agent code
        # or handle the exception here.
        if MODEL_TYPE == "SAC":
             agent.policy.load_state_dict(torch.load(f"sac_policy.pth"))
             # SAC doesn't strictly need critics for testing, just the policy
        else:
            agent.actor.load_state_dict(torch.load(actor_path))
            agent.critic.load_state_dict(torch.load(critic_path))
            
        print(f"--- Loaded {MODEL_TYPE} Weights Successfully ---")
    except FileNotFoundError as e:
        print(f"Error: Could not find weight files: {e}")
        print(f"Make sure you ran the training script for {MODEL_TYPE} first.")
        return

    # 5. Run 100 Test Episodes
    test_episodes = ENV_CONFIG["test_episodes"]
    episode_durations = []
    episode_rewards = []

    for i in range(test_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        start_time = time.time()
        
        while not done:
            # Generic get_action interface
            if MODEL_TYPE == "SAC":
                # SAC returns just the action
                action = agent.get_action(state, evaluate=True)
            elif MODEL_TYPE == "PPO":
                # PPO returns (action, log_prob), we only need action for testing
                action, _ = agent.get_action(state)
            else:
                # A2C returns just the action
                action = agent.get_action(state)
            
            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            
        duration = time.time() - start_time
        episode_durations.append(duration)
        episode_rewards.append(total_reward)
        
        if (i+1) % 10 == 0:
            print(f"Episode {i+1}: Reward={total_reward:.2f}, Duration={duration:.2f}s")

    env.close()

    # 6. Log Stats
    avg_reward = np.mean(episode_rewards)
    avg_duration = np.mean(episode_durations)
    std_duration = np.std(episode_durations)

    wandb.log({
        "test/mean_reward": avg_reward,
        "test/mean_duration": avg_duration,
        "test/std_duration": std_duration,
        "test/model_type": MODEL_TYPE
    })
    
    # 7. Upload Video
    video_path = f"./videos/{MODEL_TYPE}/{env_name}/test-agent-episode-0.mp4"
    try:
        wandb.log({"agent_video": wandb.Video(video_path, fps=30, format="mp4")})
        print("Video uploaded.")
    except Exception:
        print("Video upload failed (file might not exist yet).")

    wandb.finish()

if __name__ == "__main__":
    evaluate()