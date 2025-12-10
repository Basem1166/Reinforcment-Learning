import gymnasium as gym
import torch
import numpy as np
import time
import wandb
import os
import matplotlib.pyplot as plt
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
    actor_path = f"models/{MODEL_TYPE.lower()}_{ENV_CONFIG['env_name']}_actor.pth"
    critic_path = f"models/{MODEL_TYPE.lower()}_{ENV_CONFIG['env_name']}_critic.pth"
    
    try:
        # SAC usually has 'policy' instead of 'actor', but we can standardize naming in the agent code
        # or handle the exception here.
        if MODEL_TYPE == "SAC":
                state_dict = torch.load(f"models/sac_policy_{ENV_CONFIG['env_name']}.pth", map_location='cpu')
                result = agent.policy.load_state_dict(state_dict, strict=False)
                print(f"Loaded SAC policy from: {os.path.abspath(f'models/sac_policy_{ENV_CONFIG['env_name']}.pth')}")
                if getattr(result, 'missing_keys', None) or getattr(result, 'unexpected_keys', None):
                    print(f"Note: missing_keys={getattr(result, 'missing_keys', [])}, unexpected_keys={getattr(result, 'unexpected_keys', [])}")
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
                # For continuous SAC: guard against NaNs/Infs and clip to bounds
                if hasattr(env.action_space, 'low'):
                    action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                    low = np.asarray(env.action_space.low, dtype=np.float32)
                    high = np.asarray(env.action_space.high, dtype=np.float32)
                    action = np.clip(action, low, high)
                # For discrete SAC: action is already an int, no clipping needed
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

        wandb.log({
        "test/episode_reward": total_reward,
        "test/episode_duration": duration,
        "test/episode_index": i,
    })
        
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
    wandb.log({
        "test/episode_reward": total_reward,
        "test/episode_duration": duration,
        "test/episode_index": i,
    }) 
    
    # 7. Upload Video
    video_path = f"./videos/{MODEL_TYPE}/{env_name}/test-agent-episode-0.mp4"
    try:
        wandb.log({"agent_video": wandb.Video(video_path, fps=30, format="mp4")})
        print("Video uploaded.")
    except Exception:
        print("Video upload failed (file might not exist yet).")

    # 8. Generate Stability Plot
    plot_test_stability(episode_rewards, env_name)

    wandb.finish()


def plot_test_stability(rewards, env_name):
    """
    Generate and save a plot showing test stability across episodes.
    Includes: per-episode rewards, rolling average, std deviation bands.
    """
    episodes = np.arange(1, len(rewards) + 1)
    rewards = np.array(rewards)
    
    # Calculate rolling statistics (window=10)
    window = 10
    rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
    rolling_std = np.array([rewards[max(0,i-window):i].std() for i in range(window, len(rewards)+1)])
    rolling_episodes = episodes[window-1:]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # --- Plot 1: Episode Rewards with Rolling Average ---
    ax1 = axes[0]
    ax1.bar(episodes, rewards, alpha=0.4, color='steelblue', label='Episode Reward')
    ax1.plot(rolling_episodes, rolling_mean, color='red', linewidth=2, label=f'Rolling Avg (window={window})')
    ax1.fill_between(rolling_episodes, 
                     rolling_mean - rolling_std, 
                     rolling_mean + rolling_std, 
                     alpha=0.2, color='red', label='±1 Std Dev')
    ax1.axhline(y=np.mean(rewards), color='green', linestyle='--', linewidth=1.5, 
                label=f'Mean: {np.mean(rewards):.2f}')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title(f'{MODEL_TYPE} Test Stability on {env_name}\n({len(rewards)} Episodes)', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Reward Distribution Histogram ---
    ax2 = axes[1]
    ax2.hist(rewards, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(rewards):.2f}')
    ax2.axvline(x=np.median(rewards), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(rewards):.2f}')
    ax2.set_xlabel('Reward', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Reward Distribution', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Add summary stats as text box
    stats_text = (f"Mean: {np.mean(rewards):.2f}\n"
                  f"Std: {np.std(rewards):.2f}\n"
                  f"Min: {np.min(rewards):.2f}\n"
                  f"Max: {np.max(rewards):.2f}\n"
                  f"Success Rate: {(rewards > np.mean(rewards)).sum()}/{len(rewards)}")
    ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{MODEL_TYPE}_{env_name}_test_stability.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Stability plot saved to: {os.path.abspath(plot_path)}")
    
    # Also log to W&B
    try:
        wandb.log({"test/stability_plot": wandb.Image(plot_path)})
    except:
        pass
    
    plt.show()


if __name__ == "__main__":
    evaluate()