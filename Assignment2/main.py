import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import wandb
import torch
import numpy as np
import argparse
import json
import os
import csv
from collections import namedtuple

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# A named tuple to store transitions (Must be defined at the top level for all files)
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))


class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    Wraps the Pendulum-v1 environment to discretize the continuous action space.
    """
    def __init__(self, env, num_bins):
        super().__init__(env)
        self.num_bins = num_bins
        # Create 'num_bins' evenly spaced points between the low and high action bounds
        self.action_bins = np.linspace(
            self.action_space.low[0],
            self.action_space.high[0],
            num=num_bins,
            dtype=np.float32
        )
        # Update the action space to be discrete
        self.action_space = gym.spaces.Discrete(num_bins)

    def action(self, action):
        """
        Takes the discrete action (e.g., 0, 1, 2...) and returns the
        corresponding continuous action (e.g., -2.0, -1.8, ...)
        """
        # 'action' is now an integer index. We return the continuous value from our bins.
        # We need to return it as a numpy array, as the Pendulum env expects.
        return np.array([self.action_bins[action]])

def main():
    # --- 1. Create a parser that just wants one --config file ---
    parser = argparse.ArgumentParser(description="DQN/DDQN Agent")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON config file")
    
    args = parser.parse_args()

    # --- 2. Load all hyperparameters from the JSON file ---
    with open(args.config) as f:
        config = json.load(f)
    
    # --- 3. Initialize W&B with this config ---
    wandb.init(
        project="cmps458-assignment2",
        name=config["run_name"],  # Use the name from the file
        config=config             # Log ALL parameters
    )

    # --- 4. Use the config dictionary to set up everything ---
    
    # Check for special environment arguments (like for Pendulum-v1)
    env_kwargs = config.get("env_kwargs", {})
    env = gym.make(config["env"], **env_kwargs)
    
    # --- !! CRITICAL CHECK !! ---
    # Check if we need to apply the discretization wrapper
    if isinstance(env.action_space, gym.spaces.Box):
        print(f"Detected continuous action space for {config['env']}.")
        if "discrete_actions" in config:
            print(f"Applying discretization wrapper with {config['discrete_actions']} bins.")
            env = DiscretizeActionWrapper(env, config["discrete_actions"])
        else:
            print(f"ERROR: Env {config['env']} has a continuous action space, but")
            print("'discrete_actions' key not found in config. DQN requires a discrete space.")
            env.close()
            wandb.finish()
            exit()
            
    # Now we can safely get the dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Dynamically import the agent to avoid circular dependencies if Transition is here
    from agent import DQNAgent
    agent = DQNAgent(state_dim, action_dim, 
                     config["memory_size"], 
                     config["batch_size"],
                     config["lr"], 
                     config["gamma"], 
                     config["epsilon_decay"], 
                     config["epsilon_min"], 
                     config["ddqn"],
                     config["layer1_size"],
                     config["layer2_size"])

    # --- Training Loop ---
    print("--- Starting Training Loop ---")
    steps_done = 0
    for episode in range(config["num_episodes"]):
        state, info = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32)
        
        episode_reward = 0
        episode_loss = 0
        num_steps = 0
        
        done = False
        while not done:
            action = agent.select_action(state)
            steps_done += 1
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32) # Use float32 for rewards
            done = terminated or truncated

            next_state = torch.tensor(observation, device=device, dtype=torch.float32)

            # Store the transition
            agent.memory.push(state, action, next_state, reward_tensor, done)

            # Move to the next state
            state = next_state

            # Perform one step of optimization
            loss = agent.optimize_model()
            if loss:
                episode_loss += loss

            episode_reward += reward # Use the raw Python reward for logging
            num_steps += 1
            
            if done:
                break
                
        # Update epsilon at the end of the episode
        agent.update_epsilon()
            
        # Log to W&B
        wandb.log({
            "episode": episode,
            "reward": episode_reward,
            "avg_loss": episode_loss / num_steps if num_steps > 0 else 0,
            "duration": num_steps,
            "epsilon": agent.epsilon
        })
        
        # Log to console
        if episode % 10 == 0 or episode == config["num_episodes"] - 1:
            print(f"Ep: {episode} | Reward: {episode_reward:.2f} | Duration: {num_steps} | "
                  f"Avg Loss: {episode_loss / num_steps:.4f} | Epsilon: {agent.epsilon:.3f}")

        # Update the target network
        if episode % config["target_update_freq"] == 0:
            print(f"--- Episode {episode}: Target network updated. ---")
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print("--- Training Complete. ---")

    # --- Evaluation Loop (Part 7) ---
    print(f"--- Running {config['num_test_episodes']} Test Episodes ---")
    
    # Use a fresh env for testing
    test_env_kwargs = config.get("env_kwargs", {})
    test_env = gym.make(config["env"], **test_env_kwargs)
    
    # Apply wrapper if needed
    if isinstance(test_env.action_space, gym.spaces.Box):
        if "discrete_actions" in config:
            test_env = DiscretizeActionWrapper(test_env, config["discrete_actions"])
    
    test_durations = []
    agent.epsilon = 0.0 # Turn off exploration completely
    
    for i in range(config['num_test_episodes']):
        state, info = test_env.reset()
        duration = 0
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                q_values = agent.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
                
            state, reward, terminated, truncated, _ = test_env.step(action)
            duration += 1
            done = terminated or truncated
        test_durations.append(duration)
        if (i+1) % 10 == 0:
            print(f"Test Episode {i+1}/{config['num_test_episodes']} finished. Duration: {duration}")

    test_env.close()
    
    # Log the test durations to W&B
    wandb.log({
        "test_durations_histogram": wandb.Histogram(test_durations),
        "test_avg_duration": np.mean(test_durations),
        "test_std_duration": np.std(test_durations)
    })
    print(f"Average test duration: {np.mean(test_durations):.2f} +/- {np.std(test_durations):.2f}")

    # --- Save test results to CSV ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, f"{config['run_name']}_test_results.csv")
    
    with open(results_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Duration'])
        for i, duration in enumerate(test_durations):
            writer.writerow([i+1, duration])
    print(f"Test results saved to {results_filename}")


    # --- Record Video (Part 9) ---
    print("--- Recording video... ---")
    
    # Use a fresh env for video
    video_env_kwargs = config.get("env_kwargs", {})
    video_env = gym.make(config['env'], render_mode="rgb_array", **video_env_kwargs)
    
    # Apply wrapper if needed
    if isinstance(video_env.action_space, gym.spaces.Box):
        if "discrete_actions" in config:
            video_env = DiscretizeActionWrapper(video_env, config["discrete_actions"])
            
    video_env = RecordVideo(video_env, "videos",
                            name_prefix=f"{config['run_name']}",
                            episode_trigger=lambda e: e == 0) # Record only one episode

    state, info = video_env.reset()
    done = False
    video_steps = 0
    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            action = agent.policy_net(state_tensor).max(1)[1].item()
            
        state, reward, terminated, truncated, _ = video_env.step(action)
        video_steps += 1
        done = terminated or truncated

    print(f"Video recorded. Length: {video_steps} steps.")
    video_env.close()
    
    wandb.finish()

# --- Run the main function ---
if __name__ == "__main__":
    # This makes sure that 'agent.py' and 'model.py' can find the 'Transition' tuple
    from agent import DQNAgent
    from model import QNetwork, ReplayMemory
    main()