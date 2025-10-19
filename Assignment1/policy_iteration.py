from grid_maze import GridMazeEnv
import time
import numpy as np
from gymnasium.wrappers import RecordVideo

def build_model(env):
    """
    Extracts the transition (T) and reward (R) models from a fixed 
    environment instance.
    
    The state 's' is the agent's (x, y) position, mapped to an integer 0-24.
    """
    size = env.size
    num_states = size * size
    num_actions = env.action_space.n
    
    # T[s, a, s'] = Probability of transition from s to s' given action a
    T = np.zeros((num_states, num_actions, num_states))
    
    # R[s, a, s'] = Reward for transition s -> s' given action a
    # We'll simplify: R[s, a, s'] = R(s')
    # The reward is based on the *landing cell*, not the action.
    R = np.zeros((num_states, num_actions, num_states))

    # Get fixed maze elements
    goal_pos = env.goal_pos
    bad1_pos = env.bad1_pos
    bad2_pos = env.bad2_pos
    
    for y in range(size):
        for x in range(size):
            s = y * size + x  # Current state (0-24)
            current_pos = np.array([x, y])

            # Check if 's' is a terminal state (goal or bad cell)
            is_terminal = False
            if np.array_equal(current_pos, goal_pos):
                is_terminal = True
            elif np.array_equal(current_pos, bad1_pos) or \
                 np.array_equal(current_pos, bad2_pos):
                is_terminal = True

            if is_terminal:
                # Terminal states transition to themselves with 0 reward
                for a in range(num_actions):
                    T[s, a, s] = 1.0
                    R[s, a, s] = 0.0
                continue # Done with this state

            # If not a terminal state, calculate transitions for all actions
            for a in range(num_actions):
                # Get the 3 possible directions and their probabilities
                intended_dir = env._action_to_direction[a]
                perp1_dir = np.array([intended_dir[1], -intended_dir[0]])
                perp2_dir = np.array([-intended_dir[1], intended_dir[0]])
                
                outcomes = [
                    (intended_dir, 0.70),
                    (perp1_dir, 0.15),
                    (perp2_dir, 0.15)
                ]
                
                for direction, prob in outcomes:
                    # Calculate new position
                    new_pos = current_pos + direction
                    # Handle boundaries
                    new_pos = np.clip(new_pos, 0, size - 1)
                    
                    s_prime = new_pos[1] * size + new_pos[0] # Next state (0-24)
                    
                    # Calculate reward for this transition
                    reward = -0.1 # Default move cost
                    if np.array_equal(new_pos, goal_pos):
                        reward = 10.0
                    elif np.array_equal(new_pos, bad1_pos) or \
                         np.array_equal(new_pos, bad2_pos):
                        reward = -10.0
                        
                    # Add to our model
                    T[s, a, s_prime] += prob
                    R[s, a, s_prime] = reward

    return T, R, num_states, num_actions

def policy_iteration(T, R, num_states, num_actions, gamma=0.99, theta=1e-6):
    """
    Implements the Policy Iteration algorithm.
    """
    
    # 1. Initialize a random policy (policy[s] = action)
    policy = np.random.randint(0, num_actions, num_states)
    
    # Initialize value function
    V = np.zeros(num_states)
    
    is_policy_stable = False
    total_iterations = 0
    
    print("Running Policy Iteration...")
    while not is_policy_stable:
        total_iterations += 1
        
        # --- 1. Policy Evaluation ---
        while True:
            delta = 0
            for s in range(num_states):
                v_old = V[s]
                a = policy[s] # Get action from current policy
                
                # E[R + gamma * V(s')] under policy pi
                v_new = 0
                for s_prime in range(num_states):
                    prob = T[s, a, s_prime]
                    reward = R[s, a, s_prime]
                    v_new += prob * (reward + gamma * V[s_prime])
                
                V[s] = v_new
                delta = max(delta, abs(v_old - V[s]))

            if delta < theta:
                break # Value function converged for this policy
            
        # --- 2. Policy Improvement ---
        is_policy_stable = True
        
        for s in range(num_states):
            old_action = policy[s]
            
            # Calculate Q-value for all actions in state 's'
            action_values = np.zeros(num_actions)
            for a in range(num_actions):
                q_sa = 0
                for s_prime in range(num_states):
                    prob = T[s, a, s_prime]
                    reward = R[s, a, s_prime]
                    q_sa += prob * (reward + gamma * V[s_prime])
                action_values[a] = q_sa
            
            # Find the best action (greedily)
            best_action = np.argmax(action_values)
            
            # Update the policy
            policy[s] = best_action
            
            # Check if policy changed
            if old_action != best_action:
                is_policy_stable = False
                
    print(f"Policy Iteration converged in {total_iterations} iterations.")
    return policy, V, total_iterations

def run_policy(env, policy, maze_seed):
    """
    Runs the trained policy in the environment with rendering.
    """
    print("\nRunning trained policy...")
    try:
        # Re-create the environment with 'human' render mode
        # NOTE: This creates a NEW random maze.
        # For a true test, you'd need to modify GridMazeEnv
        # to accept and set fixed positions in reset().
        
        # Let's just create a new env and run the policy.
        # This tests if the policy *logic* is good, but it's
        # running on a *different* maze than it was trained on.
        # This is a key point for your report.
        
        test_env = GridMazeEnv(size=env.size, render_mode="human")
        (obs, info) = test_env.reset(seed=maze_seed) # Fixed seed for reproducibility
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Map the full observation (8-vec) to our simple state (0-24)
            agent_x, agent_y = obs[0], obs[1]
            s = agent_y * test_env.size + agent_x
            
            # Get action from our policy
            action = policy[s]
            
            (obs, reward, terminated, truncated, info) = test_env.step(action)
            
            # Render the environment
            test_env.render()
            
            time.sleep(0.2) # Slow down for human viewing

        print("Episode finished.")
        test_env.close()

    except Exception as e:
        print(f"Could not run policy test (display error?): {e}")
        print("This often fails if you are not on a desktop.")
        print("The policy and V-function are still computed.")
        
        
def record_policy_video(policy, seed,size):
    """
    Runs the trained policy and saves a video recording.
    Uses the same seed as training to ensure the maze is identical.
    """
    print("\nRecording policy to video...")

    # 1. Create a new environment
    # We must set render_mode="rgb_array" for the wrapper
    video_env = GridMazeEnv(size=size, render_mode="rgb_array")

    # 2. Wrap the environment with RecordVideo 
    video_env = RecordVideo(
        video_env, 
        video_folder="videos", # Creates a 'videos' folder
        name_prefix="policy-iteration-agent"
    )

    # 3. Reset with the *same seed* to get the same maze
    (obs, info) = video_env.reset(seed=seed)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # The wrapper handles rendering automatically

        # Map observation to state
        agent_x, agent_y = obs[0], obs[1]
        s = agent_y * video_env.env.size + agent_x

        # Get action from our policy
        action = policy[s]

        (obs, reward, terminated, truncated, info) = video_env.step(action)

    print("Video saved. Closing environment.")
    video_env.close() # This is crucial to finalize the video file

# --- Main execution ---
if __name__ == "__main__":

    maze_seed = 1
    size = 5
    
    
    # 1. Create the environment
    env = GridMazeEnv(size=size)
    
    # 2. Reset it ONCE to get a fixed maze
    (obs, info) = env.reset(seed=maze_seed)

    print("--- Fixed Maze Configuration ---")
    print(f"Agent Start: {env.agent_pos}")
    print(f"Goal: {env.goal_pos}")
    print(f"Bad Cell 1: {env.bad1_pos}")
    print(f"Bad Cell 2: {env.bad2_pos}")
    print("---------------------------------")

    # 3. Build the T and R models from this fixed maze
    T, R, num_states, num_actions = build_model(env)
    
    # 4. Run Policy Iteration
    policy, V, iterations = policy_iteration(T, R, num_states, num_actions)

    # 5. Print the results
    print("\n--- Results ---")
    print(f"Iterations to converge: {iterations}")
    
    # Reshape V and Policy for easy viewing
    # 0: right, 1: up, 2: left, 3: down
    action_map = {0: ">", 1: "^", 2: "<", 3: "v"}
    
    policy_grid = np.empty((env.size, env.size), dtype=str)
    value_grid = np.zeros((env.size, env.size))

    for y in range(env.size):
        for x in range(env.size):
            s = y * env.size + x
            policy_grid[y, x] = action_map[policy[s]]
            value_grid[y, x] = V[s]

            if np.array_equal([x, y], env.goal_pos):
                policy_grid[y, x] = "G"
            elif np.array_equal([x, y], env.bad1_pos) or \
                 np.array_equal([x, y], env.bad2_pos):
                policy_grid[y, x] = "X"

    print("\nOptimal Value Function (V*):")
    print(np.round(value_grid, 2))
    
    print("\nOptimal Policy (pi*):")
    print(policy_grid)
    
    # 6. Run the policy (optional, requires desktop)
    #run_policy(env, policy, maze_seed)

    record_policy_video(policy, seed=maze_seed, size=size)