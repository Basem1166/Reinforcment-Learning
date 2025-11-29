# ==========================================
# HYPERPARAMETER CONFIGURATION
# ==========================================

# OPTIONS: "A2C", "PPO", "SAC"
MODEL_TYPE = "PPO"

# 1. Environment Settings
ENV_CONFIG = {
    # Options: CartPole-v1, Acrobot-v1, MountainCar-v0
    "env_name": "CartPole-v1", 
    "total_timesteps": 100000, 
    "test_episodes": 100 
}

# 2. Weights & Biases Config
WANDB_CONFIG = {
    "project_name": "RL_Assignment3_Group12", 
    "entity": None, 
    "tags": ["PPO", "CartPole", "Baseline"],
    "run_name": "PPO_CartPole_Run_1"
}

# 3. Tuning Hyperparameters
HYPERPARAMETERS = {
    "gamma": 0.99,            # Discount factor
    "learning_rate": 3e-4,    # NN Learning Rate
    
    # PPO Specifics
    "ppo_clip": 0.2,          # Clipping range (epsilon)
    "entropy_coef": 0.01,     # Entropy coefficient to encourage exploration
    
    # For PPO, batch_size usually refers to the rollout buffer length (T)
    # We collect this many steps before performing an update.
    "batch_size": 2048,       
    
    # Unused by PPO (kept for compatibility or safe to remove)
    "buffer_size": 10000,     # Not used (PPO is on-policy)
    "tau": 0.005              # Not used (PPO doesn't use soft target updates)
}