# ==========================================
# HYPERPARAMETER CONFIGURATION
# ==========================================

# OPTIONS: "A2C", "PPO", "SAC"
from numpy import tri


MODEL_TYPE = "SAC"

# 1. Environment Settings
ENV_CONFIG = {
    # Discrete examples: CartPole-v1, Acrobot-v1, MountainCar-v0
    # Continuous examples: Pendulum-v1, MountainCarContinuous-v0
    "env_name": "Pendulum-v1", 
    "total_timesteps": 1000000, 
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
    "auto_entropy": True,     # Enable automatic entropy tuning
    
    # PPO Specifics
    "ppo_clip": 0.2,          # Clipping range (epsilon)
    "entropy_coef": 0.2,      # Entropy coefficient to encourage exploration
    
    # Buffer settings
    "batch_size": 256,       
    "buffer_size": 100000,
    "tau": 0.005,             # Soft update coefficient
    
    # ==========================================
    # Prioritized Experience Replay (PER) Settings
    # Especially useful for sparse reward envs like MountainCar
    # ==========================================
    "use_prioritized_replay": False,   # Enable PER
    "per_alpha": 0.6,                 # Priority exponent (0=uniform, 1=full priority)
    "per_beta": 0.4,                  # Importance sampling start (anneals to 1)
    "per_beta_increment": 0.001,      # How fast beta anneals
    "reward_priority_weight": 0.5,    # Weight for reward-based priority (vs TD error)
    "success_bonus": 10.0,            # Extra priority for successful transitions
}