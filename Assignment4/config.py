"""
Configuration file for Assignment 4: SAC, PPO, TD3 on Box2D environments
Edit hyperparameters here to tune your models
"""

# ============= Environment Settings =============
ENVIRONMENTS = {
    'lunarlander': {
        'name': 'LunarLander-v3',
        'continuous': True,
        'max_episode_steps': 1000
    },
    'carracing': {
        'name': 'CarRacing-v3',
        'continuous': True,
        'max_episode_steps': 1000
    }
}

# ============= Training Settings =============
TRAINING = {
    'total_timesteps': 100_000,      # Total training timesteps
    'eval_frequency': 10_000,          # Evaluate every N timesteps
    'eval_episodes': 10,               # Number of evaluation episodes
    'save_frequency': 50_000,          # Save model every N timesteps
    'start_timesteps': 10_000,         # Random actions for exploration at start (SAC/TD3)
    'batch_size': 256,                 # Batch size for training
    'buffer_size': 50_000,          # Replay buffer size (SAC/TD3)
    'rollout_length': 2048,            # Rollout length for PPO
}

# ============= Wandb Settings =============
WANDB = {
    'project': 'RL-Assignment4-Box2D',
    'entity': None,  # Set to your wandb username or leave None
    'enabled': True,  # Set to False to disable wandb logging
}

# ============= SAC Hyperparameters =============
SAC_HYPERPARAMETERS = {
    # Learning rates
    'lr': 3e-4,                        # Learning rate for all networks
    
    # Network architecture
    'hidden_dim': 256,                 # Hidden layer dimension
    
    # SAC specific
    'gamma': 0.99,                     # Discount factor
    'tau': 0.005,                      # Soft update coefficient
    'alpha': 0.2,                      # Temperature parameter (initial value)
    'auto_entropy': True,              # Automatic entropy tuning
    
    # Training settings (from TRAINING)
    'batch_size': TRAINING['batch_size'],
    'buffer_size': TRAINING['buffer_size'],
}

# ============= PPO Hyperparameters =============
PPO_HYPERPARAMETERS = {
    # Learning rates
    'lr': 3e-4,                        # Learning rate
    
    # Network architecture
    'hidden_dim': 256,                 # Hidden layer dimension
    
    # PPO specific
    'gamma': 0.99,                     # Discount factor
    'gae_lambda': 0.95,                # GAE lambda
    'clip_epsilon': 0.2,               # PPO clipping parameter
    'value_loss_coef': 0.5,            # Value loss coefficient
    'entropy_coef': 0.01,              # Entropy coefficient
    'max_grad_norm': 0.5,              # Gradient clipping
    
    # Training settings
    'ppo_epochs': 10,                  # Number of PPO update epochs
    'mini_batch_size': 64,             # Mini-batch size for PPO updates
    'rollout_length': TRAINING['rollout_length'],
}

# ============= TD3 Hyperparameters =============
TD3_HYPERPARAMETERS = {
    # Learning rates
    'lr': 3e-4,                        # Learning rate for all networks
    
    # Network architecture
    'hidden_dim': 256,                 # Hidden layer dimension
    
    # TD3 specific
    'gamma': 0.99,                     # Discount factor
    'tau': 0.005,                      # Soft update coefficient
    'policy_noise': 0.2,               # Noise added to target policy
    'noise_clip': 0.5,                 # Range to clip target policy noise
    'policy_freq': 2,                  # Frequency of delayed policy updates
    'expl_noise': 0.1,                 # Exploration noise std
    
    # Training settings (from TRAINING)
    'batch_size': TRAINING['batch_size'],
    'buffer_size': TRAINING['buffer_size'],
}

# ============= Environment-Specific Tuning =============
# You can override hyperparameters for specific environments here

# LunarLander-v3 specific settings
LUNARLANDER_SAC = {
    **SAC_HYPERPARAMETERS,
    'alpha': 0.2,
    'lr': 3e-4,
}

LUNARLANDER_PPO = {
    **PPO_HYPERPARAMETERS,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
}

LUNARLANDER_TD3 = {
    **TD3_HYPERPARAMETERS,
    'expl_noise': 0.1,
    'policy_noise': 0.2,
}

# CarRacing-v3 specific settings
CARRACING_SAC = {
    **SAC_HYPERPARAMETERS,
    'alpha': 0.1,
    'lr': 1e-4,
    'hidden_dim': 512,  # Larger network for image-based task
}

CARRACING_PPO = {
    **PPO_HYPERPARAMETERS,
    'lr': 1e-4,
    'entropy_coef': 0.001,
    'hidden_dim': 512,
}

CARRACING_TD3 = {
    **TD3_HYPERPARAMETERS,
    'lr': 1e-4,
    'expl_noise': 0.2,
    'hidden_dim': 512,
}

# ============= Helper Functions =============

def get_hyperparameters(algorithm, environment):
    """
    Get hyperparameters for a specific algorithm and environment
    
    Args:
        algorithm: 'sac', 'ppo', or 'td3'
        environment: 'lunarlander' or 'carracing'
    
    Returns:
        Dictionary of hyperparameters
    """
    config_map = {
        ('sac', 'lunarlander'): LUNARLANDER_SAC,
        ('sac', 'carracing'): CARRACING_SAC,
        ('ppo', 'lunarlander'): LUNARLANDER_PPO,
        ('ppo', 'carracing'): CARRACING_PPO,
        ('td3', 'lunarlander'): LUNARLANDER_TD3,
        ('td3', 'carracing'): CARRACING_TD3,
    }
    
    key = (algorithm.lower(), environment.lower())
    if key not in config_map:
        raise ValueError(f"Unknown algorithm-environment combination: {algorithm}-{environment}")
    
    return config_map[key]


def get_env_config(environment):
    """Get environment configuration"""
    env_key = environment.lower()
    if env_key not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {environment}")
    
    return ENVIRONMENTS[env_key]


# ============= Quick Access Configs =============
# For easy importing in training scripts

SAC_LUNARLANDER = {
    'env': ENVIRONMENTS['lunarlander'],
    'hyperparameters': LUNARLANDER_SAC,
    'training': TRAINING,
    'wandb': WANDB,
}

SAC_CARRACING = {
    'env': ENVIRONMENTS['carracing'],
    'hyperparameters': CARRACING_SAC,
    'training': TRAINING,
    'wandb': WANDB,
}

PPO_LUNARLANDER = {
    'env': ENVIRONMENTS['lunarlander'],
    'hyperparameters': LUNARLANDER_PPO,
    'training': TRAINING,
    'wandb': WANDB,
}

PPO_CARRACING = {
    'env': ENVIRONMENTS['carracing'],
    'hyperparameters': CARRACING_PPO,
    'training': TRAINING,
    'wandb': WANDB,
}

TD3_LUNARLANDER = {
    'env': ENVIRONMENTS['lunarlander'],
    'hyperparameters': LUNARLANDER_TD3,
    'training': TRAINING,
    'wandb': WANDB,
}

TD3_CARRACING = {
    'env': ENVIRONMENTS['carracing'],
    'hyperparameters': CARRACING_TD3,
    'training': TRAINING,
    'wandb': WANDB,
}
