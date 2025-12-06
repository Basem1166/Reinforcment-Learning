# Assignment 4: Deep Reinforcement Learning on Box2D Environments

This assignment implements three state-of-the-art continuous control algorithms (SAC, PPO, TD3) and tests them on Box2D environments: LunarLander-v3 and CarRacing-v3.

## 📋 Algorithms Implemented

1. **SAC (Soft Actor-Critic)**: Off-policy algorithm with automatic entropy tuning
2. **PPO (Proximal Policy Optimization)**: On-policy algorithm with clipped surrogate objective
3. **TD3 (Twin Delayed DDPG)**: Off-policy algorithm with twin Q-networks and delayed policy updates

## 🗂️ File Structure

```
Assignment4/
├── models.py               # Neural network architectures for all algorithms
├── buffer.py               # Replay buffer (SAC/TD3) and Rollout buffer (PPO)
├── sac_agent.py           # SAC agent implementation
├── ppo_agent.py           # PPO agent implementation
├── td3_agent.py           # TD3 agent implementation
├── config.py              # Hyperparameter configuration (EDIT THIS!)
├── train_sac.py           # SAC training script
├── train_ppo.py           # PPO training script
├── train_td3.py           # TD3 training script
├── test_and_record.py     # Testing and video recording script
├── README.md              # This file
├── models/                # Saved model checkpoints (created during training)
├── results/               # Test results and statistics (created during testing)
└── videos/                # Recorded videos (created during testing)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install gymnasium[box2d] torch numpy wandb matplotlib
```

### 2. Configure Hyperparameters

Edit `config.py` to adjust hyperparameters for different algorithms and environments. The config file is well-documented with all available options.

Key sections:
- `TRAINING`: Training settings (timesteps, batch size, etc.)
- `WANDB`: Weights & Biases configuration
- `SAC_HYPERPARAMETERS`: SAC-specific settings
- `PPO_HYPERPARAMETERS`: PPO-specific settings
- `TD3_HYPERPARAMETERS`: TD3-specific settings
- Environment-specific overrides (e.g., `LUNARLANDER_SAC`, `CARRACING_PPO`)

### 3. Train Models

#### Train SAC on LunarLander
```bash
python train_sac.py --env lunarlander
```

#### Train PPO on LunarLander
```bash
python train_ppo.py --env lunarlander
```

#### Train TD3 on LunarLander
```bash
python train_td3.py --env lunarlander
```

#### Train on CarRacing
```bash
python train_sac.py --env carracing
python train_ppo.py --env carracing
python train_td3.py --env carracing
```

#### Disable Wandb Logging
```bash
python train_sac.py --env lunarlander --no-wandb
```

### 4. Test Trained Models

```bash
# Test SAC model
python test_and_record.py --algorithm sac --env lunarlander --model models/SAC_LunarLander-v3_best.pth

# Test PPO model
python test_and_record.py --algorithm ppo --env lunarlander --model models/PPO_LunarLander-v3_best.pth

# Test TD3 model
python test_and_record.py --algorithm td3 --env lunarlander --model models/TD3_LunarLander-v3_best.pth

# Test with custom number of episodes
python test_and_record.py --algorithm sac --env lunarlander --model models/SAC_LunarLander-v3_best.pth --episodes 50

# Test without video recording
python test_and_record.py --algorithm sac --env lunarlander --model models/SAC_LunarLander-v3_best.pth --no-video
```

## 🎯 Environments

### LunarLander-v3
- **Observation Space**: 8-dimensional continuous vector
- **Action Space**: 2-dimensional continuous (main engine, side engines)
- **Goal**: Land the lunar lander safely on the landing pad
- **Reward**: +100 for landing, -100 for crashing, penalties for fuel usage

### CarRacing-v3
- **Observation Space**: 96x96x3 RGB image (will need preprocessing for pixel-based learning)
- **Action Space**: 3-dimensional continuous (steering, gas, brake)
- **Goal**: Complete the race track as fast as possible
- **Reward**: -0.1 per frame, +1000/N for each track tile visited

## ⚙️ Hyperparameter Tuning Guide

### General Tips
1. **Learning Rate**: Start with 3e-4, decrease if unstable
2. **Batch Size**: Larger batches (256-512) for more stable updates
3. **Hidden Dimensions**: 256 for simple tasks, 512+ for complex tasks
4. **Gamma**: 0.99 is standard, increase to 0.995 for longer horizons

### SAC-Specific
- **Alpha**: Temperature parameter. Auto-tuning recommended (`auto_entropy: True`)
- **Tau**: Soft update rate. 0.005 is standard
- **Buffer Size**: 1M is typical, can reduce for memory constraints

### PPO-Specific
- **Clip Epsilon**: 0.2 is standard. Smaller (0.1) for more conservative updates
- **Entropy Coefficient**: 0.01 typical. Increase for more exploration
- **GAE Lambda**: 0.95 standard. Higher values reduce bias but increase variance
- **Rollout Length**: 2048 typical. Adjust based on episode length

### TD3-Specific
- **Policy Noise**: 0.2 typical. Controls target policy smoothing
- **Exploration Noise**: 0.1 typical. Controls action exploration
- **Policy Frequency**: 2 means update policy every 2 critic updates

## 📊 Monitoring Training

### Weights & Biases
The training scripts automatically log to wandb. View metrics at: https://wandb.ai

Key metrics logged:
- Episode reward
- Episode length
- Evaluation reward
- Loss values (Q-loss, policy loss, value loss)
- Algorithm-specific metrics (alpha, entropy, etc.)

### Local Monitoring
Training progress is printed to console:
```
Timestep: 50000, Episode: 123, Reward: 245.67, Length: 456
Evaluation at timestep 50000: Avg Reward: 267.89
New best model saved with reward: 267.89
```

## 💾 Model Checkpoints

Models are saved in the `models/` directory:
- `*_best.pth`: Best model based on evaluation reward
- `*_final.pth`: Model at end of training
- `*_checkpoint_*.pth`: Periodic checkpoints (every 50k timesteps)

## 📈 Results Analysis

After testing, results are saved in `results/`:
- `*_test_results.csv`: Episode-by-episode results
- `*_test_summary.txt`: Overall statistics

Example summary:
```
Mean Reward: 267.89 ± 45.23
Min Reward: 150.34
Max Reward: 298.76
Mean Episode Length: 456.78
```

## 🎥 Video Recording

Videos are recorded every 10th test episode by default. Find them in `videos/`.

To change recording frequency, edit `test_and_record.py`:
```python
episode_trigger=lambda x: x % 10 == 0  # Record every 10th episode
```

## 🔧 Common Issues

### 1. CUDA Out of Memory
- Reduce `batch_size` in `config.py`
- Reduce `hidden_dim` in `config.py`
- Reduce `buffer_size` for SAC/TD3

### 2. Training Unstable
- Decrease learning rate
- Increase batch size
- For PPO: decrease `clip_epsilon`
- For SAC: enable `auto_entropy`
- For TD3: increase `policy_freq`

### 3. Slow Training
- Use GPU if available
- Reduce `eval_frequency`
- Reduce `save_frequency`
- For PPO: reduce `ppo_epochs`

### 4. Poor Performance
- Train longer (`total_timesteps`)
- Tune hyperparameters (start with learning rate)
- Check environment-specific configs
- Ensure proper exploration (check noise parameters for SAC/TD3)

## 📚 Algorithm Details

### SAC
- **Type**: Off-policy, model-free
- **Key Features**: Entropy regularization, automatic temperature tuning
- **Best For**: Continuous control, sample efficiency
- **Paper**: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)

### PPO
- **Type**: On-policy, model-free
- **Key Features**: Clipped surrogate objective, multiple epochs per rollout
- **Best For**: Stability, ease of tuning
- **Paper**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

### TD3
- **Type**: Off-policy, model-free
- **Key Features**: Twin critics, delayed policy updates, target policy smoothing
- **Best For**: Continuous control, deterministic policies
- **Paper**: [Twin Delayed DDPG](https://arxiv.org/abs/1802.09477)

## 🎓 Assignment Tips

1. **Start Simple**: Begin with LunarLander before CarRacing
2. **Baseline First**: Get one algorithm working well before trying others
3. **Hyperparameter Search**: Use wandb sweeps for systematic tuning
4. **Compare Fairly**: Use same evaluation protocol for all algorithms
5. **Document Results**: Keep track of what worked and what didn't

## 📝 Customization

### Adding New Environments
1. Add environment config to `ENVIRONMENTS` in `config.py`
2. Add algorithm-specific hyperparameters
3. Update training scripts with new environment option

### Modifying Network Architecture
Edit `models.py` to change:
- Number of hidden layers
- Layer sizes
- Activation functions
- Network initialization

### Custom Reward Shaping
Modify the reward in training scripts:
```python
# Example: Add extra reward for staying alive
modified_reward = reward + 0.1  # Small bonus per timestep
```

## 🤝 Getting Help

If you encounter issues:
1. Check hyperparameters in `config.py`
2. Review console output for error messages
3. Check wandb logs for training metrics
4. Verify environment installation: `python -c "import gymnasium; print(gymnasium.__version__)"`

Good luck with your assignment! 🚀
