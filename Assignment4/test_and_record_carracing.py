"""
Test and record script for CNN-based PPO model from Kaggle notebook
Designed specifically for models trained with frame stacking (84x84 grayscale)
"""

import gymnasium as gym
import torch
import numpy as np
import argparse
import os
import csv
from datetime import datetime
import cv2
from collections import deque
import wandb
import config


# ============= Preprocessing Wrappers (from notebook) =============

class ActionRepeatWrapper(gym.Wrapper):
    """Repeat the chosen action for `repeat` frames and sum rewards."""
    
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    """Convert RGB to grayscale and resize to 84x84. Output: (1, H, W)."""
    
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, shape[0], shape[1]),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize to 84×84
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        # Add channel dimension (C, H, W)
        return resized[np.newaxis, :, :]


class FrameStackingWrapper(gym.ObservationWrapper):
    """Stack last N grayscale frames. Output: (N, H, W)."""
    
    def __init__(self, env, num_frames: int = 4):
        super().__init__(env)
        self.num_frames = num_frames

        c, h, w = env.observation_space.shape  # should be (1, 84, 84)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_frames, h, w),
            dtype=np.uint8
        )

        self.frames = deque(maxlen=num_frames)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self._get_stacked(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked(), reward, terminated, truncated, info
    
    def _get_stacked(self):
        return np.concatenate(list(self.frames), axis=0)


class CarRacingPreprocessor:
    """Apply preprocessing: ActionRepeat → Grayscale+Resize → FrameStack"""
    
    @staticmethod
    def apply(env: gym.Env, use_grayscale=True, num_frames=4, action_repeat=4) -> gym.Env:
        
        if action_repeat > 1:
            env = ActionRepeatWrapper(env, repeat=action_repeat)

        if use_grayscale:
            env = GrayscaleResizeWrapper(env, shape=(84, 84))

        if num_frames > 1:
            env = FrameStackingWrapper(env, num_frames=num_frames)

        return env


# ============= CNN Networks (from notebook) =============

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weights_init_(m):
    """Initialize network weights"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class CarRacingCNNPPOEncoder(nn.Module):
    """CNN encoder for CarRacing with grayscale 84x84 frames and frame stacking"""
    
    def __init__(self, feature_dim: int = 256, input_channels: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        
        conv_out_dim = 64 * 7 * 7
        self.fc = nn.Linear(conv_out_dim, feature_dim)
        
        self.apply(weights_init_)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        return x


class PPOActorCNN(nn.Module):
    """Actor network for PPO with CNN encoder"""
    
    def __init__(self, encoder: CarRacingCNNPPOEncoder, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        
        self.fc1 = nn.Linear(encoder.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.apply(weights_init_)
    
    def forward(self, state):
        features = self.encoder(state)
        x = torch.tanh(self.fc1(features))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std
    
    def get_dist(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)


class PPOCriticCNN(nn.Module):
    """Critic network for PPO with CNN encoder"""
    
    def __init__(self, encoder: CarRacingCNNPPOEncoder, hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        
        self.fc1 = nn.Linear(encoder.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        features = self.encoder(state)
        x = torch.tanh(self.fc1(features))
        x = torch.tanh(self.fc2(x))
        value = self.value(x)
        return value


# ============= Simplified PPO Agent for Testing =============

class NotebookPPOAgent:
    """PPO Agent specifically for testing notebook-trained models"""
    
    def __init__(self, action_dim, device, feature_dim=256, num_frames=4, hidden_dim=512):
        self.device = device
        self.action_dim = action_dim
        
        self.encoder = CarRacingCNNPPOEncoder(
            feature_dim=feature_dim,
            input_channels=num_frames
        ).to(device)
        
        self.actor = PPOActorCNN(self.encoder, action_dim, hidden_dim).to(device)
        self.critic = PPOCriticCNN(self.encoder, hidden_dim).to(device)
    
    def select_action(self, state, eval_mode=True):
        """Select action for testing"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state = state.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, _ = self.actor(state)
            action = mean
        
        return action.cpu().numpy()[0]
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"✓ Model loaded from {filepath}")


# ============= Testing Function =============

def test_notebook_model(model_path, num_episodes=100, record_video=True, num_frames=4, use_wandb=True):
    """Test CNN-based PPO model from notebook
    
    Args:
        model_path: Path to .pth checkpoint file
        num_episodes: Number of test episodes
        record_video: Whether to record videos
        num_frames: Number of stacked frames (should match training)
        use_wandb: Whether to log to Weights & Biases
    """
    
    # Get wandb config
    wandb_cfg = config.WANDB
    
    # Initialize wandb if enabled
    if use_wandb and wandb_cfg['enabled']:
        wandb.init(
            project=wandb_cfg['project'],
            entity=wandb_cfg['entity'],
            name=f"PPO-CarRacing-Notebook-Test",
            config={
                'algorithm': 'PPO',
                'environment': 'CarRacing-v3',
                'model_path': model_path,
                'num_episodes': num_episodes,
                'num_frames': num_frames,
            },
            tags=['notebook', 'test', 'ppo']
        )
    
    # Create environment with proper preprocessing
    print("[*] Creating environment...")
    env = gym.make('CarRacing-v3', continuous=True, render_mode='rgb_array' if record_video else None)
    env = CarRacingPreprocessor.apply(env, use_grayscale=True, num_frames=num_frames, action_repeat=4)
    
    if record_video:
        os.makedirs('videos', exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/PPO_CarRacing_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            episode_trigger=lambda x: x % 10 == 0
        )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    # Initialize agent
    print("[*] Initializing agent...")
    action_dim = env.action_space.shape[0]
    agent = NotebookPPOAgent(
        action_dim=action_dim,
        device=device,
        feature_dim=256,
        num_frames=num_frames,
        hidden_dim=512
    )
    
    # Load model
    print(f"[*] Loading model from {model_path}...")
    agent.load(model_path)
    
    # Test agent
    print(f"[*] Testing for {num_episodes} episodes...\n")
    
    episode_rewards = []
    episode_lengths = []
    timestep = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.uint8)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, eval_mode=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.uint8)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            timestep += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log episode to wandb (same format as training)
        if use_wandb and wandb_cfg['enabled']:
            wandb.log({
                'timestep': timestep,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
            })
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1:3d}/{num_episodes} | Avg Reward (last 10): {avg_reward:7.2f}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print("\n" + "="*60)
    print("TEST RESULTS - PPO CarRacing-v3 (Notebook Model)")
    print("="*60)
    print(f"Mean Reward:       {mean_reward:7.2f} ± {std_reward:6.2f}")
    print(f"Min Reward:        {min_reward:7.2f}")
    print(f"Max Reward:        {max_reward:7.2f}")
    print(f"Mean Episode Length: {mean_length:7.2f}")
    print("="*60)
    
    # Log final statistics to wandb
    if use_wandb and wandb_cfg['enabled']:
        wandb.log({
            'timestep': timestep,
            'eval_reward': mean_reward,
        })
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"results/PPO_CarRacing_notebook_test_{timestamp}.csv"
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Length'])
        for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
            writer.writerow([i+1, reward, length])
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Save summary
    summary_file = f"results/PPO_CarRacing_notebook_test_{timestamp}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("TEST RESULTS - PPO CarRacing-v3 (Notebook Model)\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of Episodes: {num_episodes}\n")
        f.write(f"Frame Stack Size: {num_frames}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nMean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Min Reward: {min_reward:.2f}\n")
        f.write(f"Max Reward: {max_reward:.2f}\n")
        f.write(f"Mean Episode Length: {mean_length:.2f}\n")
    
    print(f"✓ Summary saved to {summary_file}")
    
    # Finish wandb run
    if use_wandb and wandb_cfg['enabled']:
        wandb.finish()
    
    env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'mean_length': mean_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test CNN-based PPO model from Kaggle notebook',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_notebook_model.py ./models/PPO_CarRacing_best.pth
  python test_notebook_model.py ./models/PPO_CarRacing_best.pth --episodes 50
  python test_notebook_model.py ./models/PPO_CarRacing_best.pth --no-video
        """
    )
    
    parser.add_argument('model_path', type=str,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of test episodes (default: 100)')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--frames', type=int, default=4,
                        help='Number of stacked frames (default: 4)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        exit(1)

    test_notebook_model(
        model_path=args.model_path,
        num_episodes=args.episodes,
        record_video=not args.no_video,
        num_frames=args.frames,
        use_wandb=not args.no_wandb
    )
