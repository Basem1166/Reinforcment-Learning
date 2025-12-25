import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers import RecordVideo
import ale_py  # make sure ale-py is installed
from collections import deque

class FrameStack(gym.Wrapper):
    """Frame stack wrapper for Gymnasium."""
    def __init__(self, env, k):
        super().__init__(env)  # important!
        self.k = k
        self.frames = deque(maxlen=k)

        c, h, w = 1, 84, 84  # grayscale frames after AtariPreprocessing
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(k, h, w), dtype=np.uint8
        )
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)



def make_env(record=False, video_dir="videos"):
    # Remove gym.register_envs(ale_py) — not needed
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")

    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        scale_obs=True,
        frame_skip=1,
        screen_size=80
    )

    env = FrameStack(env, k=3)  # stack 3 frames

    if record:
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda e: True
        )

    return env
