import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame 

class GridMazeEnv(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=5 , render_mode=None):
        super().__init__()
        
        self.size = size  
        self.window_size = 512  # The size of the PyGame window
        self.render_mode = render_mode

        # Observation Space: 8 integers
        # (agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y)
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(8,), dtype=int)

        # Action Space: 4 discrete actions
        # 0: right, 1: up, 2: left, 3: down
        self.action_space = spaces.Discrete(4)
        
        # Action-to-direction mapping
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, -1]), # up
            2: np.array([-1, 0]), # left
            3: np.array([0, 1]),  # down
        }
        
        # Pygame setup
        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.concatenate([
            self.agent_pos, 
            self.goal_pos, 
            self.bad1_pos, 
            self.bad2_pos
        ]).astype(int)

    def _get_info(self):
        return {"distance": np.linalg.norm(self.agent_pos - self.goal_pos)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly place agent, goal, and 2 bad cells
        locations = self.np_random.choice(
            self.size * self.size, 
            size=4, 
            replace=False
        )
        
        self.agent_pos = np.array([locations[0] % self.size, locations[0] // self.size])
        self.goal_pos = np.array([locations[1] % self.size, locations[1] // self.size])
        self.bad1_pos = np.array([locations[2] % self.size, locations[2] // self.size])
        self.bad2_pos = np.array([locations[3] % self.size, locations[3] // self.size])
        
        observation = self._get_obs()
        info = self._get_info()
        
        # --- ADDED FOR RENDERING ---
        if self.render_mode == "human":
            self._render_frame()
        # ---------------------------

        return observation, info

    def step(self, action):
        intended_direction = self._action_to_direction[action]
        
        # Stochastic movement
        p = self.np_random.random()
        if p < 0.70: # 70% chance
            direction = intended_direction
        elif p < 0.85: # 15% chance
            direction = np.array([intended_direction[1], -intended_direction[0]])
        else: # 15% chance
            direction = np.array([-intended_direction[1], intended_direction[0]])

        new_pos = np.clip(self.agent_pos + direction, 0, self.size - 1)
        self.agent_pos = new_pos
        
        terminated = False
        reward = -0.1  # Your chosen reward
        
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10.0
            terminated = True
        elif np.array_equal(self.agent_pos, self.bad1_pos) or \
             np.array_equal(self.agent_pos, self.bad2_pos):
            reward = -10.0
            terminated = True
            
        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        # --- ADDED FOR RENDERING ---
        if self.render_mode == "human":
            self._render_frame()
        # ---------------------------

        return observation, reward, terminated, truncated, info

    # --- NEW RENDER FUNCTION ---
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    # --- NEW HELPER FUNCTION ---
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # White background
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid-square in pixels

        # Draw goal
        pygame.draw.rect(
            canvas,
            (0, 255, 0), # Green
            pygame.Rect(
                pix_square_size * self.goal_pos,
                (pix_square_size, pix_square_size),
            ),
        )
        
        # Draw bad cells
        pygame.draw.rect(
            canvas,
            (255, 0, 0), # Red
            pygame.Rect(
                pix_square_size * self.bad1_pos,
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 0), # Red
            pygame.Rect(
                pix_square_size * self.bad2_pos,
                (pix_square_size, pix_square_size),
            ),
        )
        
        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255), # Blue
            (self.agent_pos + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0, # Black
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0, # Black
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # Update the display
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    # --- UPDATED CLOSE FUNCTION ---
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()