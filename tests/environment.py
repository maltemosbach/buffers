import gym

import torch
from tqdm import tqdm
from typing import Dict, Tuple, Any


class DummyVecEnv:
    def __init__(self, num_envs: int, observation_size: int, action_size: int) -> None:
        self.num_envs = num_envs
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(observation_size,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_size,))

        self.max_episode_length = 100
        self.progress_buffer = torch.zeros(self.num_envs)
        self.infos = {}

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs = torch.stack([torch.from_numpy(self.observation_space.sample()) for _ in range(self.num_envs)])
        obs_dict = {'obs': obs}
        rewards = torch.rand(self.num_envs)
        dones = self.progress_buffer > self.max_episode_length

        self.progress_buffer += 1
        self.reset_idx(torch.arange(self.num_envs)[dones])

        return obs_dict, rewards, dones, self.infos

    def reset(self) -> Dict[str, torch.Tensor]:
        """Called once to provide the first observations."""
        obs = torch.stack([torch.from_numpy(self.observation_space.sample()) for _ in range(self.num_envs)])
        return {'obs': obs}

    def reset_idx(self, env_ids: torch.Tensor):
        self.progress_buffer[env_ids] = 0
