from replay.sampler import Sampler
from tests.environment import DummyVecEnv
import torch
from tqdm import tqdm

class TestRunner:
    def __init__(self, replay_sampler: Sampler, num_epochs: int, horizon_length: int) -> None:
        self.replay_sampler = replay_sampler

        self.num_epochs = num_epochs
        self.horizon_length = horizon_length

        self.obs = None

    def run(self) -> None:
        env = DummyVecEnv()
        self.obs = env.reset()

        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            for step in tqdm(range(self.horizon_length), desc="Step"):
                actions = torch.stack([torch.from_numpy(env.action_space.sample()) for _ in range(env.num_envs)])
                self.obs, rewards, self.dones, infos = env.step(actions)
                self.replay_sampler.add_step(
                    {'obs': self.obs['obs'], 'action': actions, 'reward': rewards, 'done': self.dones}, self.done)