from collections import defaultdict
from functools import partial, wraps
import gymnasium as gym
from prettytable import PrettyTable
from buffers import ReplayBuffer
from buffers.utils import to_torch
import time
from typing import Any, Callable, Optional


class StopWatch:
    def __init__(self) -> None:
        self.records = defaultdict(list)

    def __repr__(self) -> str:
        table = PrettyTable()
        table.field_names = ["Function", "Avg Duration (s)", "Min Duration (s)", "Max Duration (s)", "Calls #"]
        table.align["Function"] = "r"
        for func_name, durations in self.records.items():
            table.add_row([func_name, sum(durations) / len(durations), min(durations), max(durations), len(durations)])
        return str(table)

    def get_duration(self, func: Optional[Callable] = None, name: Optional[str] = None) -> Callable:
        if func is None:
            return partial(self.get_duration, name=name)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            record_name = name if name is not None else func.__name__
            self.records[record_name].append(time.time() - start_time)
            return result
        return wrapper

    def reset(self) -> None:
        self.records = defaultdict(list)


def add_sample_data(replay_buffer: ReplayBuffer, num_steps: int = 1000) -> None:
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.action_space.seed(42)
    observation, info = env.reset(seed=42)

    for _ in range(num_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = to_torch(env.step(action))
        if terminated or truncated:
            observation, info = to_torch(env.reset())
        replay_buffer.add_step({"obs": observation, "reward": reward}, done=terminated or truncated)
    env.close()
