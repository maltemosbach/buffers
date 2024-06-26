from collections import defaultdict
from buffers.dataset.episode_dataset import EpisodeDataset
from buffers.utils import Batcher
import torch
from typing import Dict, Optional


class ReplayBuffer:
    def __init__(
        self,
        dataset: EpisodeDataset,
        batch_size: int,
        sequence_length: int = 1,
        chunk_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.dataloader = self.dataset.get_loader(batch_size, sequence_length, chunk_size, num_workers)
        self._iterator = None
        self.batcher = Batcher()
        self.current_episodes = defaultdict(lambda: defaultdict(list))

    @property
    def iterator(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        return self._iterator

    def add_step(self, step_data: Dict[str, torch.Tensor], done: bool | torch.Tensor) -> None:
        """Add a batch of step data to the buffer.

        Adds a batch of step data from parallel environments where all step data is of shape [batch_size, *data_size].

        Args:
            step_data (Dict[str, torch.Tensor]): Batch of step data to add.
            done (bool | torch.Tensor): Whether each episode is done (was done for vectorized environments).
        """
        step_data_batch, was_done_batch = self.batcher(step_data, done)

        for batch_index in range(self.batcher.batch_size):
            if was_done_batch[batch_index]:
                if len(self.current_episodes[batch_index].keys()) > 0 and len(next(iter(self.current_episodes[batch_index].values()))) > 1:
                    self.dataset.add_episode(self.current_episodes[batch_index])
                    self.current_episodes[batch_index] = defaultdict(list)

            for key, value in step_data_batch.items():
                self.current_episodes[batch_index][key].append(value[batch_index])

    def sample(self) -> Dict[str, torch.Tensor]:
        if self.is_empty:
            raise RuntimeError("Cannot sample from empty dataset.")
        else:
            batch_dict = next(self.iterator)
            for key, value in batch_dict.items():
                batch_dict[key] = value.to(self.dataset.fields[key].device)
            return batch_dict

    @property
    def num_episodes(self) -> int:
        return self.dataset.num_episodes

    @property
    def num_timesteps(self) -> int:
        return self.dataset.num_timesteps

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def is_empty(self) -> bool:
        return self.dataset.is_empty
