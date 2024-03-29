from collections import deque
from buffers.dataset.episode_dataset import EpisodeDataset
from buffers.dataset.fields import Fields
from buffers.dataset.info import EpisodeDatasetInfo
from buffers.evictor.base import Evictor
import torch
from typing import List, Dict, Optional
import warnings


class RingBufferDataset(EpisodeDataset):
    def __init__(
        self,
        capacity: int,
        info: Optional[EpisodeDatasetInfo] = None,
        evictor: Optional[Evictor] = None,
        device: torch.device = torch.device('cpu'),
        half_precision: bool = False,
    ) -> None:
        super().__init__(capacity, info, evictor)
        self.device = device
        self.half_precision = half_precision
        self.ring_buffer = {}

        # Indices used to keep track of episodes stored in contiguous memory.
        self.episode_boundaries = deque()
        self.head = 0
        self.tail = 0

    def append_episode(self, episode: Dict[str, List]) -> None:
        # Determine place for the new episode.
        start = self.tail
        end = self.tail + len(next(iter(episode.values())))
        self.episode_boundaries.append((start, end))

        # The episode fits in the buffer without wrapping.
        if end <= self.capacity:
            for key, value in episode.items():
                self.ring_buffer[key][start:end] = torch.stack(value).to(self.device, dtype=self.ring_buffer[key].dtype)

        # The episode wraps around the buffer.
        else:
            split = self.capacity - start
            end %= self.capacity  # Wrap remainder of episode around to the start of the buffer.
            for key, value in episode.items():
                self.ring_buffer[key][start:] = torch.stack(value[:split]).to(self.device, dtype=self.ring_buffer[key].dtype)
                self.ring_buffer[key][:end] = torch.stack(value[split:]).to(self.device, dtype=self.ring_buffer[key].dtype)

        # Update tail index.
        self.tail = end % self.capacity

    def evict_episode(self, episode_index: int) -> None:
        if episode_index != 0:
            raise NotImplementedError("Only the oldest episode can be evicted from RingBufferDataset.")

        # Set head index to the start of the second-oldest episode.
        self.head = self.episode_boundaries[1][0] if len(self.episode_boundaries) > 1 else self.tail

        # Remove the oldest episode from the episode boundaries.
        self.episode_boundaries.popleft()

    def _initialize_from_fields(self, fields: Fields) -> None:
        for key, value in fields.items():
            if self.half_precision and value.dtype == torch.float32:
                self.ring_buffer[key] = torch.empty(
                    (self.capacity, *value.shape), dtype=torch.float16, device=self.device)
            else:
                self.ring_buffer[key] = torch.empty(
                    (self.capacity, *value.shape), dtype=value.dtype, device=self.device)

    @staticmethod
    def _get_preferred_num_workers(num_workers: Optional[int] = None) -> int:
        if num_workers is not None and num_workers > 0:
            warnings.warn("RingBufferDataset does not support multiprocessing. Overriding 'num_workers' to 0.")
        return 0

    @staticmethod
    def _get_preferred_batch_chunking(batch_size: int, chunk_size: Optional[int] = None) -> int:
        if chunk_size is not None and chunk_size < batch_size:
            warnings.warn("Setting the 'chunk_size' smaller than the 'batch_size' does not make sense for "
                          "the RingBufferDataset, which uses only one process. Overriding 'chunk_size' to "
                          "'batch_size'.")
        return batch_size

    @staticmethod
    def _get_preferred_memory_pinning(pin_memory: Optional[bool]) -> bool:
        if pin_memory:
            warnings.warn("RingBufferDataset does not support memory pinning. Overriding 'pin_memory' to False.")
        return False

    def iter(self) -> Dict[str, torch.Tensor]:
        # Sample random episodes.
        episode_indices = torch.randint(len(self.episode_boundaries), (self.chunk_size,))
        # Get tensor of episode boundaries.
        selected_boundaries = torch.tensor([self.episode_boundaries[i] for i in episode_indices])

        if self.sequence_length is None:
            assert self.chunk_size == 1
            # Sequence indices span the full episode.
            start_indices = selected_boundaries[:, 0]
            linear_sequence_indices = start_indices.unsqueeze(1) + torch.arange(
                selected_boundaries[:, 1] - selected_boundaries[:, 0])
        else:
            # Compute maximum offset that can be used for each sequence.
            max_offsets = torch.remainder(
                selected_boundaries[:, 1] - selected_boundaries[:, 0], self.capacity) - self.sequence_length
            offsets = (max_offsets * torch.rand(self.chunk_size)).long()

            # Sample start indices for each sequence.
            start_indices = selected_boundaries[:, 0] + offsets
            linear_sequence_indices = start_indices.unsqueeze(1) + torch.arange(self.sequence_length)

        # Adjust indices to account for circular buffer.
        sequence_indices = torch.remainder(linear_sequence_indices, self.capacity)

        return {key: self.ring_buffer[key][sequence_indices].to(device=self.fields[key].device)
                for key in self.ring_buffer.keys()}
