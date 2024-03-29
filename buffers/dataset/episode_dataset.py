from abc import abstractmethod
from multiprocessing import Manager
import numpy as np
import random
from buffers.dataset.fields import Fields
from buffers.dataset.info import EpisodeDatasetInfo
from buffers.evictor.base import Evictor
from buffers.evictor.fifo import FIFOEvictor
from buffers.dataset.verification import VerificationMode, basic_check
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Dict, Generator, List, Optional


class EpisodeDatasetInfoMixin:
    def __init__(self, info: EpisodeDatasetInfo) -> None:
        self.info = info

    def _update_stats_on_append(self, episode: Dict[str, List]) -> None:
        self.info.num_episodes += 1
        self.info.num_timesteps += len(next(iter(episode.values())))
        self.info.episode_lengths.append(len(next(iter(episode.values()))))

    def _update_stats_on_evict(self, episode_index: int) -> None:
        self.info.num_episodes -= 1
        self.info.num_timesteps -= self.info.episode_lengths.pop(episode_index)
        self.info.is_full = True  # is_full is set to True once episodes are started being evicted.

    @property
    def num_episodes(self) -> int:
        return self.info.num_episodes

    @property
    def num_timesteps(self) -> int:
        return self.info.num_timesteps

    @property
    def episode_lengths(self) -> List[int]:
        return self.info.episode_lengths

    @property
    def is_empty(self) -> bool:
        return self.info.is_empty

    @property
    def is_full(self) -> bool:
        return self.info.is_full

    @property
    def average_episode_length(self) -> float:
        return sum(self.episode_lengths) / self.num_episodes if self.num_episodes > 0 else 0

    def __len__(self) -> int:
        return self.num_timesteps

    @property
    def fields(self) -> Fields:
        return self.info.fields

    @property
    def verification_mode(self) -> VerificationMode:
        return self.info.verification_mode


class IterableMixin:
    """Adds iteration functionality to EpisodeDataset.

    Makes the EpisodeDataset compatible with the PyTorch IterableDataset and Dataloader framework.
    '_get_preferred' methods allow the dataset to overwrite defaults and limit allowed values for data loading.
    """

    def __iter__(self) -> Generator[Dict, None, None]:
        while True:
            yield self.iter()
            #yield self._check_sampled_chunk(self.iter())

    @abstractmethod
    def iter(self) -> Dict[str, torch.Tensor]:
        """Returns a chunk of sequences from the dataset.

        Returns:
            Dict[str, torch.Tensor]: A chunk of sequences from the dataset, where all values are of shape
                [chunk_size, sequence_length, ...]. If sequence_length is None, an entire episode is returned.
        """
        raise NotImplementedError

    @basic_check
    def _check_sampled_chunk(self, sequence_chunk: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, value in sequence_chunk.items():
            if value.shape[0] != self.chunk_size:
                raise RuntimeError(f"Expected chunk of size {self.chunk_size} for key '{key}', got {value.shape[0]}.")

            if self.sequence_length is not None and value.shape[1] != self.sequence_length:
                raise RuntimeError(f"Expected sequence of length {self.sequence_length} for key '{key}', got "
                                   f"{value.shape[1]}.")
        return sequence_chunk

    def get_loader(
        self,
        batch_size: int,
        sequence_length: Optional[int] = None,
        chunk_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ) -> DataLoader:

        # Set sampling parameters.
        self.chunk_size = self._get_preferred_batch_chunking(batch_size, chunk_size)
        self._check_batch_chunking(batch_size, self.chunk_size)

        if sequence_length is not None and sequence_length <= 0:
            raise ValueError(f"'sequence_length' must be greater than 0, got {sequence_length}.")
        self.sequence_length = sequence_length

        # Find compatible data loading parameters.
        self.num_workers = self._get_preferred_num_workers(num_workers)
        pin_memory = self._get_preferred_memory_pinning(pin_memory)

        # Initialize centralized multiprocessing variables.
        self.manager = Manager()
        self._shared_list = self.manager.list()

        return DataLoader(self, batch_size=batch_size // self.chunk_size, num_workers=self.num_workers,
                          pin_memory=pin_memory, collate_fn=self._sequence_chunk_collate_fn,
                          worker_init_fn=self._worker_init_fn)

    @staticmethod
    def _get_preferred_batch_chunking(batch_size: int, chunk_size: Optional[int] = None) -> int:
        """Defines the 'chunk_size'.

        A sampled batch is formed by composing 'batch_size' / 'chunk_size' chunks. If 'chunk_size' is None, the dataset
        will determine the ideal 'chunk_size' to split up the batch.

        Args:
            batch_size (int): Number of sampled sequences.
            chunk_size (int, optional): Requested chunk size.
        """

        return chunk_size or 1

    @staticmethod
    def _check_batch_chunking(batch_size: int, chunk_size: int) -> None:
        if batch_size <= 0 or chunk_size <= 0:
            raise ValueError(f"'batch_size' and 'chunk_size' must be greater than 0, got 'batch_size'={batch_size} and "
                             f"'chunk_size'={chunk_size}.")

        if batch_size % chunk_size != 0:
            raise ValueError(f"The 'chunk_size' must divide the 'batch_size' without remainder, got "
                             f"'chunk_size'={chunk_size} and 'batch_size'={batch_size}.")

    @staticmethod
    def _get_preferred_num_workers(num_workers: Optional[int] = None) -> int:
        """Defines the 'num_workers' of the DataLoader.

        Datasets may overwrite the 'num_workers', if the requested value is not supported or raise warnings for
        inefficient configurations.

        Args:
            num_workers (int): Requested number of workers.
        """

        return num_workers or 0

    @staticmethod
    def _get_preferred_memory_pinning(pin_memory: Optional[bool]) -> bool:
        """Defines the 'pin_memory' of the DataLoader.

        Datasets may overwrite the pin_memory, if the requested value is not supported.

        Args:
            pin_memory (bool): Requested memory pinning.
        """

        return pin_memory or True

    @staticmethod
    def _worker_init_fn(worker_id):
        seed = np.random.get_state()[1][0] + worker_id
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _sequence_chunk_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for chunks of sequences.

        Args:
            batch (List[Dict[str, torch.Tensor]]): Batch of sequence chunks to collate.

        Returns:
            Dict[str, torch.Tensor]: Collated batch.
        """
        # Concatenate all data at 0-th dimension.
        return {k: torch.cat([d[k] for d in batch]) for k in batch[0].keys()}

    @property
    def worker_id(self) -> int:
        try:
            return torch.utils.data.get_worker_info().id
        except AttributeError:
            return 0


class EpisodeDataset(EpisodeDatasetInfoMixin, IterableMixin, IterableDataset):
    def __init__(
        self,
        capacity: int,
        info: Optional[EpisodeDatasetInfo] = None,
        evictor: Optional[Evictor] = None,
    ) -> None:
        info = info or EpisodeDatasetInfo()
        EpisodeDatasetInfoMixin.__init__(self, info=info)
        self.capacity = capacity
        self.evictor = evictor or FIFOEvictor()

        if self.info.fields:
            self._initialize_from_fields(self.info.fields)

    def __repr__(self) -> str:
        summary = (f"{self.__class__.__name__}("
                   f"\n    num_episodes: {self.num_episodes}"
                   f"\n    num_timesteps: {self.num_timesteps}")
        summary += f"\n    fields: {list(self.fields.keys())} \n)" if self.fields else "\n    fields: None\n)"
        return summary

    def _initialize_from_fields(self, fields: Fields) -> None:
        """Used to initialize any storage via the fields of the dataset."""
        pass

    def _add_episode_to_storage(self, episode: Dict[str, List]) -> None:
        # If dataset fields are not defined, initialize them from the first episode.
        if not self.info.fields:
            self.info.fields = Fields.from_episode(episode)
            self._initialize_from_fields(self.info.fields)

        # Remove episodes until there is enough space to store the new episode.
        while len(self) + len(next(iter(episode.values()))) > self.capacity:
            evict_episode_index = self.evictor()
            self._update_stats_on_evict(evict_episode_index)

        # Add new episode to storage.
        self.append_episode(episode)
        self._update_stats_on_append(episode)

    def add_episode(self, episode: Dict[str, List]) -> None:
        if hasattr(self, "other_dataset"):
            self.other_dataset._add_episode_to_storage(episode)
            self.episodes_added_since_last_fetch += 1

            if self.is_full and self.episodes_added_since_last_fetch >= self.fetch_from_other_every:
                self.episodes_added_since_last_fetch = 0
                other_episodes = next(self.other_iterator)

                for i in range(next(iter(other_episodes.values())).shape[0]):
                    self._add_episode_to_storage({k: torch.unbind(v[i]) for k, v in other_episodes.items()})
            else:
                self._add_episode_to_storage(episode)
        else:
            self._add_episode_to_storage(episode)

    @abstractmethod
    def append_episode(self, episode: Dict[str, List]) -> None:
        """Append an episode to the storage.

        Args:
            episode (dict): Episode to append.
        """

        raise NotImplementedError

    @abstractmethod
    def evict_episode(self, episode_index: int) -> None:
        """Evict an episode from the storage.

        The episode at episode_index is evicted from the storage.

        Args:
            episode_index (int, optional): Index of the episode to evict.
        """

        raise NotImplementedError

    def extend(
        self,
        other: "EpisodeDataset",
        other_batch_size: Optional[int] = 64,
        other_num_workers: Optional[int] = None,
        fetch_from_other_every: Optional[int] = 1
    ) -> None:
        """Extend this dataset by another.

        When extending a dataset by another, once this dataset is full, it will fetch episodes from the other
        (ideally larger) dataset. This can be used to augment a faster, but smaller dataset with a larger one.

        Args:
            other (EpisodeDataset): EpisodeDataset to extend this dataset by.
            other_batch_size (int, optional): Batch size for fetching episodes from the other dataset.
            other_num_workers (int, optional): Number of workers used to sample from the other dataset.
            fetch_from_other_every (int, optional): Interval at which to fetch episodes from the other dataset.
        """

        self.other_dataset = other
        self.other_dataloader = self.other_dataset.get_loader(other_batch_size, None, 1, other_num_workers)
        self._other_iterator = None

        self.episodes_added_since_last_fetch = 0
        self.fetch_from_other_every = fetch_from_other_every

    @property
    def other_iterator(self):
        if self._other_iterator is None:
            self._other_iterator = iter(self.other_dataloader)
        return self._other_iterator
