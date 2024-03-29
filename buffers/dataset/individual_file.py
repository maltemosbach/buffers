from abc import ABC, abstractmethod
from dataclasses import dataclass
import datetime
import io
import numpy as np
from os import PathLike
import pathlib
import random
from buffers.dataset.episode_dataset import EpisodeDataset
from buffers.dataset.info import EpisodeDatasetInfo
from buffers.dataset.verification import debugging_check
from buffers.evictor.base import Evictor
import torch
from typing import List, Optional, Dict
import warnings


@dataclass
class EpisodeFile:
    id: int
    length: int
    suffix: str

    def __post_init__(self) -> None:
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = pathlib.Path(f"{self.id:07d}_{self.length}_{self.timestamp}{self.suffix}")


class EpisodeSerializationStrategy(ABC):
    suffix: str = ""

    @abstractmethod
    def save(self, episode: Dict[str, torch.Tensor], file: pathlib.Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, file: pathlib.Path) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class NumpyCompressedSerialization(EpisodeSerializationStrategy):
    suffix = ".npz"

    def save(self, episode: Dict[str, torch.Tensor], file: pathlib.Path) -> None:
        numpy_episode_dict = {key: value.detach().cpu().numpy() for key, value in episode.items()}
        with io.BytesIO() as bs:
            np.savez_compressed(bs, **numpy_episode_dict)
            bs.seek(0)
            with file.open('wb') as f:
                f.write(bs.read())

    def load(self, file: pathlib.Path) -> Dict[str, torch.Tensor]:
        with file.open('rb') as f:
            numpy_episode_dict = np.load(f)
            return {key: torch.from_numpy(value) for key, value in numpy_episode_dict.items()}


class TorchSerialization(EpisodeSerializationStrategy):
    suffix = ".pt"

    def save(self, episode: Dict[str, torch.Tensor], file: pathlib.Path) -> pathlib.Path:
        torch.save(episode, file)
        return file

    def load(self, file: pathlib.Path) -> Dict[str, torch.Tensor]:
        return torch.load(file)


class IndividualFileDataset(EpisodeDataset):
    """Dataset that stores each episode in a separate file."""

    def __init__(
        self,
        capacity: int,
        info: Optional[EpisodeDatasetInfo] = None,
        evictor: Optional[Evictor] = None,
        directory: Optional[PathLike] = "./individual_file_dataset",
        serialization_strategy: Optional[EpisodeSerializationStrategy] = None,
        max_sampling_attempts: Optional[int] = 100,
    ) -> None:
        super().__init__(capacity, info, evictor)
        self.serialization_strategy = serialization_strategy or NumpyCompressedSerialization()
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_sampling_attempts = max(1, max_sampling_attempts)
        self.episode_id = 0

    def append_episode(self, episode: Dict[str, List]) -> None:
        file = EpisodeFile(self.episode_id, len(next(iter(episode.values()))), self.serialization_strategy.suffix)
        self.episode_id += 1
        self.serialization_strategy.save({
            k: torch.stack(v).detach().cpu() for k, v in episode.items()}, self.directory / file.path)
        self._shared_list.append(file)

    def evict_episode(self, episode_index: int) -> None:
        file = self._shared_list.pop(episode_index)
        (self.directory / file.path).unlink(missing_ok=True)

    @staticmethod
    def _get_preferred_batch_chunking(batch_size: int, chunk_size: Optional[int] = None) -> int:
        if chunk_size is None:
            chunk_size = 1

        if chunk_size > 1:
            warnings.warn(f"Using a 'chunk_size' larger than 1 on the individual file dataset means that each worker "
                          f"will load multiple files on each __iter__ call. ")

        return chunk_size

    def iter(self) -> Dict[str, torch.Tensor]:
        # Sample chunk of episodes.
        chunk_list = []
        for _ in range(self.chunk_size):
            episode = self._sample_episode()

            # Sample sequence from episode.
            if self.sequence_length:
                sequence_index = np.random.randint(0, len(next(iter(episode.values()))) - self.sequence_length + 1)
                #return {k: v[sequence_index:sequence_index + self.sequence_length].unsqueeze(0) for k, v in episode.items()}
                chunk_list.append({k: v[sequence_index:sequence_index + self.sequence_length] for k, v in episode.items()})

            # Return one full episode.
            else:
                assert self.chunk_size == 1
                chunk_list.append(episode)

        return {k: torch.stack(v) for k, v in zip(*[chunk_list[0].keys(), zip(*[d.values() for d in chunk_list])])}

    def _sample_episode(self) -> Dict[str, torch.Tensor]:
        for attempt in range(self.max_sampling_attempts):
            selected_episode_path = random.choice(self.episode_paths)
            episode = self.serialization_strategy.load(self.directory / selected_episode_path)
            episode_length = len(next(iter(episode.values())))
            if self.sequence_length is None or episode_length >= self.sequence_length:
                break

        if self.sequence_length is not None and episode_length < self.sequence_length:
            raise RuntimeError(f"Could not sample episode of length {self.sequence_length} after "
                               f"{self.max_sampling_attempts} attempts.")

        return episode

    @debugging_check
    def _check_worker_view(self, episode_paths: List) -> None:
        print(f"Worker {self.worker_id} has {len(episode_paths)} episodes with paths {episode_paths}.")

    @debugging_check
    def _check_sampling_attempts(self, num_attempts: int) -> None:
        print(f"Sampled episode of sufficient length after {num_attempts} attempts.")

    @property
    def episode_paths(self) -> List[pathlib.Path]:
        return [file.path for file in self._shared_list if file.id % max(1, self.num_workers) == self.worker_id]

    @property
    def is_empty(self) -> bool:
        # There needs to be at least one episode per worker to sample.
        return len(self._shared_list) < max(1, self.num_workers)
