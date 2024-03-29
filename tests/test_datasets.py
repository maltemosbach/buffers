import pytest
import torch

from replay.replay_buffer import ReplayBuffer
from replay.dataset.individual_file import IndividualFileDataset
from replay.dataset.ring_buffer import RingBufferDataset


EPISODE_DATASETS = [RingBufferDataset, IndividualFileDataset]


class TestDatasets:

    @pytest.mark.parametrize("dataset_class", EPISODE_DATASETS)
    def test_add_from_single_environment(self, dataset_class) -> None:
        dataset = dataset_class(capacity=100)
        buffer = ReplayBuffer(dataset, batch_size=5, sequence_length=4)

        for time_step in range(31):
            buffer.add_step({"obs": torch.zeros(10), "act": torch.zeros(5)}, False)
        buffer.add_step({"obs": torch.zeros(10), "act": torch.zeros(5)}, True)

        assert dataset.num_timesteps == 32
        assert dataset.num_episodes == 1

        sequence_batch = buffer.sample()

        assert sequence_batch["obs"].shape == (5, 4, 10)
        assert sequence_batch["act"].shape == (5, 4, 5)

    @pytest.mark.parametrize("dataset_class", EPISODE_DATASETS)
    def test_add_from_parallel_environments(self, dataset_class, num_envs: int = 4) -> None:
        dataset = dataset_class(capacity=1000)
        buffer = ReplayBuffer(dataset, batch_size=5, sequence_length=4)

        # Test adding data with boolean done signal.
        for time_step in range(31):
            buffer.add_step({"obs": torch.zeros(num_envs, 10), "act": torch.zeros(num_envs, 5)}, False)
        buffer.add_step({"obs": torch.zeros(num_envs, 10), "act": torch.zeros(num_envs, 5)}, True)

        assert dataset.num_timesteps == 32 * num_envs
        assert dataset.num_episodes == num_envs

        # Test adding data with tensor done signal.
        for time_step in range(10):
            buffer.add_step({"obs": torch.zeros(num_envs, 10), "act": torch.zeros(num_envs, 5)}, False)
        is_done = torch.zeros(num_envs, dtype=torch.bool)
        is_done[0] = True
        buffer.add_step({"obs": torch.zeros(num_envs, 10), "act": torch.zeros(num_envs, 5)}, is_done)

        assert dataset.num_timesteps == 32 * num_envs + 11
        assert dataset.num_episodes == num_envs + 1

        sequence_batch = buffer.sample()
        assert sequence_batch["obs"].shape == (5, 4, 10)
        assert sequence_batch["act"].shape == (5, 4, 5)

    @pytest.mark.parametrize("dataset_class", EPISODE_DATASETS)
    def test_episode_eviction(self, dataset_class) -> None:
        pass

    @pytest.mark.parametrize("dataset_class", EPISODE_DATASETS)
    def test_sequence_integrity(self, dataset_class) -> None:
        pass
        #buffer.add_step({"obs": torch.cat(torch.arange(num_envs), torch.ones(num_envs) * time_step), "act": torch.zeros(num_envs, 5)}, False)


