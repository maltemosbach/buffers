# buffers

This repository is designed to provide **two main features**:

- 📖 **bookkeeping of experience:** the [`ReplayBuffer`](./buffers/replay_buffer.py) provides a simple interface to store and replay arbitrary 
  experiences from single or parallelized environments while keeping episodes intact,
- ⚡ **efficient and scalable storage**: experiences are stored in an [`EpisodeDataset`](./buffers/dataset/episode_dataset.py), which 
  builds on the PyTorch [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) for efficient multi-process sampling.


## Installation
The only mandatory dependency is [PyTorch](https://pytorch.org/get-started/locally/). buffers can be installed as
follows:
```bash
pip install -e .
```
Further dependencies used for testing and visualization can be installed via:
```bash
pip install -e .[extras]
```

## Getting started
The [quickstart notebook](./notebooks/quickstart.ipynb) provides a brief overview of the main features of the package.
Additional notebooks are available in the [notebooks](./notebooks) directory, demonstrating examples for multi-process 
sampling and composing of datasets.

## Vectorized Environments
In a vectorized environment, the individual environments are reset automatically at the end of an episode.
A typical `step` function in a vectorized environment looks like this:
```python
def step(self, action: torch.Tensor):
   """Execute one time-step of the environment dynamics and reset terminated environments.
   
   1. Process actions and step physics.
   2. Reset environments that have reached a terminal state.
   3. Compute observations and return step information.
   """
```
Therefore, if ` done[env_index] == True`, the observation returned for this environment is actually the first observation of
the following episode not the last observation of the episode that just ended 
[[stable-baselines]](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).

We assume vectorized environments to adhere to this behavior. If `step_data` is added with a batch dimension, the 
environment is assumed to behave as a vectorized environment and `done` is best understood as a `was_done` indicator.
If `step_data` is added without a batch dimension, the environment is assumed to behave as a regular, single 
environment.