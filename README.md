# 🗄️ buffers

This repository is designed to provide **two main features**:

- **easy bookkeeping of experience:** the [`ReplayBuffer`](./buffers/replay_buffer.py) provides a simple interface to store and replay arbitrary 
  experiences from single or parallelized environments while keeping episodes intact,
- **efficient and scalable storage**: experiences are stored in an [`EpisodeDataset`](./buffers/dataset/episode_dataset.py), which 
  builds on the PyTorch [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) for efficient multi-process sampling.


## Getting started
The [quickstart notebook](./notebooks/quickstart.ipynb) provides a brief overview of the main features of the package.
Additional notebooks are available in the [notebooks](./notebooks) directory, demonstrating examples for multi-process 
sampling and composing of datasets.