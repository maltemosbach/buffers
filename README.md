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