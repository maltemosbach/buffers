from .dataset.episode_dataset import EpisodeDataset
from .dataset.fields import Scalar, Vector, ColorImage, PaddedPointcloud
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch
from typing import Dict


class BaseVisualizer:

    def __init__(self, dataset: EpisodeDataset, merge_pointclouds: bool = True) -> None:
        self.fields = dataset.fields
        self.merge_pointclouds = merge_pointclouds

    def visualize(self, sequence_batch: Dict[str, torch.Tensor], batch_index: int = 0) -> None:
        raise NotImplementedError


class MatplotlibVisualizer(BaseVisualizer):
    def __init__(self, dataset: EpisodeDataset) -> None:
        super().__init__(dataset)
        self.fig = plt.figure()
        self.axs = {}

        num_axes = len(self.fields)
        if self.merge_pointclouds and any([isinstance(self.fields[key], PaddedPointcloud) for key in self.fields]):
            num_axes -= sum([1 for key in self.fields if isinstance(self.fields[key], PaddedPointcloud)]) - 1

        for index in range(num_axes):
            name = list(self.fields.keys())[index]

            if isinstance(self.fields[name], PaddedPointcloud):
                if self.merge_pointclouds and "merged_pointcloud" not in self.axs:
                    self.axs["merged_pointcloud"] = self.fig.add_subplot(1, num_axes, index + 1, projection='3d')
                else:
                    self.axs[name] = self.fig.add_subplot(1, num_axes, index + 1, projection='3d')
            else:
                self.axs[name] = self.fig.add_subplot(1, num_axes, index + 1)

        plt.subplots_adjust(bottom=0.2)

    def visualize(self, sequence_batch: Dict[str, torch.Tensor], batch_index: int = 0) -> None:
        self.sequence_batch = sequence_batch
        if self.merge_pointclouds and any([isinstance(self.fields[key], PaddedPointcloud) for key in self.fields]):
            pointcloud_keys = [key for key in self.fields if isinstance(self.fields[key], PaddedPointcloud)]
            self.sequence_batch["merged_pointcloud"] = torch.cat(
                [self.sequence_batch[key] for key in pointcloud_keys], dim=2)

            for key in pointcloud_keys:
                del self.sequence_batch[key]

        self.batch_index = batch_index

        time_step_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.time_step_slider = Slider(time_step_ax, 'Time step', 0, sequence_batch[list(sequence_batch.keys())[0]].shape[1] - 1, valinit=0, valstep=1)
        self.time_step_slider.on_changed(self.update)

        self.show_frame(batch_index, 0)
        plt.show()

    def update(self, val):
        time_step = int(self.time_step_slider.val)
        self.show_frame(self.batch_index, time_step)

    def show_frame(self, batch_index, time_step: int) -> None:
        for key, value in self.sequence_batch.items():
            self.axs[key].cla()
            self.axs[key].set_title(key)
            self.axs[key].set_xticks([])

            if key == "merged_pointcloud" or isinstance(self.fields[key], PaddedPointcloud):
                self.visualize_padded_pointcloud(key, value[batch_index, time_step])

            elif isinstance(self.fields[key], Scalar):
                self.visualize_scalar(key, value[batch_index], time_step)
                self.axs[key].set_xlim(-0.5, value.shape[1] - 0.5)

            elif isinstance(self.fields[key], Vector):
                self.visualize_vector(key, value[batch_index, time_step])
                self.axs[key].set_yticks([])

            elif isinstance(self.fields[key], ColorImage):
                self.visualize_image(key, value[batch_index, time_step])
                self.axs[key].set_yticks([])

            else:
                assert False

    def visualize_scalar(self, key, vector: torch.Tensor, time_step: int) -> None:
        self.axs[key].plot(np.arange(0, len(vector)), vector.detach().cpu().numpy(), '--', color='darkblue',
                           marker='o')
        self.axs[key].axvspan(time_step - 0.5, time_step + 0.5, facecolor='lightsteelblue', edgecolor='none', alpha=.5)
        #self.axs[key].bar([0.], vector.detach().cpu().numpy())

    def visualize_vector(self, key: str, vector: torch.Tensor) -> None:
        self.axs[key].imshow(vector.unsqueeze(1).detach().cpu().numpy(), cmap="Blues")

    def visualize_image(self, key: str, image: torch.Tensor) -> None:
        self.axs[key].imshow(image.detach().cpu().numpy())

    def visualize_padded_pointcloud(self, key: str, pointcloud: torch.Tensor, unpad: bool = False) -> None:
        pointcloud = pointcloud.detach().cpu().numpy()

        if unpad:
            pointcloud = pointcloud[pointcloud[:, 3] > 0.5]
        semantic_features = pointcloud[:, 3]  # PaddedPointcloud features are [x, y, z, mask/semantics].

        colors = plt.get_cmap('jet')(semantic_features.squeeze() / np.max(semantic_features))[:, :3]
        self.axs[key].scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], color=colors)
