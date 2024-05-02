from collections.abc import Sequence
import torch
from typing import Dict, Optional, Tuple


def get_batch_size(step_data: Dict[str, torch.Tensor]) -> Optional[int]:
    first_tensor = step_data[next(iter(step_data.keys()))]
    batch_size = safe_batch_size(first_tensor)

    for tensor in step_data.values():
        # If the batch dimension is not the same for all tensors, return None.
        if safe_batch_size(tensor) != batch_size:
            return None
    return batch_size


def safe_batch_size(tensor: torch.Tensor) -> int:
    if tensor.ndim == 0:
        return 1
    return tensor.shape[0]


class Batcher:
    def __init__(self):
        self.batch_size = None
        self._was_done_before = None  # The 'done' indicator from the previous step.

    def was_done_before(self, done: torch.Tensor) -> torch.Tensor:
        if self._was_done_before is None:
            was_done_before = torch.zeros_like(done)
        else:
            was_done_before = self._was_done_before
        self._was_done_before = done
        return was_done_before

    def __call__(self, step_data: Dict[str, torch.Tensor], done: bool | torch.Tensor) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor]:
        """Converts step data to a batch.

        Args:
            step_data (Dict[str, torch.Tensor]): Step data.
            done (bool | torch.Tensor): Whether the episode is done.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, int]: Batched 'step_data' and 'was_done' indicator.
        """

        batch_size = get_batch_size(step_data)

        # If 'done' is a bool, convert it to a tensor.
        if isinstance(done, bool):
            done = torch.tensor([done] * batch_size)
        elif isinstance(done, torch.Tensor) and done.ndim == 0:
            done = done.view(1)

        # Infer 'was_done' depending on whether this is a vectorized or regular environment.
        if batch_size is None:
            was_done = self.was_done_before(done)
        else:
            was_done = done

        # If the data does not have a leading batch dimension, add it.
        if batch_size is None:
            step_data = {key: value[None] for key, value in step_data.items()}
            batch_size = 1
        else:
            if done.shape != (batch_size,):
                raise ValueError(f"'done' has shape {done.shape} but 'step_data' has batch size {batch_size}.")

        if self.batch_size is None:
            self.batch_size = batch_size
        else:
            if self.batch_size != batch_size:
                raise ValueError(f"Batch size changed from {self.batch_size} to {batch_size}.")

        return step_data, was_done


def safe_convert_to_tensor(value):
    try:
        return torch.tensor(value)
    except RuntimeError:
        return value


def to_torch(input_value):
    if isinstance(input_value, Sequence) and not isinstance(input_value, (str, bytes, dict)):
        return [safe_convert_to_tensor(item) for item in input_value]
    else:
        return safe_convert_to_tensor(input_value)
