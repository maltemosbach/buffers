from collections.abc import Sequence
import torch
from typing import Dict, Tuple, Union


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


def to_batch(
    step_data: Dict[str, torch.Tensor], done: Union[bool, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
    batch_size = get_batch_size(step_data)

    # If the data does not have a leading batch dimension, add it.
    if batch_size is None:
        step_data = {key: value[None] for key, value in step_data.items()}
        batch_size = 1

    # If 'done' is a bool, convert it to a tensor.
    if isinstance(done, bool):
        done = torch.tensor([done] * batch_size)
    elif isinstance(done, torch.Tensor) and done.ndim == 0:
        done = done.view(1)

    if done.shape != (batch_size,):
        raise ValueError(f"'done' has shape {done.shape} but 'step_data' has batch size {batch_size}.")

    return step_data, done, batch_size


def get_batch_size(step_data: Dict[str, torch.Tensor]) -> Union[int, None]:
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
