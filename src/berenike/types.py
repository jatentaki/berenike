from __future__ import annotations

from typing import Callable, NamedTuple

import torch
from torch import Tensor

class Slicer:
    """ A utility for easily slicing all tensors in a Batch. """
    def __init__(self, example: Batch):
        self.example = example

    def __getitem__(self, index) -> Batch:
        return self.example.apply_to_tensors(lambda t: t[index])

class Batch(NamedTuple):
    """
    Represents a training batch.
    `weights` indicates whether a given pixel should be masked out for loss/evaluation (e.g. it is outside of the image extent).
    """

    bitmaps: Tensor # [batch_size, 3, height, width]
    labels: Tensor # [batch_size, 1, height, width]
    weights: Tensor # [batch_size, 1, height, width]

    def apply_to_tensors(self, f: Callable[[Tensor], Tensor]) -> Batch:
        """ Applies a function to all tensors in the batch and returns a new batch. """
        applied = {}
        for field in self._fields:
            value = getattr(self, field)
            if torch.is_tensor(value):
                value = f(value)
            applied[field] = value

        return Batch(**applied)

    def unsqueeze(self, dim: int) -> Batch:
        return self.apply_to_tensors(lambda t: t.unsqueeze(dim))
    
    def to(self, *args, **kwargs) -> Batch:
        return self.apply_to_tensors(lambda t: t.to(*args, **kwargs))
    
    def clone(self):
        return self.apply_to_tensors(lambda t: t.clone())
    
    @property
    def slice(self) -> Slicer:
        return Slicer(self)
    
class NaNException(RuntimeError):
    pass