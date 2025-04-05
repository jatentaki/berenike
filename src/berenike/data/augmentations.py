import torch
from torch import nn

from berenike.types import Batch

class RandomCrop(nn.Module):
    def __init__(self, size: tuple[int, int]):
        super().__init__()

        self.size = size

    def __call__(self, example: Batch) -> Batch:
        h, w = example.bitmaps.shape[1:]
        max_h = h - self.size[0]
        max_w = w - self.size[1]

        start_h = torch.randint(0, max_h, ())
        start_w = torch.randint(0, max_w, ())

        return example.slice[..., start_h:start_h+self.size[0], start_w:start_w+self.size[1]]

class RandomFlip(nn.Module):
    def __init__(self, axis: int):
        super().__init__()

        self.axis = axis

    def __call__(self, example: Batch) -> Batch:
        if torch.rand(()) < 0.5:
            flip_fn = lambda t: torch.flip(t, (self.axis, ))
            example = example.apply_to_tensors(flip_fn)

        return example

class RandomRotate90(nn.Module):
    def __call__(self, example: Batch) -> Batch:
        k = torch.randint(0, 3, ())
        rot_fn = lambda t: torch.rot90(t, k, dims=(-1, -2)) 
        return example.apply_to_tensors(rot_fn)
