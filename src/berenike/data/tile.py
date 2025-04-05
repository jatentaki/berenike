from __future__ import annotations
from dataclasses import dataclass

import torch
import numpy as np
from torch import Tensor

@dataclass
class Tile:
    """
    A class representing a tile of a Tumulus dataset.
    Conceptually very similar to a `archeo.types.Batch` but with a different meaning for the `extent` tensor.
    The two could probably be merged.
    """
    bitmap: Tensor
    labels: Tensor
    extent: Tensor # [height, width], indicates if the pixel is black (outside of the satellite image).

    def slice(self, index) -> Tile:
        return Tile(
            self.bitmap[index],
            self.labels[index],
            self.extent[index],
        )
    
    @staticmethod
    def load_npz(path: str) -> Tile:
        """ Loads a tile from a .npz file. """
        npz = np.load(path)
        bitmap = torch.from_numpy(npz['bitmap'])
        labels = torch.from_numpy(npz['labels']).to(bool)
        extent = torch.from_numpy(npz['extent']).to(bool)
        return Tile(bitmap=bitmap, labels=labels, extent=extent)

    def clip_to_extent(self) -> Tile:
        """
        Clips a tile to skip the empty space as indicated by `extent`.
        """
        def slice_to_bbox(s: np.ndarray) -> tuple[slice, slice]:
            ymin = s.any(axis=1).argmax()
            ymax = -1-s.any(axis=1)[::-1].argmax()
            xmin = s.any(axis=0).argmax()
            xmax = -1-s.any(axis=0)[::-1].argmax()

            return (slice(ymin, ymax), slice(xmin, xmax))
        
        s = slice_to_bbox(self.extent.numpy())

        return self.slice(s)