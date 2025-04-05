import glob

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from berenike.types import Batch
from berenike.data.augmentations import RandomFlip, RandomRotate90
from berenike.data.tile import Tile

def tile_to_batch(tile: Tile) -> Batch:
    """ Converts a tile to a batch. """
    return Batch(
        # normalize to [0, 1] and permute to [batch_size, channels, height, width]
        bitmaps=tile.bitmap.permute(2, 0, 1).to(torch.float32) / 255,

        # add a batch dimension
        labels=tile.labels.to(torch.float32).unsqueeze(0),

        # define weights to sum to 1
        weights=tile.extent.to(torch.float32).unsqueeze(0) * (1 / tile.extent.numel()),
    )

class RandomCroppingTumulusDataset:
    """ A training dataset which samples random crops from a tile. """
    def __init__(
        self,
        tile: Tile,
        length: int,
        size: int = 512,
        min_extent: float = 0.25,
    ):
        self.tile = tile
        self.length = length
        self.size = size
        self.min_extent = min_extent

        self.augmentation = torch.nn.Sequential(
            RandomFlip(axis=-1),
            RandomRotate90(),
        )

    def sample_region_proposal(self) -> Tile:
        i = torch.randint(0, self.tile.bitmap.shape[0] - self.size, (1,)).item()
        j = torch.randint(0, self.tile.bitmap.shape[1] - self.size, (1,)).item()

        s = (slice(i, i + self.size), slice(j, j + self.size))

        return self.tile.slice(s)
    
    def __getitem__(self, index: int) -> Batch:
        del index # unused
        tile = self.sample_region_proposal()
        while tile.extent.sum() < self.min_extent * tile.extent.numel():
            tile = self.sample_region_proposal()
        example = tile_to_batch(tile)
        return self.augmentation(example)
    
    def __len__(self) -> int:
        return self.length

class TiledTumulusDataset:
    """ A validation dataset which loads full tiles from a directory one by one. """
    def __init__(self, path: str, divisible_by: int = 8):
        self.path = path
        self.tiles = [Tile.load_npz(f) for f in glob.glob(f'{path}/*.npz')]
        self.divisible_by = divisible_by
    
    def __len__(self):
        return len(self.tiles)
    
    def get_tile(self, index: int):
        tile = self.tiles[index]
        d = self.divisible_by
        h, w = tile.bitmap.shape[:2]
        tile = tile.slice(( # make tile sizes divisible by `self.divisible_by`
            slice(0, (h // d) * d),
            slice(0, (w // d) * d),
        ))
        return tile
    
    def __getitem__(self, index: int):
        return tile_to_batch(self.get_tile(index))

class TumulusDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        n_train: int,
        train_size: int = 512,
        min_extent: float = 0.25,
        batch_size: int = 32,
        num_workers: int = 8,
        divisible_by: int = 8,
    ):
        super().__init__()
        self.path = path
        self.n_train = n_train
        self.train_size = train_size
        self.min_extent = min_extent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.divisible_by = divisible_by

        assert self.train_size % self.divisible_by == 0
    
    def setup(self, stage=None):
        self.train_dataset = RandomCroppingTumulusDataset(
            Tile.load_npz(f'{self.path}/train-chunks/train.npz'),
            length=self.n_train,
            size=self.train_size,
            min_extent=self.min_extent,
        )
        self.val_dataset = TiledTumulusDataset(
            f'{self.path}/val-chunks',
            divisible_by=self.divisible_by,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=True,
        )