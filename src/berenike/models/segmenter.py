import torch
import lightning.pytorch as pl
from torch import nn, Tensor

from berenike.types import Batch, NaNException
from berenike.loss import WeightedLossFn
from berenike.metrics import BinaryIoU

class Segmenter(pl.LightningModule):
    def __init__(
        self,
        inner,
        preprocess: nn.Module = nn.Identity(),
        loss_fn: nn.Module = WeightedLossFn(),
    ):
        super().__init__()
        
        self.inner = inner
        self.preprocess = preprocess
        self.loss_fn = loss_fn
        self.metric = BinaryIoU()
    
    @property
    def n_classes(self) -> int:
        return self.inner.segmentation_head[0].out_channels
    
    def forward(self, images: Tensor) -> Tensor:
        return self.inner(images)
    
    def configure_optimizers(self):
        weights = []
        biases = []
        for n, p in self.named_parameters():
            if 'weight' in n:
                weights.append(p)
            else:
                biases.append(p)

        return torch.optim.AdamW(
            [
                {'params': weights, 'weight_decay': 1e-3},
                {'params': biases, 'weight_decay': 0.0},
            ],
            lr=1e-4,
        )
    
    def shared_step(self, batch: Batch, stage: str):
        """ The part of logic that is shared between train, val and test steps. """
        batch = self.preprocess(batch)
        predictions: Tensor = self.inner(batch.bitmaps)
        loss = self.loss_fn(predictions, batch)
        self.log(f'{stage}/loss', loss)

        if not torch.isfinite(loss):
            raise NaNException(f'Loss is NaN or Inf: {loss}')

        self.metric.on_step(self, batch, predictions, stage)
        
        return loss

    def shared_epoch_end(self, stage):
        self.metric.on_epoch_end(self, stage)
    
    def shared_epoch_start(self, stage):
        self.metric.on_epoch_start(self, stage)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def on_train_epoch_start(self) -> None:
        return self.shared_epoch_start("train")

    def on_training_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        return self.shared_epoch_start("val")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def on_test_epoch_start(self) -> None:
        return self.shared_epoch_start("test")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")