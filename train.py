import os

import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from lightning.pytorch.callbacks import ModelCheckpoint

from berenike.models import Segmenter
from berenike.loss import WeightedLossFn
from berenike.vis import BinaryVisualisationCallback
from berenike.data import TumulusDataModule

EPOCH_SIZE = 500
BATCH_SIZE = 32

data = TumulusDataModule(
    path="data",
    batch_size=BATCH_SIZE,
    n_train=BATCH_SIZE*EPOCH_SIZE,
    num_workers=12,
)

inner = smp.Unet(
    encoder_name='resnet18',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    encoder_depth=3,
    decoder_channels=(256, 128, 32),
    decoder_attention_type=None,
)

model = Segmenter(
    inner=inner,
    preprocess=torch.nn.Identity(),
    loss_fn=WeightedLossFn(
        inner=smp.losses.FocalLoss(mode='binary', reduction='none'),
    ),
)

callbacks = [
    BinaryVisualisationCallback(n_val=10, n_train=4),
    ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        filename='best-loss',
    ),
    ModelCheckpoint(
        monitor='val/dataset_iou',
        mode='max',
        filename='best-dataset-iou',
    ),
]

def trainer() -> pl.Trainer:
    current_path = os.path.split(__file__)[0]
    default_root_dir = os.path.join(current_path, 'training-results')
    return pl.Trainer(
        default_root_dir=default_root_dir,
        accelerator='cuda',
        devices=1,
        max_epochs=10,
        precision='16-mixed',
        callbacks=[*callbacks],
    )

if __name__ == '__main__':
    trainer().fit(model, data)