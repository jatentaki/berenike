import matplotlib.pyplot as plt
import lightning.pytorch as pl
from torch import Tensor
from skimage import color

from berenike.models import Segmenter
from berenike.types import Batch

def overlay_labels_binary(
    image: Tensor,
    labels: Tensor,
) -> Tensor:
    """
    Overlay binary labels on top of an image (turned to grayscale)."
    """
    colored = color.label2rgb(
        labels.squeeze(0).cpu().numpy(),
        image.permute(1, 2, 0).cpu().numpy(),
        bg_label=0,
    )
    return colored

class BinaryVisualisationCallback(pl.Callback):
    def __init__(self, n_val: int, n_train: int = 0, figure_side: int = 10, threshold: float = 0.5):
        super().__init__()
        self.n_val = n_val
        self.n_train = n_train
        self.figure_side = figure_side
        self.threshold = threshold
    
    def shared_batch_end(self, trainer, pl_module: Segmenter, batch: Batch, batch_idx: int, stage: str):
        batch = pl_module.preprocess(batch)
        predictions = pl_module(batch.bitmaps)
        figure = self.build_figure(
            batch.bitmaps,
            predictions,
            batch.labels,
        )
        pl_module.logger.experiment.add_figure(
            tag=f'{stage}/vis_callback/{batch_idx}',
            figure=figure,
            global_step=trainer.global_step,
        )
    
    def on_validation_batch_end(self, trainer, pl_module: Segmenter, outputs, batch: Batch, batch_idx):
        if batch_idx >= self.n_val:
            return
        self.shared_batch_end(trainer, pl_module, batch, batch_idx, 'val')
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        if batch_idx >= self.n_train:
            return
        self.shared_batch_end(trainer, pl_module, batch, batch_idx, 'train')
    
    def build_figure(
        self,
        inputs: Tensor,
        predictions: Tensor,
        labels: Tensor,
    ):
        fig, (a1, a2) = plt.subplots(
            1,
            2,
            figsize=(2 * self.figure_side, self.figure_side),
            constrained_layout=True,
        )

        predictions = predictions.detach().sigmoid() > self.threshold

        a1.imshow(overlay_labels_binary(inputs[0], predictions[0].squeeze(0)))
        a2.imshow(overlay_labels_binary(inputs[0], labels[0]))

        a1.set_title('Predictions')
        a2.set_title('Ground truth')

        for ax in (a1, a2):
            ax.axis('off')
            
        return fig