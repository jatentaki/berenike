import torch
import torch.nn as nn

from berenike.types import Batch

class WeightedLossFn(nn.Module):
    """ Weights the inner loss function by the extent of the labels. """

    def __init__(
        self,
        inner: nn.Module = nn.CrossEntropyLoss(reduction='none'),
    ):
        super().__init__()
        self.inner = inner
    
    def forward(self, predictions: torch.Tensor, example: Batch) -> torch.Tensor:
        pixelwise_loss = self.inner(predictions, example.labels)
        return torch.dot(pixelwise_loss, example.weights.flatten()) / predictions.shape[0]