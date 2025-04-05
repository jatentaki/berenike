import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp

class BinaryIoU:
    def __init__(self):
        self.outputs = []

    def on_step(self, model: pl.LightningModule, batch, predictions, stage) -> None:
        # Compute true positive, false positive, false negative, and true negative
        tp, fp, fn, tn = smp.metrics.get_stats(
            predictions.sigmoid() > 0.5,
            batch.labels.int(),
            mode='binary',
        )

        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        self.outputs.append({k: v.detach().cpu() for k, v in metrics.items()})

    def on_epoch_start(self, model: pl.LightningModule, stage):
        self.outputs = []

    def on_epoch_end(self, model: pl.LightningModule, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in self.outputs])
        fp = torch.cat([x["fp"] for x in self.outputs])
        fn = torch.cat([x["fn"] for x in self.outputs])
        tn = torch.cat([x["tn"] for x in self.outputs])

        # Compute IoU scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}/per_image_iou": per_image_iou,
            f"{stage}/dataset_iou": dataset_iou,
        }

        model.log_dict(metrics, prog_bar=True)