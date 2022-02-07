import torch
import pytorch_lightning as pl
import torchmetrics as tm

from typing import Any, Dict, List, Optional

CORE_DESC = "**Must not be used!** Base model of TweeSent NLP models. It includes the different steps (train/val/test & predict)."


class CoreTweeSent(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.desc = CORE_DESC
        metrics = {}

        # Define the loss.
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Define the metrics.
        if "acc" in self.hparams.metrics:
            metrics["acc"] = tm.Accuracy()
        if "recall" in self.hparams.metrics:
            metrics["recall"] = tm.Recall()

        # Add the metrics log per stage.
        basic_metrics = tm.MetricCollection(metrics)
        self.train_metrics = basic_metrics.clone(prefix="train_")
        self.val_metrics = basic_metrics.clone(prefix="val_")
        self.test_metrics = basic_metrics.clone(prefix="test_")
        self.pred_metrics = basic_metrics.clone(prefix="pred_")

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self(**x)
        loss = self.loss_fn(y_hat, y)
        if isinstance(self.log, list):
            for i, _ in enumerate(self.log):
                self.log[i]("loss/train", loss)
        else:
            self.log("loss/train", loss)

        met_out = self.train_metrics(y_hat, y)
        self.log_dict(met_out)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self(**x)
        loss = self.loss_fn(y_hat, y)
        if isinstance(self.log, list):
            for i, _ in enumerate(self.log):
                self.log[i]("loss/valid", loss)
        else:
            self.log("loss/valid", loss)

        met_out = self.val_metrics(y_hat, y)
        self.log_dict(met_out)

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self(**x)
        loss = self.loss_fn(y_hat, y)
        if isinstance(self.log, list):
            for i, _ in enumerate(self.log):
                self.log[i]("loss/test", loss)
        else:
            self.log("loss/test", loss)

        met_out = self.test_metrics(y_hat, y)
        self.log_dict(met_out)

        return loss

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        x, y = batch
        y_hat = self(**x)
        metrics = self.pred_metrics(y_hat, y)

        return y, y_hat, metrics

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduluer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2
        )
        return {
            "optimizer": optimizer,
            "interval": "epoch",
            "lr_scheduler": {"scheduler": scheduluer, "monitor": "loss/valid"},
        }

    def get_metrics(self, mode: List[str]) -> Dict:
        metrics = {}

        if "train" in mode or "all" in mode:
            metrics["Train"] = self.train_metrics.compute()
        if "val" in mode or "all" in mode:
            metrics["Validation"] = self.val_metrics.compute()
        if "test" in mode or "all" in mode:
            metrics["Test"] = self.test_metrics.compute()
        if "pred" in mode or "all" in mode:
            metrics["Predict"] = self.pred_metrics.compute()

        return metrics
