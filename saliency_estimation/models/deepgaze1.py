"""ResNet model for 3D MRI images"""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


class DeepGaze1(pl.LightningModule):
    """
    # TODO
    """

    def __init__(self, args):
        """
        init function
        Args:
            args (argparse.Namespace): Arguments for the system
        """
        super().__init__()
        self.save_hyperparameters()
        # Variables

        # Model

    def forward(self, x):
        """
        Forward function for the model
        """
        x = self.model(x)
        # ...
        output = F.softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        # getting outputs
        data, target = batch
        y_hat = self.forward(data)
        # metrics
        acc = torchmetrics.functional.accuracy(y_hat, target, num_classes=self.num_classes)
        auroc = torchmetrics.functional.auroc(y_hat, target, num_classes=self.num_classes)
        loss = F.cross_entropy(y_hat, target)
        # logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_AUROC", auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # getting outputs
        data, target = batch
        y_hat = self.forward(data)
        # metrics
        acc = torchmetrics.functional.accuracy(y_hat, target, num_classes=self.num_classes)
        auroc = torchmetrics.functional.auroc(y_hat, target, num_classes=self.num_classes)
        loss = F.cross_entropy(y_hat, target)
        # logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_AUROC", auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # getting outputs
        data, target = batch
        y_hat = self.forward(data)
        # metrics
        acc = torchmetrics.functional.accuracy(y_hat, target, num_classes=self.num_classes)
        auroc = torchmetrics.functional.auroc(y_hat, target, num_classes=self.num_classes)
        loss = F.cross_entropy(y_hat, target)
        # logging
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_AUROC", auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-3,
            threshold_mode="abs",
            min_lr=1e-7,
            verbose=True,
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,  # The LR schduler
            "interval": "epoch",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
            "reduce_on_plateau": True,  # For ReduceLROnPlateau scheduler
            "monitor": "val_loss",  # Metric for ReduceLROnPlateau to monitor
            "strict": False,  # Whether to crash the training if `monitor` is not found
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
