from pathlib import Path
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple


import lightning.pytorch as pl
import logging
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import wandb

from .config import SpotipyTrainingConfig
from .losses import AdaptiveWingLoss
from ..utils import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SpotipyTrainingWrapper(pl.LightningModule):
    """Supervised spot detector using a multi-stage neural network as a backbone for 
       feature extraction followed by a Feature Pyramid Network (Lin et al., CVPR '17)
       module to allow loss computation and optimization at different resolution levels.
    """
    def __init__(self, model, training_config: SpotipyTrainingConfig):
        super().__init__()
        self.model = model

        self.training_config = training_config

        self._loss_funcs = self._loss_switcher()

        self._valid_inputs = []
        self._valid_targets = []
        self._valid_outputs = []
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.model(x)

    @staticmethod
    def _wrap_loss(func: callable, pos_weight: float=1.0, positive_threshold: float=0.01) -> callable:
        """Wrap a loss function to add a weight to positive pixels."""
        def _loss(y, target):
            loss = func(y, target)
            if pos_weight == 0:
                return loss.sum()/y.numel()
            mask_pos_tgt = target>positive_threshold
            mask_pos_y = y>positive_threshold
            mask_pos = torch.max(mask_pos_tgt, mask_pos_y)
            wloss = (1+pos_weight*mask_pos)*loss
            return wloss.sum()/y.numel()
        return _loss

    def _loss_switcher(self, loss_kwargs: dict={}) -> Tuple[callable]:
        loss_f_str = self.training_config.loss_f.lower()
        if loss_f_str == "bce":
            loss_cls = nn.BCEWithLogitsLoss
        elif loss_f_str == "adawing":
            loss_cls = AdaptiveWingLoss
        elif loss_f_str == "mse":
            loss_cls = nn.MSELoss
        elif loss_f_str == "smoothl1":
            loss_cls = nn.SmoothL1Loss
        else:
            raise NotImplementedError(f"Loss function {loss_f_str} not implemented.")
        return tuple(self._wrap_loss(loss_cls(reduction="none", **loss_kwargs), self.training_config.pos_weight if level==0 else 0) for level in range(self.model._levels))

    def training_step(self, batch, batch_idx):
        heatmap_lvs = [batch[f"heatmap_lv{lv}"] for lv in range(self.model._levels)]
        imgs = batch["img"]

        out = self(imgs)
        loss = sum(tuple(loss_f(out[lv], heatmap_lvs[lv])/4**lv for lv, loss_f in zip(range(self.model._levels), self._loss_funcs)))
        self.log_dict({
            "train_loss": loss,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        heatmap_lvs = [batch[f"heatmap_lv{lv}"] for lv in range(self.model._levels)]
        img = batch["img"]

        out = self(img)
        
        loss = sum(tuple(loss_f(out[lv], heatmap_lvs[lv])/4**lv for lv, loss_f in zip(range(self.model._levels), self._loss_funcs)))
        
        high_lv_pred = self.model._sigmoid(out[0].squeeze(0).squeeze(0)).detach().cpu().numpy()
        self._valid_inputs.append(img[0].detach().cpu().numpy())
        self._valid_targets.append(batch["heatmap_lv0"][0,0].detach().cpu().numpy())
        self._valid_outputs.append(high_lv_pred)

        self.log_dict({
            "val_loss": loss,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        valid_outs = self._valid_outputs
        valid_tgts = self.trainer.val_dataloaders.dataset.centers
        val_pred_centers = [utils.prob_to_points(p, exclude_border=False, min_distance=1) for p in valid_outs]
        stats = utils.points_matching_dataset(valid_tgts, val_pred_centers, cutoff_distance=3, by_image=True)
        val_f1, val_acc = stats.f1, stats.accuracy
        self.log_dict({
            "val_f1": val_f1,
            "val_acc": val_acc,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.logger is not None:
            for i in range(min(3, len(self._valid_inputs))):
                self.logger.experiment.add_image(
                        "input", self._valid_inputs[i], self.current_epoch, dataformats="CHW"
                    )
                self.logger.experiment.add_image(
                        "target", self._valid_targets[i], self.current_epoch, dataformats="HW"
                    )
                self.logger.experiment.add_image(
                        "output", self._valid_outputs[i], self.current_epoch, dataformats="HW"
                    )
        
        self._valid_inputs.clear()
        self._valid_targets.clear()
        self._valid_outputs.clear()

    def log_images(self, valid_outs):
        n_images_to_log = min(3, len(self.trainer.val_dataloaders.dataset), len(valid_outs))
        if isinstance(self.logger, pl.loggers.WandbLogger): # Wandb logger
            self.logger.log_image(
                key="val_img", images=[self.trainer.val_dataloaders.dataset[idx]["img"][0].unsqueeze(-1).numpy() for idx in range(n_images_to_log)], step=self.global_step,
            )
            self.logger.log_image(
                key="val_tgt", images=[cm.magma(self.trainer.val_dataloaders.dataset[idx]["heatmap_lv0"][0].numpy()) for idx in range(n_images_to_log)], step=self.global_step,
            )
            self.logger.log_image(
                key="val_pred", images=[cm.magma(valid_outs[idx]) for idx in range(n_images_to_log)], step=self.global_step,
            )

        elif isinstance(self.logger, pl.loggers.TensorBoardLogger): # TensorBoard logger
            for i in range(n_images_to_log):
                self.logger.experiment.add_image(
                        "input", self._valid_inputs[i], self.current_epoch, dataformats="CHW"
                    )
                self.logger.experiment.add_image(
                        "target", self._valid_targets[i], self.current_epoch, dataformats="HW"
                    )
                self.logger.experiment.add_image(
                        "output", self._valid_outputs[i], self.current_epoch, dataformats="HW"
                    )
        return

    def configure_optimizers(self) -> dict:
        if self.training_config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW((p for p in self.parameters() if p.requires_grad), lr=self.training_config.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.training_config.optimizer} not implemented.")
        scheduler = ReduceLROnPlateau(optimizer,
                                      factor=0.5,
                                      patience=10,
                                      threshold=1e-4,
                                      min_lr=3e-6,
                                      cooldown=5,
                                      verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def generate_dataloaders(self, train_ds: torch.utils.data.Dataset, val_ds: torch.utils.data.Dataset, num_train_samples:int=None, num_workers:int=0) -> torch.utils.data.DataLoader:
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.training_config.batch_size,
            sampler = torch.utils.data.RandomSampler(train_ds, num_samples=num_train_samples, replacement=True) if num_train_samples is not None else None,
            shuffle=True if num_train_samples is None else False,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_dl, val_dl

# a model checkpoint that uses Spotipy.save() to save the model
class SpotipyModelCheckpoint(pl.callbacks.Callback):
    def __init__(self, logdir, train_config: SpotipyTrainingConfig, monitor: str = "val_loss"):
        self._logdir = Path(logdir)
        self._monitor = monitor
        self._best = float("inf")
        self._train_config = train_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            log.info(f"Creating logdir {self._logdir} and saving training config...")
            self._logdir.mkdir(parents=True, exist_ok=True)
            self._train_config.save(self._logdir/"train_config.yaml")

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            value = trainer.logged_metrics[self._monitor]
            if value < self._best:
                self._best = value
                log.info(f"Saved best model with {self._monitor}={value:.3f}.")
                pl_module.model.save(self._logdir, which="best")
    
    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            pl_module.model.optimize_threshold(
                val_ds=trainer.val_dataloaders.dataset,
                cutoff_distance=3,
                min_distance=1,
                exclude_border=False,
                batch_size=1,
                device=pl_module.device,
            )
            pl_module.model.save(self._logdir, which="last", update_thresholds=True)
            log.info("Saved last model with optimized threshold.")
    