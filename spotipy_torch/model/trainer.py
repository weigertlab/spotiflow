from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple


import lightning.pytorch as pl
import logging
import torch
import torch.nn as nn
from .losses import AdaptiveWingLoss
from ..utils import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SpotipyTrainingWrapper(pl.LightningModule):
    """Supervised spot detector using a multi-stage neural network as a backbone for 
       feature extraction followed by a Feature Pyramid Network (Lin et al., CVPR '17)
       module to allow loss computation and optimization at different resolution levels.
    """
    def __init__(self, model, lr: float=3e-4, pos_weight: float=10., loss_f: str="bce"):
        super().__init__()
        self.model = model
        self._lr = lr
        self._loss_funcs = self._loss_switcher(loss_f, pos_weight)
        # For validation. maybe can be removed and replaced by a callback somehow?

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

    def _loss_switcher(self, loss_f_str: str, pos_weight: float=10., loss_kwargs: dict={}) -> Tuple[callable]:
        loss_f_str = loss_f_str.lower()
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
        return tuple(self._wrap_loss(loss_cls(reduction="none", **loss_kwargs), pos_weight if level==0 else 0) for level in range(self.model._levels))

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
        
        self._valid_outputs.clear()

    def on_train_end(self) -> None:
        self.model.optimize_threshold(
            val_ds=self.trainer.val_dataloaders.dataset,
            cutoff_distance=3,
            min_distance=1,
            exclude_border=False,
            batch_size=1
        )
        if self.trainer.checkpoint_callback is not None:
            self.model.save_model(
                save_path=self.trainer.checkpoint_callback.dirpath,
                which="last",
                only_config=True
            )
        return

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW((p for p in self.parameters() if p.requires_grad), lr=self._lr)
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
