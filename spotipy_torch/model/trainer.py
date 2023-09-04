from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple


import lightning.pytorch as pl
import logging
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn

from .config import SpotipyTrainingConfig
from .losses import AdaptiveWingLoss
from ..utils import prob_to_points, points_matching_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _img_to_rgb_or_gray(x: torch.Tensor):
    """  x.shape = C, H, W"""
    assert x.ndim == 3
    n_channels = x.shape[0] 
    if n_channels in (1,3):
        return x 
    elif n_channels == 2:
        return torch.cat((x[:1], x[1:], x[:1]), dim=0)
    else: 
        _cs = np.linspace(0,n_channels-1, 3).astype(int)
        return x[_cs]
        
    

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

        if self.model.config.compute_flow:
            self._loss_func_flow = nn.MSELoss()

        self._valid_inputs = []
        self._valid_targets = []
        self._valid_outputs = []
        self._valid_flows = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.model(x)

    @staticmethod
    def _wrap_loss(
        func: callable, pos_weight: float = 1.0, positive_threshold: float = 0.01
    ) -> callable:
        """Wrap a loss function to add a weight to positive pixels."""

        def _loss(y, target):
            loss = func(y, target)
            if pos_weight == 0:
                return loss.sum() / y.numel()
            mask_pos_tgt = target > positive_threshold
            mask_pos_y = y > positive_threshold
            mask_pos = torch.max(mask_pos_tgt, mask_pos_y)
            wloss = (1 + pos_weight * mask_pos) * loss
            return wloss.sum() / y.numel()

        return _loss

    def _loss_switcher(self, loss_kwargs: dict = {}) -> Tuple[callable]:
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
        return tuple(
            self._wrap_loss(
                loss_cls(reduction="none", **loss_kwargs),
                self.training_config.pos_weight if level == 0 else 0,
            )
            for level in range(self.model._levels)
        )

    def _common_step(self, batch):
        heatmap_lvs = [batch[f"heatmap_lv{lv}"] for lv in range(self.model._levels)]
        imgs = batch["img"]

        if self.model.config.compute_flow:
            flow = batch["flow"]

        out = self(imgs)
        pred_heatmap = out["heatmaps"]

        loss_heatmap = sum(
            tuple(
                loss_f(pred_heatmap[lv], heatmap_lvs[lv]) / 4**lv
                for lv, loss_f in zip(range(self.model._levels), self._loss_funcs)
            )
        )
        loss = loss_heatmap

        if self.model.config.compute_flow:
            pred_flow = out["flow"]
            loss_flow = self._loss_func_flow(pred_flow, flow)
            loss = loss + loss_flow
        else:
            pred_flow = None
            loss_flow = torch.tensor(0.0)

        return dict(
            loss=loss,
            loss_flow=loss_flow,
            loss_heatmap=loss_heatmap,
            pred_heatmap=pred_heatmap,
            pred_flow=pred_flow,
        )

    def training_step(self, batch, batch_idx):
        out = self._common_step(batch)

        self.log_dict(
            {
                "heatmap_loss": out["loss_heatmap"],
                "flow_loss": out["loss_flow"],
                "train_loss": out["loss"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        [batch[f"heatmap_lv{lv}"] for lv in range(self.model._levels)]
        img = batch["img"]

        out = self._common_step(batch)

        heatmap = (
            self.model._sigmoid(out["pred_heatmap"][0].squeeze(0).squeeze(0))
            .detach()
            .cpu()
            .numpy()
        )
        if self.model.config.compute_flow:
            flow = out["pred_flow"].squeeze(0).detach().cpu().numpy()
        else:
            flow = None

        self._valid_inputs.append(img[0].detach().cpu().numpy())
        self._valid_targets.append(batch["heatmap_lv0"][0, 0].detach().cpu().numpy())
        self._valid_outputs.append(heatmap)
        self._valid_flows.append(flow)

        self.log_dict(
            {
                "val_loss": out["loss"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_validation_epoch_end(self) -> None:
        valid_tgt_centers = [
            prob_to_points(t, exclude_border=False, min_distance=1)
            for t in self._valid_targets
        ]
        valid_pred_centers = [
            prob_to_points(p, exclude_border=False, min_distance=1)
            for p in self._valid_outputs
        ]
        stats = points_matching_dataset(
            valid_tgt_centers,
            valid_pred_centers,
            cutoff_distance=2 * self.training_config.sigma + 1,
            by_image=True,
        )
        val_f1, val_acc = stats.f1, stats.accuracy
        self.log_dict(
            {
                "val_f1": np.float32(val_f1),
                "val_acc": np.float32(val_acc),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if not self.trainer.sanity_checking:
            self.log_images()
        self._valid_inputs.clear()
        self._valid_targets.clear()
        self._valid_outputs.clear()
        self._valid_flows.clear()

    def log_images(self):
        n_images_to_log = min(3, len(self._valid_inputs))
        if isinstance(self.logger, pl.loggers.WandbLogger):  # Wandb logger
            self.logger.log_image(
                key="input",
                images=[
                    _img_to_rgb_or_gray(v).transpose(1, 2, 0) for v in self._valid_inputs[:n_images_to_log]
                ],
                step=self.global_step,
            )
            self.logger.log_image(
                key="target",
                images=[cm.magma(v) for v in self._valid_targets[:n_images_to_log]],
                step=self.global_step,
            )
            self.logger.log_image(
                key="output",
                images=[cm.magma(v) for v in self._valid_outputs[:n_images_to_log]],
                step=self.global_step,
            )
            if self.model.config.compute_flow:
                self.logger.log_image(
                    key="flow",
                    images=[
                        0.5 * (1 + v.transpose(1, 2, 0))
                        for v in self._valid_flows[:n_images_to_log]
                    ],
                    step=self.global_step,
                )

        elif isinstance(
            self.logger, pl.loggers.TensorBoardLogger
        ):  # TensorBoard logger
            for i in range(n_images_to_log):
                self.logger.experiment.add_image(
                    f"images/input/{i}",
                    _img_to_rgb_or_gray(self._valid_inputs[i]),
                    self.current_epoch,
                    dataformats="CHW",
                )
                self.logger.experiment.add_image(
                    f"images/target/{i}",
                    cm.magma(self._valid_targets[i]),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"images/output/{i}",
                    cm.magma(self._valid_outputs[i]),
                    self.current_epoch,
                    dataformats="HWC",
                )
                if self.model.config.compute_flow:
                    self.logger.experiment.add_image(
                        f"images/flow/{i}",
                        0.5 * (1 + self._valid_flows[i]),
                        self.current_epoch,
                        dataformats="CHW",
                    )
        return

    def configure_optimizers(self) -> dict:
        if self.training_config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                (p for p in self.parameters() if p.requires_grad),
                lr=self.training_config.lr,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.training_config.optimizer} not implemented."
            )
        
        out = dict(optimizer=optimizer) 
        
        if self.training_config.lr_reduce_patience>0:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=self.training_config.lr_reduce_patience,
                threshold=1e-4,
                min_lr=3e-6,
                cooldown=5,
                verbose=True,
            )
            out["lr_scheduler"] =  {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            } 
        
        return out

    def generate_dataloaders(
        self,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        num_train_samples: int = None,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.training_config.batch_size,
            sampler=torch.utils.data.RandomSampler(
                train_ds, num_samples=num_train_samples, replacement=True
            )
            if num_train_samples is not None
            else None,
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
    def __init__(
        self, logdir, train_config: SpotipyTrainingConfig, monitor: str = "val_loss"
    ):
        self._logdir = Path(logdir)
        self._monitor = monitor
        self._best = float("inf")
        self._train_config = train_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            log.info(f"Creating logdir {self._logdir} and saving training config...")
            self._logdir.mkdir(parents=True, exist_ok=True)
            self._train_config.save(self._logdir / "train_config.yaml")

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
                cutoff_distance=2 * self._train_config.sigma + 1,
                min_distance=1,
                exclude_border=False,
                batch_size=1,
                device=pl_module.device,
            )
            pl_module.model.save(self._logdir, which="last", update_thresholds=True)
            log.info("Saved last model with optimized threshold.")
