from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Union


import lightning.pytorch as pl
import logging
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn

from .config import SpotiflowTrainingConfig
from .losses import AdaptiveWingLoss
from ..data import collate_spots
from ..utils import prob_to_points, points_matching_dataset, remove_device_id_from_device_str

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _img_to_rgb_or_gray(x: torch.Tensor):
    """x.shape = C, H, W"""
    assert x.ndim == 3
    n_channels = x.shape[0]
    if n_channels in (1, 3):
        return x
    elif n_channels == 2:
        return torch.cat((x[:1], x[1:], x[:1]), dim=0)
    else:
        _cs = np.linspace(0, n_channels - 1, 3).astype(int)
        return x[_cs]


class SpotiflowTrainingWrapper(pl.LightningModule):
    """Lightning module that wraps a Spotiflow model and handles the training stage."""

    def __init__(self, model, training_config: SpotiflowTrainingConfig):
        """Initializes the SpotiflowTrainingWrapper.

        Args:
            model (SpotiflowModel): The model to train.
            training_config (SpotiflowTrainingConfig): The training configuration.
        """
        super().__init__()
        self.model = model

        self.training_config = training_config

        if self.training_config.loss_levels is None:
            self._loss_levels = self.model._levels
        elif self.training_config.loss_levels <= model._levels:
            self._loss_levels = self.training_config.loss_levels
        else:
            raise ValueError(
                f"Loss levels ({self.training_config.loss_levels}) must be less than or equal to model levels ({model._levels})."
            )

        self._heatmap_loss_funcs = self._heatmap_loss_switcher()
        self._flow_loss_func = self._flow_loss_switcher()

        self._valid_inputs = []
        self._valid_targets = []
        self._valid_targets_hm = []
        self._valid_outputs = []
        self._valid_flows = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor]: The output of the model.
        """
        return self.model(x)

    def _loss_weight(
        self,
        y: torch.Tensor,
        target: torch.Tensor,
        pos_weight: float = 1.0,
        positive_threshold: float = 0.01,
    ):
        """compute the loss weight mask for a given prediction and target"""
        weight = torch.ones(y.shape, dtype=torch.float32, device=y.device)
        if pos_weight == 0:
            return weight
        mask_pos_tgt = target >= positive_threshold
        mask_pos_y = y >= positive_threshold
        mask_pos = torch.max(mask_pos_tgt, mask_pos_y)
        weight = (1 + pos_weight * mask_pos) * weight
        return weight

    def _heatmap_loss_switcher(self, loss_kwargs: dict = {}) -> Tuple[callable]:
        """Helper function to switch between loss functions for the multiscale heatmap regression.

        Args:
            loss_kwargs (dict, optional): The keyword arguments to be passed to the loss function. Defaults to no arguments.

        Returns:
            Tuple[callable]: The loss functions to be applied at different resolution levels (from highest to lowest).
        """
        heatmap_loss_f_str = self.training_config.heatmap_loss_f.lower()
        if heatmap_loss_f_str == "bce":
            loss_cls = nn.BCEWithLogitsLoss
        elif heatmap_loss_f_str == "adawing":
            loss_cls = AdaptiveWingLoss
        elif heatmap_loss_f_str == "mse":
            loss_cls = nn.MSELoss
        elif heatmap_loss_f_str == "smoothl1":
            loss_cls = nn.SmoothL1Loss
        else:
            raise NotImplementedError(f"Loss function {heatmap_loss_f_str} not implemented.")
        return tuple(
            loss_cls(reduction="none", **loss_kwargs)
            for _ in range(self._loss_levels)
        )

    def _flow_loss_switcher(self, loss_kwargs: dict = {}) -> callable:
        """Helper function to switch between loss functions for the stereographic flow regression.

        Args:
            loss_kwargs (dict, optional): The keyword arguments to be passed to the loss function. Defaults to no arguments.

        Returns:
            callable: The loss function to be applied to the stereographic flow.
        """
        flow_loss_f_str = self.training_config.flow_loss_f.lower()
        if flow_loss_f_str == "l1":
            loss_cls = nn.L1Loss
        else:
            raise NotImplementedError(f"Loss function {flow_loss_f_str} not implemented.")
        return loss_cls(reduction="none", **loss_kwargs)

    def _common_step(self, batch):
        heatmap_lvs = [batch[f"heatmap_lv{lv}"] for lv in range(self._loss_levels)]
        imgs = batch["img"]

        if self.model.config.compute_flow:
            flow = batch["flow"]

        out = self(imgs)
        pred_heatmap = out["heatmaps"]

        loss_heatmaps = list(
            loss_f(pred_heatmap[lv], heatmap_lvs[lv]) / 4**lv
            for lv, loss_f in zip(range(self._loss_levels), self._heatmap_loss_funcs)
        )

        # reweight first
        loss_weight = self._loss_weight(
            pred_heatmap[0], heatmap_lvs[0], pos_weight=self.training_config.pos_weight
        )
        loss_heatmaps[0] = loss_heatmaps[0] * loss_weight

        loss_heatmap = sum([_loss.mean() for _loss in loss_heatmaps])

        loss = loss_heatmap

        if self.model.config.compute_flow:
            pred_flow = out["flow"]
            loss_flow = self._flow_loss_func(pred_flow, flow)
            loss_flow = (loss_flow * loss_weight).mean()
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
        """Training step of the model.

        Args:
            batch (torch.Tensor): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: loss of the model for the given batch
        """
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
            batch_size=self.training_config.batch_size,
        )

        return out["loss"]

    def validation_step(self, batch, batch_idx) -> None:
        """Validation step of the model. Populates the lists of inputs, targets and outputs for logging when the validation epoch ends.

        Args:
            batch (torch.Tensor): The batch of data.
            batch_idx (int): The index of the batch.
        """
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
        self._valid_targets.append(batch["pts"][0].detach().cpu().numpy())
        self._valid_targets_hm.append(batch["heatmap_lv0"][0,0].detach().cpu().numpy())
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
            batch_size=self.training_config.batch_size,
        )

    def on_validation_epoch_end(self) -> None:
        """Called when the validation epoch ends.
        Logs the F1 score and accuracy of the model on the validation set, as well as some sample images.
        """
        valid_pred_centers = [
            prob_to_points(
                p,
                exclude_border=False,
                min_distance=self.model.config.sigma,
                mode="fast",
            )
            for p in self._valid_outputs
        ]
        stats = points_matching_dataset(
            self._valid_targets,
            valid_pred_centers,
            cutoff_distance=2 * self.model.config.sigma + 1,
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
            batch_size=self.training_config.batch_size,
        )

        if not self.trainer.sanity_checking:
            self.log_images()
        self._valid_inputs.clear()
        self._valid_targets.clear()
        self._valid_targets_hm.clear()
        self._valid_outputs.clear()
        self._valid_flows.clear()

    def log_images(self):
        """Helper function to log sample images according to different loggers (wandb and tensorboard)."""
        n_images_to_log = min(3, len(self._valid_inputs))
        if isinstance(self.logger, pl.loggers.WandbLogger):  # Wandb logger
            self.logger.log_image(
                key="input",
                images=[
                    _img_to_rgb_or_gray(v).transpose(1, 2, 0)
                    for v in self._valid_inputs[:n_images_to_log]
                ],
                step=self.global_step,
            )
            self.logger.log_image(
                key="target",
                images=[cm.magma(v) for v in self._valid_targets_hm[:n_images_to_log]],
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
                    cm.magma(self._valid_targets_hm[i]),
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
        """Build the optimizer and scheduler.

        Returns:
            dict: The optimizer and scheduler to be used during training.
        """
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

        if self.training_config.lr_reduce_patience > 0:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=self.training_config.lr_reduce_patience,
                threshold=1e-4,
                min_lr=3e-6,
                cooldown=5,
                verbose=True,
            )
            out["lr_scheduler"] = {
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
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Generate the training and validation dataloaders according to the training configuration.

        Args:
            train_ds (torch.utils.data.Dataset): training dataset
            val_ds (torch.utils.data.Dataset): validation dataset
            num_train_samples (int, optional): number of training samples assumed to be in the training data. Useful to keep number of updates per epoch constant across different datasets. Defaults to None (len(train_ds)).
            num_workers (int, optional): number of workers to use. Defaults to 0.

        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: training and validation dataloaders
        """
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
            collate_fn=collate_spots,
            pin_memory=True,
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_spots,
        )
        return train_dl, val_dl


# a model checkpoint that uses Spotiflow.save() to save the model
class SpotiflowModelCheckpoint(pl.callbacks.Callback):
    """Callback to save the best model according to a given metric.
    Uses Spotiflow.save() to save the model for maximum flexibility.
    """

    def __init__(
        self,
        logdir: Union[str, Path],
        train_config: SpotiflowTrainingConfig,
        monitor: str = "val_loss",
    ):
        """

        Args:
            logdir (str): path to the directory where the model checkpoints will be saved.
            train_config (SpotiflowTrainingConfig): training configuration.
            monitor (str, optional): metric to be minimized. Defaults to "val_loss".
        """
        self._logdir = Path(logdir) if isinstance(logdir, str) else logdir
        self._monitor = monitor
        self._best = float("inf")
        self._train_config = train_config

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when the training starts.

        Args:
            trainer (pl.Trainer): lightning trainer object.
            pl_module (pl.LightningModule): lightning module object.
        """
        if trainer.is_global_zero:
            log.info(f"Creating logdir {self._logdir} and saving training config...")
            self._logdir.mkdir(parents=True, exist_ok=True)
            self._train_config.save(self._logdir / "train_config.yaml")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when each validation epoch ends.

        Args:
            trainer (pl.Trainer): lightning trainer object.
            pl_module (pl.LightningModule): lightning module object.
        """
        if trainer.is_global_zero and not trainer.sanity_checking:
            value = trainer.logged_metrics[self._monitor]
            if value < self._best:
                self._best = value
                log.info(f"Saved best model with {self._monitor}={value:.3f}.")
                pl_module.model.save(self._logdir, which="best")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when the training ends.

        Args:
            trainer (pl.Trainer): lightning trainer object.
            pl_module (pl.LightningModule): lightning module object.
        """
        if trainer.is_global_zero:
            # save last
            pl_module.model.optimize_threshold(
                val_ds=trainer.val_dataloaders.dataset,
                cutoff_distance=2 * trainer.model.model.config.sigma + 1,
                min_distance=1,
                exclude_border=False,
                batch_size=1,
                device=pl_module.device,
                subpix=trainer.model.model.config.compute_flow,
            )
            pl_module.model.save(self._logdir, which="last", update_thresholds=True)
            log.info("Saved last model with optimized thresholds.")
            # load best and optimize thresholds...
            device_str = remove_device_id_from_device_str(str(pl_module.device))
            pl_module.model.load(self._logdir, which="best", map_location=device_str)
            pl_module.model.optimize_threshold(
                val_ds=trainer.val_dataloaders.dataset,
                cutoff_distance=2 * trainer.model.model.config.sigma + 1,
                min_distance=1,
                exclude_border=False,
                batch_size=1,
                subpix=trainer.model.model.config.compute_flow,
            )
            pl_module.model.save(self._logdir, which="best", update_thresholds=True)
            log.info("Saved best model with optimized thresholds.")


class CustomEarlyStopping(pl.callbacks.early_stopping.EarlyStopping):
    """Callback implementing early stopping starting at a certain epoch."""

    def __init__(
        self,
        min_epochs: int,
        **kwargs,
    ):
        """

        Args:
            min_epochs (int): minimum number of epochs before early stopping is triggered.
            **kwargs: arguments to be passed to the parent class. See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
        """
        super().__init__(**kwargs)
        self._min_epochs = min_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # looks like pl calls this method after each epoch, so we just need to pass.
        pass

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Called when each validation epoch ends.

        Args:
            trainer (pl.Trainer): lightning trainer object.
            pl_module (pl.LightningModule): lightning module object.
        """
        if pl_module.current_epoch < self._min_epochs:
            pass
        else:
            self._run_early_stopping_check(trainer)
