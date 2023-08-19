from csbdeep.internals.predict import tile_iterator
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from types import SimpleNamespace
from typing import Literal, Optional, Tuple


import json
import logging
import matplotlib.cm as cm
import torch
import torch.nn as nn
import wandb

import numpy as np

from .backbones import ResNetBackbone, UNetBackbone
from .bg_remover import BackgroundRemover
from .fpn import FeaturePyramidNetwork
from .multihead import MultiHeadProcessor
from .losses import AdaptiveWingLoss
from ..utils import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

"""
class BaseConfig...


class BaseBackBone(nn.Module):
    def __init__(self, config: BaseConfig ) -> None:
        super().__init__(config)
    @abstractmethod
    def get_downsample_factors(self) -> Tuple[Tuple[int]]:
        pass
    @abstractmethod
    def get_out_channels(self) -> Tuple[int]:
        pass

class ResNetBackbone(BaseBackBone, nn.Module):
"""

class Spotipy(nn.Module):
    """Supervised spot detector using a multi-stage neural network as a backbone for 
       feature extraction followed by a Feature Pyramid Network (Lin et al., CVPR '17)
       module to allow loss computation and optimization at different resolution levels.
    """
    def __init__(self, backbone_str: str="resnet", backbone_params: dict={}, levels: int=3, pretrained_path: Optional[str]=None,
                 mode: Literal["direct", "fpn"]="fpn", background_remover: bool=True, device: str="cpu", which: str="best", inference_mode: bool=True, **kwargs: dict):
        super().__init__()
        assert mode in {"direct", "fpn"}, "Mode must be either 'direct' or 'fpn'."
        self._device = device
        if pretrained_path is not None:
            print("Pretrained path given. Loading model...")
            self._load_model(pretrained_path, which, inference_mode)
        else:
            self._backbone_str = backbone_str
            self._backbone_params = backbone_params
            self._backbone_params.update(
                {"in_channels": self._backbone_params.get("in_channels", 1),
                 "initial_fmaps": self._backbone_params.get("initial_fmaps", 32),
                 "downsample_factors": self._backbone_params.get("downsample_factors", tuple((2, 2) for _ in range(levels))),
                 "kernel_sizes": self._backbone_params.get("kernel_sizes", tuple((3, 3) for _ in range(levels))),
                 })
            self._levels = levels
            self._pretrained_path = pretrained_path
            self._mode = mode
            self._background_remover = background_remover

            # Build background remover
            self._bg_remover = BackgroundRemover(device=self._device) if background_remover else None

            # Build backbone
            self._backbone = self._backbone_switcher()

            # Build postprocessing modules
            if mode == "direct":
                self._post = MultiHeadProcessor(in_channels_list=self._backbone.out_channels_list,
                                                out_channels=1,
                                                kernel_sizes=self._backbone_params["kernel_sizes"],
                                                initial_fmaps=self._backbone_params["initial_fmaps"])
            elif mode == "fpn":
                self._post = FeaturePyramidNetwork(in_channels_list=self._backbone.out_channels_list, out_channels=1)
            else:
                raise NotImplementedError(f"Mode {mode} not implemented.")

            self._sigmoid = nn.Sigmoid()
            self.to(torch.device(self._device))

            self._optimizer = torch.optim.AdamW((p for p in self.parameters() if p.requires_grad))
            self._prob_thresh = 0.5

    def _backbone_switcher(self) -> nn.Module:
        """Switcher function to build the backbone.
        """
        if self._backbone_str == "unet":
            return UNetBackbone(**self._backbone_params)
        if self._backbone_str == "resnet":
            return ResNetBackbone(**self._backbone_params)
        else:
            raise NotImplementedError(f"Backbone {self._backbone_str} not implemented.")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward pass

        Args:
            x (torch.Tensor): input image

        Returns:
            Tuple[torch.Tensor]: iterable of results at different resolutions. Highest resolution first.
        """
        if self._bg_remover is not None:
            x = self._bg_remover(x)
        res = self._backbone(x)
        res = self._post(res)
        return tuple(res)

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
        return tuple(self._wrap_loss(loss_cls(reduction="none", **loss_kwargs), pos_weight if level==0 else 0) for level in range(self._levels))

    def fit(self, train_ds: torch.utils.data.Dataset, val_ds: torch.utils.data.Dataset, params: dict) -> dict:
        if len(params["wandb_user"])>0 and not params["skip_logging"]:
            utils.initialize_wandb(params, train_ds, val_ds)
            if wandb.run is not None:
                wandb.watch(self, log_freq=50, log_graph=True)
        num_epochs = params["num_epochs"]
        learning_rate = params["lr"]

        loss_f_str = params["loss"]
        pos_weight = params["pos_weight"]

        loss_funcs = self._loss_switcher(loss_f_str, pos_weight)
        batch_size = params["batch_size"]

        save_dir = Path(params["save_dir"])/params["run_name"]
        save_dir.mkdir(exist_ok=True, parents=True)

        device = torch.device(self._device)

        # Set LR
        for g in self._optimizer.param_groups:
            g['lr'] = learning_rate
        
        dataloader_kwargs = {"num_workers": 0, "pin_memory": True} # Avoid MP for now
        train_dataloader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
        valid_dataloader = DataLoader(dataset=val_ds, batch_size=1, shuffle=False, **dataloader_kwargs)

        scheduler = ReduceLROnPlateau(self._optimizer,
                                      factor=0.5,
                                      patience=10,
                                      threshold=1e-4,
                                      min_lr=3e-6,
                                      cooldown=5,
                                      verbose=True)

        history = {"train_loss": [], "valid_loss": []}
        best_val_loss = float("inf")
        log.info("Training...")
        log.info(f"Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        for epoch in tqdm(range(num_epochs)):
            self.train()
            tr_epoch_loss, val_epoch_loss = 0, 0
            for tr_batch in (progbar := tqdm(train_dataloader)):
                heatmap_lvs = [tr_batch[f"heatmap_lv{i}"] for i in range(self._levels)]
                self.zero_grad()
                imgs = tr_batch["img"].to(device)
                out = self(imgs)
                losses = tuple(loss_f(out[lv], heatmap_lvs[lv].to(device))/4**lv for lv, loss_f in zip(range(self._levels), loss_funcs))
                loss = sum(losses)
                loss.backward()
                self._optimizer.step()
                tr_epoch_loss += loss.item()
                losses_str = "  ".join(f"L{i} {l:.3f}" for i,l in enumerate(losses))
                progbar.set_description(f'epoch: {epoch+1} | loss: {loss.item():.8f} | {losses_str}')
                del out, loss, losses, imgs, tr_batch
            history["train_loss"].append(tr_epoch_loss/len(train_ds))
            if self._device == "cuda":
                torch.cuda.empty_cache()

            self.eval()
            val_preds = []
            with torch.inference_mode():
                for val_batch in valid_dataloader:
                    heatmap_lvs = [val_batch[f"heatmap_lv{i}"] for i in range(self._levels)]
                    imgs = val_batch["img"].to(device)
                    out = self(imgs)
                    losses = tuple(loss_f(out[lv], heatmap_lvs[lv].to(device))/4**lv for lv, loss_f in zip(range(self._levels), loss_funcs))
                    loss = sum(losses)
                    val_epoch_loss += loss.item()
                    high_lv_preds = self._sigmoid(out[0].squeeze(1)).detach().cpu().numpy()
                    # TODO: code properly
                    for batch_elem in range(high_lv_preds.shape[0]):
                        val_preds += [high_lv_preds[batch_elem]]
                    del out, loss, losses, imgs, val_batch
            if self._device == "cuda":
                torch.cuda.empty_cache()
            avg_val_loss = val_epoch_loss/len(val_ds)
            scheduler.step(avg_val_loss)
            history["valid_loss"].append(avg_val_loss)

            val_gt_centers = val_ds.get_centers()
            val_pred_centers = [utils.prob_to_points(p, exclude_border=False, min_distance=1) for p in val_preds]
            stats = utils.points_matching_dataset(val_gt_centers, val_pred_centers, cutoff_distance=3, by_image=True)

            val_f1, val_acc = stats.f1, stats.accuracy
            log.info(f"Epoch {epoch+1} | Train loss: {history['train_loss'][-1]:.8f}")
            log.info(f"Epoch {epoch+1} | Validation loss: {history['valid_loss'][-1]:.8f} | Validation F1: {val_f1:.3f}")

            if wandb.run is not None:  
                wandb.log({
                    "Training loss": history["train_loss"][-1],
                    "Validation loss": history["valid_loss"][-1],
                    "Learning rate": self._optimizer.param_groups[0]['lr'],
                    "Validation accuracy": val_acc,
                    "Validation F1-Score": val_f1,
                }, step=epoch)
                if epoch == 0:
                    # Log validation image and GT heatmap only in first epoch
                    img4wandb_img = wandb.Image(val_ds[0]["img"].numpy(), caption="Raw image (normalized)")
                    wandb.log({"Validation image": img4wandb_img}, step=epoch)
                    img4wandb_gt = Image.fromarray((255*cm.magma(val_ds[0]["heatmap_lv0"].numpy()[0])).astype(np.uint8))
                    img4wandb_gt = wandb.Image(img4wandb_gt, caption="Ground truth heatmap")
                    wandb.log({"Ground truth": img4wandb_gt}, step=epoch)
                img4wandb_hm = wandb.Image(
                    Image.fromarray((255*cm.magma(val_preds[0])).astype(np.uint8)),
                    caption="Prediction (full resolution)")
                wandb.log({"Prediction": img4wandb_hm}, step=epoch)

            if (last_val_loss := history["valid_loss"][-1]) < best_val_loss:
                best_val_loss = last_val_loss
                self._save_model(save_dir, epoch=epoch, which="best")

        self.optimize_threshold(val_ds)
        self._save_model(save_dir, epoch=epoch, which="last")
        return history
        
    def _save_model(self, save_path: str, which: Literal["best", "last"], epoch: Optional[int]=None, only_config: bool=False) -> None:
        checkpoint_path = Path(save_path)
        if not only_config:
            torch.save({
                "epoch": epoch+1 if epoch is not None else 0,
                "model_state": self.state_dict(),
                "optimizer_state": self._optimizer.state_dict(),
            }, str(checkpoint_path/f"{which}.pt"))
        with open(str(checkpoint_path/"config.json"), "w") as fb:
            json.dump({"backbone_str": self._backbone_str,
                        "backbone_params": self._backbone_params,
                        "levels": self._levels,
                        "mode": self._mode,
                        "background_remover": self._background_remover,
                        "prob_thresh": self._prob_thresh,
                        }, fb)
        return
    

    def _load_model(self, path: str, which: Literal["best", "last"]="best", inference_mode: bool=True) -> None:
        # ! TODO: add device mapping
        config_path = Path(path)/"config.json"
        with open(config_path, "r") as fb:
            config = json.load(fb)
        self.__init__(**config, device=self._device, pretrained_path=None) # To avoid infinite recursion in __init__
        states_path = Path(path)/f"{which}.pt"
        checkpoint = torch.load(states_path, map_location=self._device)
        self.load_state_dict(checkpoint["model_state"])
        self._prob_thresh = config.get("prob_thresh", 0.5)
        if inference_mode:
            self.eval()
            return
        self._optimizer.load_state_dict(checkpoint["optimizer_state"])
        return

    def predict(self, img: np.ndarray, prob_thresh: Optional[float]=None,
                n_tiles: Tuple[int, int]=(1,1), min_distance: int=1, exclude_border: bool=False,
                scale: Optional[int]=None, peak_mode: str="skimage", normalizer: Optional[callable]=None, verbose: bool=True):
        assert img.ndim == 2, "Image must be 2D (Y,X)"
        device = torch.device(self._device)
        # Add B and C dimensions
        if verbose:
            log.info(f"Predicting with prob_thresh = {self._prob_thresh if prob_thresh is None else prob_thresh:.3f}, min_distance = {min_distance}")

        if scale is None or scale == 1:
            x = img
        else:
            if verbose:
                log.info(f"Scaling image by factor {scale}")

            x = zoom(img, (scale, scale), order=1)

        self.eval()
        # Predict without tiling
        if all(n <= 1 for n in n_tiles):
            if normalizer is not None and callable(normalizer):
                x = normalizer(x)
            with torch.inference_mode():
                img_t = torch.from_numpy(x).to(torch.device(device)).unsqueeze(0).unsqueeze(0) # Add B and C dimensions
                y = self._sigmoid(self(img_t)[0].squeeze(0).squeeze(0)).detach().cpu().numpy()
            if scale is not None and scale != 1:
                y = zoom(y, (1./scale, 1./scale), order=1)
            pts = utils.prob_to_points(y, prob_thresh=self._prob_thresh if prob_thresh is None else prob_thresh, exclude_border=exclude_border, min_distance=min_distance)
            probs = y[tuple(pts.astype(int).T)].tolist()
        else: # Predict with tiling
            y = np.empty(x.shape, np.float32)
            points = []
            probs = []
            iter_tiles = tile_iterator(
                x,
                n_tiles=n_tiles,
                block_sizes=tuple(self._backbone_params["downsample_factors"][0][0]**self._levels for _ in range(img.ndim)),
                n_block_overlaps=(4,4),
            )
            if verbose:
                iter_tiles = tqdm(iter_tiles, desc="Predicting tiles", total=np.prod(n_tiles))
            for tile, s_src, s_dst in iter_tiles:
                assert all(s%t == 0 for s, t in zip(tile.shape, n_tiles)), "Currently, tile shape must be divisible by n_tiles"
                if normalizer is not None and callable(normalizer):
                    tile = normalizer(tile)
                with torch.inference_mode():
                    img_t = torch.from_numpy(tile).to(torch.device(device)).unsqueeze(0).unsqueeze(0) # Add B and C dimensions
                    y_tile = self._sigmoid(self(img_t)[0].squeeze(0).squeeze(0)).detach().cpu().numpy()
                    p = utils.prob_to_points(y_tile, prob_thresh=self._prob_thresh if prob_thresh is None else prob_thresh, exclude_border=exclude_border, min_distance=min_distance)
                    # remove global offset
                    p -= np.array([s.start for s in s_src[:2]])[None]
                    write_shape = tuple(s.stop-s.start for s in s_dst[:2])
                    p = utils._filter_shape(p, write_shape, idxr_array=p)

                    y_tile_sub = y_tile[s_src[:2]]

                    probs += y_tile_sub[tuple(p.astype(int).T)].tolist()
                    
                    # add global offset
                    p += np.array([s.start for s in s_dst[:2]])[None]
                    points.append(p)
                    y[s_dst[:2]] = y_tile_sub
            if scale is not None and scale != 1:
                y = zoom(y, (1./scale, 1./scale), order=1)
            
            points = np.concatenate(points, axis=0)

            probs = np.array(probs)
            if scale is not None and scale != 1:
                points = np.round((points.astype(float) / scale)).astype(int)
            
            probs = utils._filter_shape(probs, img.shape, idxr_array=points)
            pts = utils._filter_shape(points, img.shape, idxr_array=points)

        if verbose:
            log.info(f"Found {len(pts)} spots.")
        details = SimpleNamespace(prob=probs, heatmap=y)
        return pts, details

    def predict_dataset(self, ds: torch.utils.data.Dataset, min_distance=1, exclude_border=False, batch_size=4, prob_thresh=None, return_heatmaps=False):
        preds = []
        dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        device = torch.device(self._device)
        log.info(f"Predicting with prob_thresh = {self._prob_thresh if prob_thresh is None else prob_thresh}, min_distance = {min_distance}")
        self.eval()
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Predicting"):
                imgs = batch["img"].to(device)
                out = self(imgs)
                high_lv_preds = self._sigmoid(out[0].squeeze(1)).detach().cpu().numpy()
                for batch_elem in range(high_lv_preds.shape[0]):
                    preds += [high_lv_preds[batch_elem]]
                del out, imgs, batch
        if return_heatmaps:
            return preds
        p = [utils.prob_to_points(pred, prob_thresh=self._prob_thresh if prob_thresh is None else prob_thresh, exclude_border=exclude_border, min_distance=min_distance) for pred in preds]
        return p

    def optimize_threshold(self, val_ds: torch.utils.data.Dataset, cutoff_distance=3, min_distance=2, exclude_border=False, niter=11, batch_size=2):
        val_preds = []
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        device = torch.device(self._device)
        self.eval()
        with torch.inference_mode():
            for val_batch in val_dataloader:
                imgs = val_batch["img"].to(device)
                out = self(imgs)
                high_lv_preds = self._sigmoid(out[0].squeeze(1)).detach().cpu().numpy()
                for batch_elem in range(high_lv_preds.shape[0]):
                    val_preds += [high_lv_preds[batch_elem]]
                del out, imgs, val_batch

        val_gt_pts = val_ds.get_centers()
    
        def _metric_at_threshold(thr):
            val_pred_pts = [utils.prob_to_points(p, prob_thresh=thr, exclude_border=exclude_border, min_distance=min_distance) for p in val_preds]
            stats = utils.points_matching_dataset(val_gt_pts, val_pred_pts, cutoff_distance=cutoff_distance, by_image=True)
            return stats.f1
    
        def _grid_search(tmin, tmax):
            thr = np.linspace(tmin, tmax, niter)
            ys = tuple(_metric_at_threshold(t) for t in thr)
            i = np.argmax(ys)
            i1, i2 = max(0,i-1), min(i+1, len(thr)-1)
            return thr[i], (thr[i1],thr[i2]), ys[i]
        
        _, t_bounds, _ = _grid_search(0.3, 0.7)
        best_thr, _, best_f1 = _grid_search(*t_bounds)
        print(f"Best threshold: {best_thr:.3f}")
        print(f"Best F1-score: {best_f1:.3f}")
        self._prob_thresh = best_thr
        return
        