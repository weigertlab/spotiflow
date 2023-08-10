from pathlib import Path
from skimage import img_as_float32
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import augmend
import tifffile
import numpy as np
import torch

from .. import utils
from .base import BaseDataset


class AnnotatedSpotsDataset(BaseDataset):
    """ Annotated Spots dataset """
    def __init__(self, path, downsample_factors=[1], sigma=1,
                 mode="max", augment_probability=1, use_gpu=False,
                 size=(256, 256), should_center_crop=False,
                 norm_percentiles=(1, 99.8)) -> None:
        
        files_in_path = list(Path(path).iterdir())
        self._image_files = sorted([str(Path(path)/f) for f in files_in_path if f.suffix in {".tif", ".tiff"}])
        annotation_files = sorted([str(Path(path)/f) for f in files_in_path if f.suffix == ".csv"])
        # Make sure every image has an annotation
        assert sum((img.split(".")[0] != csv.replace("_refined_r1", "").split(".")[0] for img, csv in zip(self._image_files, annotation_files))) == 0 #! Temporal hack for refined pts
        # Load centers in-memory as they should be lightweight

        centers = [utils.read_coords_csv(f).astype(np.int32) for f in annotation_files]
        images = [utils.normalize(img_as_float32(tifffile.imread(f)), *norm_percentiles) for f in tqdm(self._image_files, desc='Reading images')]
        super().__init__(
            images=images,
            centers=centers,
            downsample_factors=downsample_factors,
            sigma=sigma,
            mode=mode,
            augment_probability=augment_probability,
            use_gpu=use_gpu,
            size=size,
            should_center_crop=should_center_crop,
        )


    def _build_augmenter(self, augment_probability, use_gpu=False):
        if augment_probability < 1e-8:
            return None
        aug = augmend.Augmend()
        axes = (-2, -1)
        aug.add([augmend.FlipRot90(axis=axes), augmend.FlipRot90(axis=axes)], probability=augment_probability)
        aug.add([augmend.Rotate(order=1, axis=axes, mode="reflect"), augmend.Rotate(order=1, axis=axes, mode="reflect")], probability=augment_probability)
        # aug.add(2*[augmend.Elastic(axis=axes, amount=5, grid=5, order=0, use_gpu=use_gpu)], probability=augment_probability)        
        aug.add([augmend.IsotropicScale(amount=(.5, 2.), axis=axes), augmend.IsotropicScale(amount=(.5, 2.), axis=axes)], probability=augment_probability)
        # aug.add([augmend.GaussianBlur(axis=axes, amount=(0, 1.5)), augmend.Identity()], probability=augment_probability)
        aug.add([augmend.AdditiveNoise(sigma=(0, 0.05)), augmend.Identity()], probability=augment_probability)
        aug.add([augmend.IntensityScaleShift(scale=(0.5, 2.), shift=(-0.2, 0.2), axis=axes), augmend.Identity()], probability=augment_probability)

        return aug
    
    def get_ordered_image_filenames(self):
        return [Path(f).stem for f in self._image_files]


class UnannotatedSpotsDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        files_in_path = list(Path(path).iterdir())
        self._image_files = sorted([Path(path)/f for f in files_in_path if f.suffix in {".tif", ".tiff"}])

    def __len__(self):
        return len(self._image_files)
    
    def __getitem__(self, idx):
        # Load, cast and rescale image
        img = utils.normalize(img_as_float32(tifffile.imread(self._image_files[idx])), 1, 99.8)

        ret_obj = {"img": torch.from_numpy(img).unsqueeze(0)}
        return ret_obj
    
    def get_ordered_image_filenames(self):
        return [Path(f).stem for f in self._image_files]
