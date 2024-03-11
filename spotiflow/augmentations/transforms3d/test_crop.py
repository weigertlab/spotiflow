from spotiflow.augmentations.transforms3d import Crop3D

import torch


if __name__ == "__main__":
    img = torch.rand(1,1,32,32,32)
    pts = torch.Tensor([[1,16,16]])
    trf = Crop3D((8,8,8), probability=1.0, point_priority=1)

    img_c, pts_c = trf.apply(img, pts)
    print(img_c.shape, pts_c)
