import torch 
from spotiflow.augmentations import Pipeline
from spotiflow.augmentations.transforms import FlipRot90, Rotation, Translation
from spotiflow.augmentations.transforms.utils import _generate_img_from_points

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    
    
    pts = torch.randint(5, 95, (10, 2)) 
    img = torch.tensor(_generate_img_from_points(pts.numpy(), (100, 100))).unsqueeze(0)
    
    pipeline = Pipeline(
        FlipRot90(), 
        Rotation(order=1), 
        Translation(shift=10)
        )
    

    plt.ion()
    fig, axs = plt.subplots(1, 4, figsize=(16, 8))
    
    def _to_rgb(img, pts):
        img2 = torch.tensor(_generate_img_from_points(pts.numpy(), (100, 100))).unsqueeze(0)
        return torch.stack((img2[0], img[0], img2[0]), -1).numpy()
    
    axs[0].imshow(_to_rgb(img, pts))
    axs[0].set_title('original')
            
    for ax in axs[1:]:
        img2, pts2 = pipeline(img, pts)
        ax.imshow(_to_rgb(img2, pts2))
        ax.set_title('augmented')
        
    for ax in axs.flatten(): 
        ax.axis('off')

    plt.show()