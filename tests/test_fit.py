
import numpy as np
from spotiflow.utils import points_to_prob, estimate_params
from spotiflow.model import Spotiflow

def test_fit2d():
    np.random.seed(42)
    
    n_points=64
    points = np.random.randint(20,245-20, (n_points,2))
    sigmas = np.random.uniform(1, 5, n_points)
    
    x = points_to_prob(points, (256,256), sigma=sigmas, mode='sum')
    
    x += .2+0.05*np.random.normal(0, 1, x.shape)
    
    params = estimate_params(x, points)
    
    return x, sigmas, params

def test_fit3d():

    np.random.seed(42)
    ndim=3 
    
    n_points=64
    points = np.random.randint(20,128-20, (n_points,ndim))
    sigmas = np.random.uniform(1, 5, n_points)
    
    x = points_to_prob(points, (128,)*ndim, sigma=sigmas, mode='sum')
    
    x += .2+0.05*np.random.normal(0, 1, x.shape)
    
    params = estimate_params(x, points)
    return x, sigmas, params
        
if __name__ == "__main__":
    
    
    x, sigmas, params = test_fit3d()

    
    model = Spotiflow.from_pretrained("synth_3d")
    
    img = np.clip(200*x, 0,255).astype(np.uint8)
    
    points, details = model.predict(img, fit_params=True)