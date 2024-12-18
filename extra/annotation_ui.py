"""
Starts a napari widget for annotating points on an image starting from LoG candidates or existing results stored in a CSV.

The widget allows to:
- toggle points visibility
- save points to a csv file
- detect points using LoG detector
- increase/decrease LoG threshold with w/e keys and a slider

Usage:
    python annotation_ui.py <path_to_image> [options]
"""
import numpy as np
import napari
from pathlib import Path
from tifffile import imread
from qtpy.QtWidgets import QMessageBox
from csbdeep.utils import normalize
from skimage.feature import blob_log
from magicgui import magicgui
import pandas as pd
import argparse
from spotiflow.utils import read_coords_csv
from spotiflow.utils.fitting import estimate_params


KEY_SHORTCUTS = {
    'toggle':  ('q', 'toggle points'),
    'save':    ('s' , 'save csv'),
    'thr_dec': ('w', 'decrease thr'),
    'thr_inc': ('e', 'increase thr'),
    'detect':  ('d', 'detect')
    }

def load_points(path):
    path = Path(path)
    if path.suffix==".npy":
        return np.load(args.points)
    elif path.suffix==".csv":
        return read_coords_csv(path)
    else:
        raise ValueError(f'not supported extension {path.suffix}')
        

def save_points(path, arr):
    path = Path(path)
    print(path)
    if path.suffix==".npy":
        np.save(path, arr)
    elif path.suffix==".csv":
        pd.DataFrame(arr, columns=['y','x']).to_csv(path, index=False)
    else:
        raise ValueError(f'not supported extension {path.suffix}')
    
def filter_points_bbox(points, bbox):
    """   bbox = ((y1,x1), (y2, x2))  """
    inds = np.bitwise_and(
        np.bitwise_and(points[:,0]>=bbox[0,0],points[:,0]<bbox[1,0]),
        np.bitwise_and(points[:,1]>=bbox[0,1],points[:,1]<bbox[1,1]))
    return inds



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("img", type=str)
    parser.add_argument('-p', "--points", type=str, default=None)
    parser.add_argument("-t", "--threshold", type=float, default=0.5)

    args = parser.parse_args()

    if args.points is None:
        args.points = Path(args.img).with_suffix('.csv')            

    img = normalize(imread(args.img), .1,99)

    try:
        p = load_points(args.points)
    except Exception as e:
        print(e)
        print('using log detector')
        p = blob_log(img, min_sigma=1, max_sigma=4, threshold=args.threshold,  exclude_border=1)[:,:2]
        print(p.shape)

    viewer = napari.Viewer()                        

    image_layer = viewer.add_image(img, name=Path(args.img).name)
    image_layer.contrast_limits = [0,3]
    points_layer = viewer.add_points(p, name='points', border_color='springgreen', border_width=.1, face_color=[0,0,0,0], symbol="o", opacity=.8, size=12)
    
    # overwrite existing annotation
    @viewer.bind_key(KEY_SHORTCUTS["save"][0])
    def f(event=None):
        res = QMessageBox().warning(
                    viewer.window.qt_viewer,
                    "Confirm overwrite",
                    f"Overwrite existing annotation {args.points.name} ?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No)
        if res == QMessageBox.Yes:
            save_points(args.points, points_layer.data)
            print(points_layer.data)
            print('saved')

    @viewer.bind_key(KEY_SHORTCUTS["toggle"][0], overwrite=True)
    def f(event=None):
        if points_layer.visible:
            points_layer._mymode = points_layer.mode 
            
        points_layer.visible = not points_layer.visible

        if points_layer.visible:
            points_layer.mode = points_layer._mymode

    def detect_and_populate():
        fov = image_layer.corner_pixels
        ss = slice(fov[0,0], fov[1,0]), slice(fov[0,1], fov[1,1])
        x = img[ss] 
        points = blob_log(x, min_sigma=1, max_sigma=parameters.max_sigma.value, threshold=parameters.threshold.value, exclude_border=1)[:,:2]
        points = points + fov[0]
        points0 = points_layer.data  
        idx = filter_points_bbox(points0, fov)
        points = np.concatenate((points0[~idx], points), axis=0)
        points_layer.data = points
        points_layer.selected_data = {}


    @magicgui(auto_call=True, max_sigma={"widget_type": "FloatSlider", "min" : 1, "max": 6}, 
                              threshold={"widget_type": "FloatSlider", "min" : 1e-5, "max": 1, 'step':1000},
                              fit_gaussian={"widget_type": "CheckBox", "label": "fit gaussian"})
    def parameters(threshold: float = args.threshold, max_sigma=4, fit_gaussian=False):
        if not fit_gaussian:
            detect_and_populate()
        else:
            points = points_layer.data
            x = img
            params = estimate_params(img=x, centers=points, refine_centers=True, verbose=False)
            offsets = np.vstack([params.offset_y, params.offset_x]).T
            print(f"Could not fit {(params.offset_y==np.nan).sum()} points (out of {len(params.offset_y)})")
            points = points + np.nan_to_num(offsets, 0.0)
            points_layer.data = points


    parameters.threshold._widget._readout_widget.setDecimals(3)
    viewer.window.add_dock_widget(parameters)


    @viewer.bind_key(KEY_SHORTCUTS["detect"][0], overwrite=True)
    def f(event=None):
        detect_and_populate()

    @viewer.bind_key(KEY_SHORTCUTS["thr_inc"][0], overwrite=True)
    def f(event=None):
        parameters.threshold.value *= 1.05
        
    @viewer.bind_key(KEY_SHORTCUTS["thr_dec"][0], overwrite=True)
    def f(event=None):
        parameters.threshold.value *= 1/1.05



    viewer.title = " ".join(f"[{v[0]}] {v[1]}   " for k,v in KEY_SHORTCUTS.items())
        
    napari.run()
