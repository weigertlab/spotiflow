![Logo](artwork/spotiflow_logo.png)
---

# Spotipy - accurate and efficient spot detection with stereographic flow

*Spotiflow* is a deep learning-based, threshold-agnostic, and subpixel-accurate spot detection method for fluorescence microscopy. It is primarly delevoped for spatial transcriptomics workflows that require transcript detection in large, multiplexed FISH-images.

![Overview](artwork/overview.png)

## Installation
Create and activate a fresh conda environment:

```console
conda create -n spotiflow python=3.9
conda activate spotiflow
```

Then install PyTorch using conda/mamba (refer to [the official installation instructions](https://pytorch.org/get-started/locally/) for more info depending on your system):

For MacOS:
```console
conda install pytorch::pytorch torchvision -c pytorch # for MacOS
```

For linux with a CUDA device (one might need to change the cuda version accordingly):
```console
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia # For linux with a CUDA device, might need to change the cuda version accordingly
```

Finally, install `spotiflow`:

```console
cd spotiflow
pip install -e .
```

## Usage

### Training
See `scripts/train.py` for an example of training script.

### Inference
```python
from spotiflow.model import Spotipy
from spotiflow.sample_data import test_image_hybiss_2d

# Load sample image
img = test_image_hybiss_2d()
# Or any other image
img = tifffile.imread("myimage.tif")

# Load a pretrained model
model = Spotiflow.from_pretrained("hybiss", inference=True)
# Or load a model from folder
model = Spotiflow.from_folder("./mymodel", inference=True)

# Predict
points, details = model.predict(img) # Pred contains the detected spots, the attribute 'heatmap' of `details` contains the predicted heatmap (access it by `details.heatmap`)
```

## Napari plugin
See [spotipy-napari](https://github.com/weigertlab/napari-spotiflow) for the napari plugin.


## Tests

First, clone the repository:
```console
git clone git@github.com:weigertlab/spotiflow.git
```

Then install the `testing` extras:

```console
cd spotiflow
pip install -e ".[testing]"
```

then run the tests:

```console
pytest -v --color=yes --cov=spotiflow
```

## Docs

Install the `docs` extras:

```console
pip install -e ".[docs]"
```

and then `cd` into the `docs` folder of the cloned repository and build them:
```console
cd spotiflow/docs
sphinx-build -M html source build
```

## TODO

- [x] Refactor datasets
- [x] Refactor `spotiflow/model` (model loading/saving, config classes, see if trainer/evaler can be done in a separate file, etc.)
- [x] Add fast peak detection mode
- [x] Make prediction workable on images whose size is non-divisible by powers of 2
- [x] First docs prototype
- [x] Tests
- [x] Adjust `SpotipyModelConfig` default config (e.g. compute flow=True, batch_norm=True)
- [ ] Register all models
- [ ] Register all datasets?
- [ ] Add example notebooks (train, inference)
- [ ] Improve docs (order, etc.)

## Contributors

Albert Dominguez Mantes, Antonio Herrera, Irina Khven, Anjalie Schläppi, Gioele La Manno, Martin Weigert
