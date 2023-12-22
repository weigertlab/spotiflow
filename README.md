![Logo](artwork/spotiflow_logo.png)
---

# Spotiflow - accurate and efficient spot detection with stereographic flow

*Spotiflow* is a deep learning-based, threshold-agnostic, and subpixel-accurate spot detection method for fluorescence microscopy. It is primarily developed for spatial transcriptomics workflows that require transcript detection in large, multiplexed FISH-images, although it can also be used to detect spot-like structures in general fluorescence microscopy images.

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
pip install spotiflow
```

## Usage

### Training
See `scripts/train.py` or `notebooks/01_training.ipynb` for an example of training.

### Inference

The API allows detecting spots in a new image in a few lines of code! Please check `notebooks/02_inference.ipynb` for a more in-depth explanation/

```python
from spotiflow.model import Spotiflow
from spotiflow.sample_data import test_image_hybiss_2d

# Load sample image
img = test_image_hybiss_2d()
# Or any other image
img = tifffile.imread("myimage.tif")

# Load a pretrained model
model = Spotiflow.from_pretrained("general", inference=True)
# Or load your own trained model from folder
model = Spotiflow.from_folder("./mymodel", inference=True)

# Predict
points, details = model.predict(img) # points contains the coordinates of the detected spots, the attributes 'heatmap' and 'flow' of `details` contains the predicted full resolution heatmap and the prediction of the stereographic flow respectively (access them by `details.heatmap` or `details.flow`).
```

### Napari plugin
Our napari plugin allows detecting spots directly with an easy-to-use UI. See [napari-spotiflow](https://github.com/weigertlab/napari-spotiflow) for more information.


## For developers

We are open to contributions, and we indeed very much encourage them! Make sure that existing tests pass before submitting a PR, as well as adding new tests/updating the documentation accordingly for new features.

### Testing
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

### Docs

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
- [x] Adjust `SpotiflowModelConfig` default config (e.g. compute_flow=True, batch_norm=True)
- [ ] Register all models
- [ ] Register all datasets?
- [ ] Add example train notebook
- [x] Add example inference notebook
- [x] Improve docs (order, etc.)

## How to cite
If you use this code in your research, please cite [the Spotiflow paper](https://random.dog/):

```bibtex
@inproceedings{XYZ,
  author    = {},
  title     = {},
  ...
}
```
