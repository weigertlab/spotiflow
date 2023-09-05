![Logo](artwork/spotipy_transp_small.png)
---

# Spotipy - accurate and efficient spot detection (Torch version)

## Installation
Create and activate a fresh conda environment:

```console
(base) foo@bar:~$ conda create -n spotipy-torch python=3.9
(base) foo@bar:~$ conda activate spotipy-torch
```

Then install PyTorch using conda/mamba (refer to [the official installation instructions](https://pytorch.org/get-started/locally/) for more info depending on your system):

For MacOS:
```console
(spotipy-torch) foo@bar:~$ conda install pytorch::pytorch torchvision -c pytorch # for MacOS
```

For linux with a CUDA device (one might need to change the cuda version accordingly):
```console
(spotipy-torch) foo@bar:~$ conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia # For linux with a CUDA device, might need to change the cuda version accordingly
```


Then, install tormenter (the augmentation library) by cloning it first:

```console
(spotipy-torch) foo@bar:~$ git clone git@github.com:weigertlab/tormenter.git
(spotipy-torch) foo@bar:~$ pip install -e ./tormenter
```

Finally, install `spotipy-torch`:

```console
(spotipy-torch) foo@bar:~$ git clone git@github.com:weigertlab/spotipy-torch.git
(spotipy-torch) foo@bar:~$ pip install -e ./spotipy-torch
```

## Usage

### Training
See `scripts/train.py` for an example of training script.

### Inference
```python
import tifffile
from spotipy_torch.model import Spotipy

# Load an image
img = tifffile.imread("path/to/image") # or any other image loading library according to the image format

# Load the model
model = Spotipy.from_pretrained("path/to/trained_model", inference=True)

# Predict
pred, details = model.predict(img) # Pred contains the detected spots, the attribute 'heatmap' of `details` contains the predicted heatmap (access it by `details.heatmap`)
```

## Napari plugin
See [spotipy-napari](https://github.com/weigertlab/napari-spotipy-torch) for the napari plugin.


## Tests

Install the `testing` extras first:

```console
(spotipy-torch) foo@bar:~$ pip install -e "./spotipy-torch[testing]"
```

Then `cd` into the cloned repository and run the tests:
```console
(spotipy-torch) foo@bar:~$ cd spotipy-torch
(spotipy-torch) foo@bar:~/spotipy-torch$ pytest -v --color=yes --cov=spotipy_torch
```
## TODO

- [x] Refactor datasets
- [x] Refactor `spotipy_torch/model` (model loading/saving, config classes, see if trainer/evaler can be done in a separate file, etc.)
- [ ] Add fast peak detection mode
- [x] Make prediction workable on images whose size is non-divisible by powers of 2
- [ ] First docs prototype
- [x] Tests

## Contributors

Albert Dominguez Mantes, Antonio Herrera, Irina Khven, Anjalie Schläppi, Gioele La Manno, Martin Weigert
