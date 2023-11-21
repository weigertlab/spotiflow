:hero: Spotipy: accurate and robust spot detection for fluorescence microscopy 

=======
Spotipy
=======

Spotipy is a learning-based spot detection method for fluorescence microscopy images. For more information, please refer to our `paper <https://random.dog/>`__.

Getting Started
---------------

Installation
~~~~~~~~~~~~


First, create and activate a new conda environment. 

.. code-block:: console

   (base) $ conda create -n spotipy python=3.9
   (base) $ conda activate spotipy

Then, install Pytorch using ``conda``/ ``mamba``. Please follow the `official instructions for your system <https://pytorch.org/get-started/locally>`__.

As an example, for MacOS:

.. code-block:: console

   (spotipy) $ conda install pytorch::pytorch torchvision -c pytorch

For a linux system with CUDA (note that you should change the CUDA version to match the one installed on your system):

.. code-block:: console

   (spotipy) $ conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

Finally, install ``spotipy`` using ``pip``:

.. code-block:: console

   (spotipy) $ pip install spotipy

Predicting spots in an image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The snippet below shows how to retrieve the spots from an image using one of the pretrained models:

.. code-block:: python

   from skimage.io import imread
   from spotipy_torch.model import Spotipy
   from spotipy_torch.utils import write_coords_csv
   

   # Load the desired image
   img = imread("/path/to/your/image") 
 
   # Load a pretrained model
   model = Spotipy.from_pretrained("general")

   # Predict spots
   spots, details = model.predict(img) # predict expects a numpy array

   # spots is a numpy array with shape (n_spots, 2)
   # details contains additional information about the prediction, like the predicted heatmap, the probability per spot, the flow field, etc.

   # Save the results to a CSV file
   write_coords_csv(spots, "/path/to/save/spots.csv")

If a custom model is used, simply change the model loadings step to:

.. code-block:: python

   # Load a custom model
   model = Spotipy.from_folder("/path/to/model")
   


Contents
--------

.. toctree::
   :maxdepth: 2

   train
   finetune
   api
