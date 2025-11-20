:hero: Spotiflow: accurate and robust spot detection for fluorescence microscopy 

=========
Spotiflow
=========

Spotiflow is a learning-based subpixel-accurate spot detection method for 2D and 3D fluorescence microscopy. It is primarily developed for spatial transcriptomics workflows that require transcript detection in large, multiplexed FISH-images, although it can also be used to detect spot-like structures in general fluorescence microscopy images and volumes. For more information, please refer to our `paper <https://doi.org/10.1101/2024.02.01.578426/>`__. 

Getting Started
---------------

Installation (pip)
~~~~~~~~~~~~~~~~~~


First, create and activate a fresh ``conda`` environment (we currently support Python 3.9 to 3.13). If you don't have ``conda`` installed, we recommend using `miniforge <https://github.com/conda-forge/miniforge>`__.

.. code-block:: console

   $ conda create -n spotiflow python=3.12
   $ conda activate spotiflow

Then, install PyTorch using ``pip``:

.. code-block:: console

   $ pip install torch

**Note (for Linux/Windows users with a CUDA-capable GPU)**: one might need to change the torch installation command depending on the CUDA version. Please refer to the `PyTorch website <https://pytorch.org/get-started/locally/>` for more information.

**Note (for Windows users):** if using Windows, if using Windows, please install the latest `Build Tools for Visual Studio <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022>`__ (make sure to select the C++ build tools during installation) before proceeding to install Spotiflow.

Finally, install ``spotiflow`` using ``pip``:

.. code-block:: console

   $ pip install spotiflow


Installation (conda)
~~~~~~~~~~~~~~~~~~~~

For Linux/MacOS users, you can also install Spotiflow using ``conda`` through the ``conda-forge``. For directly creating a fresh environment with Spotiflow named ``spotiflow``, you can use the following command:

.. code-block:: console

   $ conda create -n spotiflow -c conda-forge spotiflow python=3.12

Predicting spots in an image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python API
^^^^^^^^^^

The snippet below shows how to retrieve the spots from an image using one of the pretrained models:

.. code-block:: python

   from skimage.io import imread
   from spotiflow.model import Spotiflow
   from spotiflow.utils import write_coords_csv
   

   # Load the desired image
   img = imread("/path/to/your/image") 
 
   # Load a pretrained model
   model = Spotiflow.from_pretrained("general")

   # Predict spots
   spots, details = model.predict(img) # predict expects a numpy array

   # spots is a numpy array with shape (n_spots, 2)
   # details contains additional information about the prediction, like the predicted heatmap, the probability per spot, the flow field, etc.

   # Save the results to a CSV file
   write_coords_csv(spots, "/path/to/save/spots.csv")

If a custom model needs to be used, simply change the model loading step to:

.. code-block:: python

   # Load a custom model
   model = Spotiflow.from_folder("/path/to/model")

Command Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can use the CLI to run inference on an image or folder containing several images. To do that, you can use the following command

.. code-block:: console

   $ spotiflow-predict PATH

where ``PATH`` can be either point to an image file or to a folder containing different files. By default, the command will use the ``general`` pretrained model. You can specify a different model by using the ``--pretrained-model`` flag for the pre-trained models we offer or ``--model-dir`` if you want to use a custom model. After running, the detected spots are by default saved to a subfolder ``spotiflow_results`` created inside the input folder (this can be changed with the ``--out-dir`` flag). For more information, please refer to the help message of the CLI (simply run ``spotiflow-predict -h`` on your command line).

Napari plugin
^^^^^^^^^^^^^
Spotiflow also can be run easily in a graphical user interface as a `napari <https://napari.org/>`__ plugin. See :ref:`napari:Predicting spots using the napari plugin` for more information. 

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   napari
   cli
   pretrained
   train
   finetune
   api
