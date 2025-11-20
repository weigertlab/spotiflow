Installation instructions
-------------------------

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