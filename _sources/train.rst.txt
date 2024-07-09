Training a Spotiflow model on a custom dataset
----------------------------------------------

Data format
^^^^^^^^^^^

First of all, make sure that the data is organized in the following format:

::

    spots_data
    ├── train          
    │   ├── img_001.csv
    │   ├── img_001.tif
    |   ...
    │   ├── img_XYZ.csv
    |   └── img_XYZ.tif
    └── val          
        ├── val_img_001.csv
        ├── val_img_001.tif
        ...
        ├── val_img_XYZ.csv
        └── val_img_XYZ.tif

The actual naming of the files is not important, but the ``.csv`` and ``.tif`` files corresponding to the same image **must** have the same name! The ``.csv`` files must contain the spot coordinates in the following format:

.. code-block::

    y,x 
    42.3,24.24
    252.99, 307.97
    ...

The column names can also be `axis-0` (instead of `y`) and `axis-1` instead of `x`.


Basic training
^^^^^^^^^^^^^^

You can easily train a model using the default settings as follows and save it to the directory `/my/trained/model`:

.. code-block:: python

    from spotiflow.model import Spotiflow
    from spotiflow.utils import get_data

    # Get the data
    train_imgs, train_spots, val_imgs, val_spots = get_data("/path/to/spots_data")

    # Initialize the model
    model = Spotiflow()

    # Train and save the model
    model.fit(
        train_imgs,
        train_spots,
        val_imgs,
        val_spots,
        save_dir="/my/trained/model",
    )

You can then load it by simply calling:

.. code-block:: python

    model = Spotiflow.from_folder("/my/trained/model")

Customizing the training
^^^^^^^^^^^^^^^^^^^^^^^^

You can also pass other parameters relevant for training to the `fit` method. For example, you can change the number of epochs, the batch size, the learning rate, etc. You can do that using the `train_config` parameter. For more information on the arguments allowed, see the documentation of :py:func:`spotiflow.model.spotiflow.Spotiflow.fit` method as well as :py:mod:`spotiflow.model.config.SpotiflowTrainingConfig`. As an example, let's change the number of epochs and the learning rate:

.. code-block:: python
    
    train_config = {
        "num_epochs": 100,
        "learning_rate": 0.001,
        # other parameters
    }

    model.fit(
        train_imgs,
        train_spots,
        val_imgs,
        val_spots,
        save_dir="/my/trained/model",
        train_config=train_config,
        # other parameters
    )


In order to change the model architecture (`e.g.` number of input/output channels, number of layers, variance for the heatmap generation, etc.), you can create a :py:mod:`spotiflow.model.config.SpotiflowModelConfig` object and populate it accordingly. Then you can pass it to the `Spotiflow` constructor. For example, if our image is RGB and we need the network to use 3 input channels, we can do the following:

.. code-block:: python

    from spotiflow.model import SpotiflowModelConfig

    # Create the model config
    model_config = SpotiflowModelConfig(
        in_channels=3,
        # you can pass other arguments here
    )
    model = Spotiflow(model_config)
