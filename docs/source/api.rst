API Reference
-------------

.. autoclass:: spotiflow.model.spotiflow.Spotiflow
    :members: from_pretrained, from_folder, predict, fit, save, optimize_threshold

.. autoclass:: spotiflow.model.config.SpotiflowModelConfig
    :members:

.. autoclass:: spotiflow.model.config.SpotiflowTrainingConfig
    :members:

.. autoclass:: spotiflow.data.spots.SpotsDataset
    :members:

    .. automethod:: __init__

.. autoclass:: spotiflow.data.spots3d.Spots3DDataset
    :members:

    .. automethod:: __init__

.. automodule:: spotiflow.utils
    :members: get_data, read_coords_csv, write_coords_csv, normalize

.. automodule:: spotiflow.sample_data
    :members:

.. autoclass:: spotiflow.starfish.SpotiflowDetector
    :members: run
