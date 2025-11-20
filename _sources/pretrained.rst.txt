Available pre-trained models
----------------------------

The following pre-trained models are available (for a more detailed description, please refer to the *Methods* section of the paper as well as *Supplementary Table 2*):

- ``general``: trained on a diverse dataset of spots of different modalities acquired in different microscopes with different settings. This model is the default one used in the CLI (pixel sizes: 0.04 µm, 0.1 µm, 0.11 µm, 0.15 µm, 0.32 µm, 0.34 µm).
- ``hybiss``: trained on HybISS data acquired in 3 different microscopes (pixel sizes: 0.15 µm, 0.32 µm, 0.34 µm).
- ``synth_complex``: trained on synthetic data, which includes simulations of aberrated spots and fluorescence background (pixel size: 0.1 µm).
- ``fluo_live``: trained on live-cell fluorescence imaging data corresponding to the ``Telomeres`` and ``Terra`` datasets in the manuscript (pixel size: 0.11 µm).
- ``synth_3d``: trained on synthetic 3D data, which includes simulations of aberrated spots and Z-related artifacts (voxel size: 0.2 µm).
- ``smfish_3d``: fine-tuned from the ``synth_3d`` model on smFISH 3D data of *Platynereis dumerilii* (voxel size: 0.13 µm (YX), 0.48 µm (Z)).

You can use these models to predict spots in images or to fine-tune them on a few annotations of your own data. The models can be loaded via the API as follows:

.. code-block:: python

    from spotiflow.model import Spotiflow

    pretrained_model_name = "general"
    model = Spotiflow.from_pretrained(pretrained_model_name)

You can also load them from the napari plugin or from the CLI by specifying the name of the model. See :ref:`napari:Predicting spots using the napari plugin` and :ref:`cli:Inference via CLI` for more information respectively.
