Fine-tuning a Spotiflow model on a custom dataset
-------------------------------------------------

Data format
^^^^^^^^^^^

See :ref:`train:Data format`.

Fine-tuning
^^^^^^^^^^^

Finetuning a pre-trained model on a custom dataset is very easy. You can load the model very similarly to what you would normally do to predict on new images (you only need to add one extra parameter!):

.. code-block:: python

    from spotiflow.model import Spotiflow
    from spotiflow.utils import get_data

    # Get the data
    train_imgs, train_spots, val_imgs, val_spots = get_data("/path/to/spots_data")

    # Initialize the model
    model = Spotiflow.from_pretrained(
        "general",
        inference_mode=False,
    )

    # Train and save the model
    model.fit(
        train_imgs,
        train_spots,
        val_imgs,
        val_spots,
        save_dir="/my/trained/model",
    )

Of course, you can also fine-tune from a model you have trained before. In that case, use the ``from_folder()`` method instead of ``from_pretrained()`` (see :ref:`index:Predicting spots in an image`).
All the information about training customization from :ref:`train:Customizing the training` applies here as well. However, note that you cannot change the model architecture when fine-tuning!