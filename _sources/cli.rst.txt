Inference via CLI
-----------------

You can use the CLI to run inference on an image or folder containing several images.

To do that, simply use the following command:

.. code-block::

    spotiflow-predict PATH


where PATH can be either an image or a folder.

By default, the command will use the `general` pretrained model. You can specify a different model by using the `-\-pretrained-model` flag or the `-\-model-dir` flag for a model you trained yourself.

Moreover, spots are saved to a subfolder `spotiflow_results` created inside the input folder (this can be changed with the `-\-out-dir` flag).

For more information, please refer to the help message of the CLI:

.. code-block::

    spotiflow-predict -h
