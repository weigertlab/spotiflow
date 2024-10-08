Inference via CLI
-----------------

Command Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can use the CLI to run inference on an image or folder containing several images. To do that, you can use the following command

.. code-block:: console

   $ spotiflow predict --input PATH

where ``PATH`` can be either an image or a folder. By default, the command will use the ``general`` pretrained model. You can specify a different model by using the ``--pretrained-model`` flag. Moreover, spots are saved to a subfolder ``spotiflow_results`` created inside the input folder (this can be changed with the ``--out-dir`` flag). For more information, please refer to the help message of the CLI (``spotiflow-predict -h``).
