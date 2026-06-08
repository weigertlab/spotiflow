Inference via CLI
-----------------

Command Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can use the CLI to run inference on an image or folder containing several images. To do that, you can use the following command

.. code-block:: console

   $ spotiflow-predict PATH

where ``PATH`` can be either point to an image file or to a folder containing different files. By default, the command will use the ``general`` pretrained model. You can specify a different model by using the ``--pretrained-model`` flag for the pre-trained models we offer or ``--model-dir`` if you want to use a custom model. After running, the detected spots are by default saved to a subfolder ``spotiflow_results`` created inside the input folder (this can be changed with the ``--out-dir`` flag). For more information, please refer to the help message of the CLI (simply run ``spotiflow-predict -h`` on your command line).
