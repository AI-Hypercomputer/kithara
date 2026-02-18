.. _troubleshooting:

Troubleshooting
===============

1. Disk OOM when loading HF model checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

First try emptying your cache by running the following code on your Ray Cluster.

.. code-block:: python

    import shutil
    shutil.rmtree("/home/ubuntu/.cache/huggingface/hub/", ignore_errors=True)
    shutil.rmtree("/home/ubuntu/.keras/models", ignore_errors=True)

If you are using a single VM, the path may be different.

.. code-block:: python

    import shutil
    shutil.rmtree("~/.cache/huggingface/hub/", ignore_errors=True)
    shutil.rmtree("~/.keras/models", ignore_errors=True)

If emptying the cache still doesn't help, try attaching a disk to your VM and change HF cache directory using the environment variable ``export HF_HOME=<your_new_cache_dir>``.

You may have to copy your HF token to this new cache directory with ``cp .cache/huggingface/token <your_new_cache_dir>/token``.

2. Permission denied error when uploading checkpoint to GCS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

First make sure that your bucket is in the same project as your TPU VM. 

Otherwise, verify your current authentication:

.. code-block:: bash

    gcloud auth list
    gcloud storage ls gs://your_bucket

For your Python code, you likely need to ensure you're using the same credentials.

.. code-block:: bash

    gcloud auth application-default login

3. jaxlib.xla_extension.XlaRuntimeError errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try uninstall and reinstalling ``jax`` and ``jaxlib``

.. code-block:: bash

    pip uninstall jax jaxlib
    pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install libtpu-nightly==0.1.dev20250128+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

4.  Unable to initialize backend 'tpu': INTERNAL: Failed to get global TPU topology.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try adding ``JAX_PLATFORMS=''`` to your environment variables.