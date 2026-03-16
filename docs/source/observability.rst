.. _observability:

Observability
=============

Kithara supports Tensorboard and Weights and Biases for observability.

Tensorboard
-----------

To use Tensorboard, simply specify the ``tensorboard_dir`` arg in the ``Trainer`` class to a local directory or a Google Cloud Storage bucket.

To track training and evaluation performance, launch the tensorboard server with::

    tensorboard --logdir=your_tensorboard_dir

Weights and Biases
-----------

To use Weights and Biases, import the class Settings from wandb and pass it to ``wandb_settings`` arg in the ``Trainer`` class. For example:

```
from wandb import Settings

Trainer(...,wandb_settings=Settings(project="Project name"))
```

When running the Trainer, you will be prompt to provide an API key:

```
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
```

Alternatively you can export the key as an environment variable before running the trainer:

```
WANDB_API_KEY=$YOUR_API_KEY
```

After providing the key, you will be able to see your results at https://wandb.ai/<user>/<project>