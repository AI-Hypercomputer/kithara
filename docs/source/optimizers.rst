.. _optimizers:

Optimizer Support Guide
=======================

Choosing an Optimizer
---------------------
Effective model fine-tuning relies heavily on the chosen optimization algorithm. Kithara offers flexibility by supporting optimizers from two widely-used libraries:

* **Keras Optimizers**: Standard, easy-to-use optimizers integrated directly into the Keras framework.
* **Optax Optimizers**: A powerful and composable optimization library designed for JAX (which Keras can leverage), offering fine-grained control over the gradient update process.

This guide explains how to utilize both types within the Kithara ``Trainer``.

Using Keras Optimizers
----------------------
You can directly use any optimizer available in the ``keras.optimizers`` module. Simply instantiate the desired optimizer with its configuration parameters and pass the instance to the ``Trainer``.

Here's how you can use the ``AdamW`` optimizer from Keras::

    import keras
    from Kithara import Trainer # Assuming Trainer is in Kithara namespace

    # Assume 'model' and 'train_dataloader' are already defined
    # Define hyperparameters
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    # Instantiate a Keras optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        # Add other Keras optimizer arguments as needed
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer, # Pass the Keras optimizer instance
        train_dataloader=train_dataloader,
        epochs=1,
        # ... other Trainer arguments (log_steps_interval, etc.)
    )

    # Start training
    trainer.train()

For a comprehensive list of available Keras optimizers and their specific arguments, consult the official `Keras Optimizers documentation <https://keras.io/api/optimizers/>`_.

Using Optax Optimizers
--------------------
Optax provides a highly flexible approach where optimizers are often built by chaining together multiple gradient transformations (like learning rate scheduling, gradient clipping, or weight decay) with a core optimizer function. This allows for complex optimization strategies.

To use Optax, import the ``optax`` library, construct your optimizer (potentially using ``optax.chain``), and pass the resulting object to the ``Trainer``.

This example demonstrates using ``optax.adamw`` combined with gradient clipping::

    import optax
    from Kithara import Trainer # Assuming Trainer is in Kithara namespace

    # Assume 'model' and 'train_dataloader' are already defined
    # Define hyperparameters
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    CLIP_NORM = 1.0

    # Create an Optax optimizer chain
    # Common practice: apply transformations like clipping before the main optimizer step
    optimizer = optax.chain(
        optax.clip(CLIP_NORM),
        optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        # Add other Optax transformations as needed
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer, # Pass the Optax optimizer instance
        train_dataloader=train_dataloader,
        epochs=1,
        # ... other Trainer arguments (log_steps_interval, etc.)
    )

    # Start training
    trainer.train()

Explore the extensive capabilities of Optax, including its various optimizers and gradient transformations, in the official `Optax API documentation <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_.

Key Takeaway
------------
Whether you choose a Keras or an Optax optimizer, the integration with Kithara is straightforward:

1.  Instantiate your chosen optimizer object with the desired configuration.
2.  Pass this optimizer object to the ``optimizer`` argument when initializing the ``Trainer``.

The ``Trainer`` will then handle the application of the optimizer during the training loop.
