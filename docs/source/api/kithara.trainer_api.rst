.. _trainer_api:

Trainer
=======

.. py:class:: Trainer

   A base Trainer class supporting continued pretraining and SFT, designed to be subclassed for implementing other training objectives, e.g. DPO.

   :param model: The model to be trained or evaluated
   :type model: kithara.Model
   :param optimizer: The optimizer used for training
   :type optimizer: keras.Optimizer | optax.GradientTransformation | optax.GradientTransformationExtraArgs
   :param train_dataloader: A dataloader that provides training batches
   :type train_dataloader: kithara.Dataloader
   :param eval_dataloader: A dataloader that provides evaluation batches
   :type eval_dataloader: Optional[kithara.Dataloader]
   :param steps: The total number of training steps to execute. Defaults to None and trains 1 epoch
   :type steps: Optional[int]
   :param epochs: The total number of training epochs to execute. Defaults to None
   :type epochs: Optional[int]
   :param log_steps_interval: The interval between logging steps. Each log includes the current loss value and performance metrics
   :type log_steps_interval: int
   :param eval_steps_interval: The interval between evaluation steps. Only one of eval_steps_interval or eval_epochs_interval can be set
   :type eval_steps_interval: Optional[int]
   :param eval_epochs_interval: The interval between evaluation epochs. Only one of eval_steps_interval or eval_epochs_interval can be set
   :type eval_epochs_interval: Optional[int]
   :param max_eval_samples: The maximum number of samples to use during evaluation. Uses the entire evaluation dataset if not provided
   :type max_eval_samples: int
   :param tensorboard_dir: The directory path for TensorBoard logs. Can be either a local directory or a Google Cloud Storage path
   :type tensorboard_dir: Optional[str]
   :param profiler: A profiler instance for monitoring performance metrics
   :type profiler: Optional[kithara.Profiler]
   :param checkpointer: A checkpointer instance for saving model checkpoints
   :type checkpointer: Optional[kithara.Checkpointer]

   .. py:method:: train()

      Execute the main training loop. Handles epoch iteration, batch processing, loss computation, model updates, progress logging, and periodic evaluation.


Example usage:

.. code-block:: python

   trainer = Trainer(
       model=my_model,
       optimizer=keras.optimizers.Adam(learning_rate=1e-4),
       train_dataloader=train_loader,
       eval_dataloader=eval_loader,
       steps=1000,
       log_steps_interval=10,
       eval_steps_interval=100,
       tensorboard_dir="local_dir_or_gs_bucket",
       checkpointer= kithara.Checkpointer(
           save_dir="local_dir_or_gs_bucket",
           save_interval=100,
       ),
   )

   trainer.train()

Note
----
- If both ``steps`` and ``epochs`` are None, defaults to training for 1 epoch
- If ``eval_dataloader`` is provided but no evaluation interval is set, defaults to evaluating every epoch
- The trainer automatically handles data sharding for distributed training