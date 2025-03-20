"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import time
import sys
import jax
from kithara.distributed.sharding.utils import (
    entire_tree_is_sharded,
    is_not_sharded_and_is_large,
    get_size_in_mb,
    get_size_in_gb,
)
from kithara.model import Model
from kithara.dataset import Dataloader
from kithara.callbacks import Profiler, Checkpointer
from kithara.distributed.sharding._data_sharding import DataSharding
from keras.src.backend.common import global_state
from typing import Any, Union, List, Tuple
import jax.tree_util as jtu
import numpy as np
from kithara.trainer import Trainer
import jax.numpy as jnp

class DPOTrainer(Trainer):
    """
    A Trainer class for training and evaluating Kithara models. This base class is designed to be
    subclassed for implementing custom training objectives.

    Attributes:
        model (kithara.Model): The model to be trained or evaluated.
        optimizer (keras.Optimizer): The optimizer used for training.
        train_dataloader (kithara.Dataloader): A dataloader that provides training batches.
        eval_dataloader (kithara.Dataloader, optional): A dataloader that provides evaluation batches.
            Defaults to None.
        steps (int, optional): The total number of training steps to execute, where each step processes
            one batch of data. Defaults to None and trains 1 epoch.
        epochs (int, optional): The total number of training epochs to execute. Defaults to None. If
            steps is also set to None, falls back to training for 1 epoch.
        log_steps_interval (int, optional): The interval between logging steps. Each log includes the
            current loss value and performance metrics. Defaults to 1.
        eval_steps_interval (int, optional): The interval between evaluation steps. Only one of
            eval_steps_interval or eval_epochs_interval can be set.
        eval_epochs_interval (int, optional): The interval between evaluation epochs. Only one of
            eval_steps_interval or eval_epochs_interval can be set.
        max_eval_samples (int, optional): The maximum number of samples to use during evaluation.
            Uses the entire evaluation dataset if not provided.
        tensorboard_dir (str, optional): The directory path for TensorBoard logs. Can be either a
            local directory or a Google Cloud Storage (GCS) path. Defaults to None.
        profiler (kithara.Profiler, optional): A profiler instance for monitoring performance metrics. Defaults to None.

    Methods:
        loss_fn: Returns a JAX-compatible callable that computes the loss value from logits and labels.
            Defaults to SparseCategoricalCrossentropy.
        train(): Executes the main training loop.
        evaluate(state=None): Performs evaluation using batches from eval_dataloader.
        generate(prompt, stop_token_ids="auto"): Generates text responses in inference mode.
        save_model(filepath): Saves model weights in HDF5 (.h5) format.
    """

    def __init__(
        self,
        model: Model,
        optimizer: keras.Optimizer,
        train_dataloader: Dataloader,
        eval_dataloader: Dataloader = None,
        steps=None,
        epochs=None,
        log_steps_interval=1,
        eval_steps_interval=None,
        eval_epochs_interval=None,
        max_eval_samples=sys.maxsize,
        tensorboard_dir=None,
        profiler: Profiler = None,
        checkpointer: Checkpointer = None,
        ref_model: "kithara.Model" = None,
        beta:float = 1.0,
        label_smoothing:float = 0.0
    ):
        super().__init__(
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            steps,
            epochs,
            log_steps_interval,
            eval_steps_interval,
            eval_epochs_interval,
            max_eval_samples,
            tensorboard_dir,
            profiler,
            checkpointer,
        )
        self.ref_model = ref_model
        self.beta = beta
        self.label_smoothing = label_smoothing

    @property
    def loss_fn(self):
        """Define the loss function for training and evaluation. This property
        is intended to be overriden with custom loss functions.

        Returns:
            A JAX callable that takes y_true and logits as input and returns the loss value.
        """
        def per_token_logps(logits, labels):
            """
            Compute the log probabilities of the given labels under the given logits.
            Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            """

            # TODO: there might be a cleaner syntax for the following code
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            indices = jnp.expand_dims(labels, axis=-1)
            per_token_logps = jnp.take_along_axis(log_probs, indices, axis=-1)
            per_token_logps = jnp.squeeze(per_token_logps, axis=-1)

            loss_mask = labels != self.train_dataloader.dataset.tokenizer.pad_token_id
            per_token_logps = per_token_logps * loss_mask

            return jnp.sum(per_token_logps, axis=-1)

        def dpo_loss(y, ref_logits, logits):
            all_logps_policy = per_token_logps(logits, y)  # [batch]
            all_logps_ref = per_token_logps(ref_logits, y)

            policy_chosen_logps = all_logps_policy[::2]
            policy_rejected_logps = all_logps_policy[1::2]

            ref_chosen_logps = all_logps_ref[::2]
            ref_rejected_logps = all_logps_ref[1::2]

            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps

            logits = pi_logratios - ref_logratios

            losses = (
                -jax.nn.log_sigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - jax.nn.log_sigmoid(-self.beta * logits) * self.label_smoothing
            )
            
            return losses.mean()
        
        return dpo_loss

    def compute_loss(self, trainable_variables, non_trainable_variables, x, y):
        """Compute model loss in a stateless manner. This function is intended
        to use together with jax.grad, i.e. grad_fn =
        jax.value_and_grad(compute_loss, has_aux=True)

        Args:
            trainable_variables: Model's trainable parameters, obtained with `model.trainable_variables`
            non_trainable_variables: Model's non-trainable parameters, obtained with `model.non_trainable_variables`
            x: Input data
            y: Target data

        Returns:
            tuple: (loss value, updated non-trainable variables)
        """
        logits, non_trainable_variables = self.model.stateless_call(
            trainable_variables, non_trainable_variables, x
        )
        # DO NOT SUBMIT
        ref_logits, _ = self.ref_model.stateless_call(
            trainable_variables, non_trainable_variables, x
        )

        loss = self.loss_fn(y, ref_logits, logits)

        return loss, non_trainable_variables

    @property
    def grad_fn(self):
        """Stateless function that returns the value and gradients from the
        `compute_loss` function."""

        return jax.value_and_grad(self.compute_loss, has_aux=True)

    def _train_step(self, state: Tuple[List[jax.Array]], data: dict):
        """Execute a single training step.

        Args:
            state: Current model state (trainable variables, non-trainable variables, optimizer variables)
            data: Batch of training data, a dictionary containing "x" (input) and "y" (target) entries.
            Input value is directly fed into the model, so it should be exact format expected by the model.
            Target value is used to compute the loss, and should be in the exact format expected by the loss function.

        Returns:
            tuple: (loss value, updated state)
        """
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
        ) = state
        x, y = data["x"], data["y"]
        (loss, non_trainable_variables), grads = self.grad_fn(
            trainable_variables, non_trainable_variables, x, y
        )
        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )
        return (
            loss,
            (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
            ),
        )

    def train(self):
        """Execute the main training loop.

        This method handles:
        - Epoch iteration
        - Batch processing
        - Loss computation
        - Model updates
        - Progress logging
        - Periodic evaluation
        """

        print("-> Start training...")
        print("The first training step will be slow due to JAX compilation.")

        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            optimizer_variables=True,
        )

        self.callbacks.on_train_begin()

        # Training loop
        while True:
            self.epoch_count += 1
            self.callbacks.on_epoch_begin(self.epoch_count)

            epoch_loss = 0
            batches_seen_in_epoch = 0

            # Process each batch in the epoch
            for batch_input in self.train_dataloader:
                if self.steps and self.step_count >= self.steps:
                    break
                self.step_count += 1

                start_time = time.time()
                self.callbacks.on_train_batch_begin(self.step_count)

                # Prepare and validate input
                batch_input = self._prepare_batch_input_for_training(batch_input)
                self._validate_sharding_correctness(batch_input, state)

                # Execute training step
                loss, state = self.train_step(state, batch_input)
                epoch_loss += loss
                batches_seen_in_epoch += 1

                self._update_model_with_state(state)

                # Wait for computation to complete for accurate step time
                jax.block_until_ready(loss)

                # Calculate training step statistics
                step_time = time.time() - start_time

                tokens_per_second_per_device = (
                    self.global_batch_size
                    * self.train_dataloader.dataset.max_seq_len
                    / (step_time * self.device_count)
                )

                samples_per_second = self.global_batch_size / step_time

                step_stats = {
                    "step": self.step_count,
                    "loss": round(float(loss), 3),
                    "step_time": round(step_time, 2),
                    "epoch": self.epoch_count,
                    "tokens_per_second_per_device": round(
                        tokens_per_second_per_device, 1
                    ),
                    "tokens_per_second": round(
                        tokens_per_second_per_device * self.device_count, 1
                    ),
                    "samples_per_second": round(samples_per_second, 2),
                    "train_steps_per_second": round(1 / step_time, 2),
                    "samples_seen": self.global_batch_size * self.step_count,
                    "learning_rate": self.optimizer.learning_rate.value,
                }

                # Log progress
                if (
                    self.step_count == 1
                    or self.step_count % self.log_steps_interval == 0
                ):
                    print(step_stats)

                self.callbacks.on_train_batch_end(self.step_count, step_stats)

                # Step based evaluation
                if (
                    (self.eval_dataloader is not None)
                    and (self.eval_steps_interval is not None)
                    and (self.step_count % self.eval_steps_interval == 0)
                ):
                    self.evaluate(state)

            # Compute epoch statistics
            # If no custom loss_fn is supplied, the default *step loss* calculates
            # the per-token loss (i.e. average of the loss from #non-padding tokens in batch).
            # The *epoch loss* is simply the average of the step losses. It is not the exact
            # per-token loss across the epoch, but rather a proxy.
            epoch_loss = epoch_loss / batches_seen_in_epoch
            self.callbacks.on_epoch_end(self.epoch_count, {"epoch_loss": epoch_loss})
            print(
                f"Train epoch {self.epoch_count} (epoch may be incompete) loss : {epoch_loss}"
            )

            # Epoch based evaluation
            if (
                (self.eval_dataloader is not None)
                and (self.eval_epochs_interval is not None)
                and (self.epoch_count % self.eval_epochs_interval == 0)
            ):
                self.evaluate(state)

            # Check termination conditions
            if self.steps and self.step_count >= self.steps:
                break
            if self.epochs and self.epoch_count >= self.epochs:
                break

        self.callbacks.on_train_end()

    def save_model(self, filepath):
        """Save model weights in .h5 format.

        Args:
            filepath (str): Path where model weights will be saved
        """
        self.model.save_weights(filepath)

    def _eval_step(self, state: Tuple[List[jax.Array]], data: dict):
        """Execute a single evaluation step.

        This method performs forward propagation without gradient computation
        to evaluate model performance on provided data.

        Args:
            state: Tuple containing (trainable_variables, non_trainable_variables, optimizer_state)
            data: Dictionary containing input data 'x' and target data 'y'.
            Data should be in the same format as expected by _train_step function.

        Returns:
            tuple: (logits, loss value)
        """
        (trainable_variables, non_trainable_variables, _) = state
        x, y = data["x"], data["y"]
        logits, non_trainable_variables = self.model.stateless_call(
            trainable_variables, non_trainable_variables, x, training=False
        )
        loss = self.loss_fn(y, logits)
        return logits, loss

    def evaluate(self, state=None):
        """Execute the evaluation loop on batches of data provided by the
        `eval_dataloader`.

        This method:
        1. Processes the evaluation dataset
        2. Computes model predictions and loss
        3. Tracks and reports evaluation metrics
        4. Handles callbacks for monitoring

        Args:
            state: Optional tuple of model state. If None, current model state is used.
            Contains (trainable_variables, non_trainable_variables, optimizer_variables)
        """

        if state is None:
            state = self._get_jax_state(
                trainable_variables=True,
                non_trainable_variables=True,
                optimizer_variables=True,
            )

        # Initialize evaluation
        self.callbacks.on_test_begin()
        eval_loss = 0
        eval_batches_seen = 0
        eval_start_time = time.time()
        # Process each batch in evaluation dataset
        for step_i, batch_input in enumerate(self.eval_dataloader):
            if (eval_batches_seen + 1) * self.global_batch_size > self.max_eval_samples:
                break

            start_time = time.time()
            # Prepare and shard input
            batch_input = self._prepare_batch_input_for_training(batch_input)
            self._validate_sharding_correctness(batch_input, state)

            # Eval step
            logits, loss = self.eval_step(state, batch_input)

            # Accumulate metrics
            eval_loss += loss
            eval_batches_seen += 1

            # Logging
            if (step_i + 1) % self.log_steps_interval == 0:

                jax.block_until_ready(loss)

                step_time = time.time() - start_time
                samples_per_second = self.global_batch_size / step_time

                tokens_per_second_per_device = (
                    self.global_batch_size
                    * self.train_dataloader.dataset.max_seq_len
                    / (step_time * self.device_count)
                )

                step_stats = {
                    "eval_loss": round(float(loss), 3),
                    "eval_step": step_i,
                    "step_time": round(step_time, 2),
                    "tokens_per_second_per_device": round(
                        tokens_per_second_per_device, 1
                    ),
                    "tokens_per_second": round(
                        tokens_per_second_per_device * self.device_count, 1
                    ),
                    "eval_samples_per_second": round(samples_per_second, 2),
                    "eval_steps_per_second": round(1 / step_time, 2),
                }

                print(step_stats)

        # Compute final metrics and report results
        eval_loss = eval_loss / eval_batches_seen
        eval_time = time.time() - eval_start_time

        tokens_per_second_per_device = (
            eval_batches_seen
            * self.global_batch_size
            * self.train_dataloader.dataset.max_seq_len
        ) / (eval_time * self.device_count)

        samples_per_second = eval_batches_seen * self.global_batch_size / eval_time

        self.callbacks.on_test_end(
            {
                "eval_loss": eval_loss,
                "eval_samples_seen": eval_batches_seen * self.global_batch_size,
                "eval_time": eval_time,
                "tokens_per_second_per_device": tokens_per_second_per_device,
                "tokens_per_second": tokens_per_second_per_device * self.device_count,
                "samples_per_second": samples_per_second,
                "eval_steps_per_second": eval_batches_seen / eval_time,
            }
        )

        print(f"Eval loss after {self.step_count} training steps: ", eval_loss)

        return eval_loss
