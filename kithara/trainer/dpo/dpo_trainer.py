import os

os.environ["KERAS_BACKEND"] = "jax"

import sys
import time
from dataclasses import dataclass
from typing import Any, Union
import jax
import ray
import keras
import kithara
from kithara.model.model import (
    ModelConfig,
    OptimizerConfig,
)
from kithara.callbacks import Profiler, Checkpointer
from kithara.trainer.validation_mixin import ValidationMixin
from kithara.trainer.dpo.dpo_policy_model import DPOPolicyModel
from kithara.trainer.dpo.dpo_reference_model import DPOReferenceModel


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization (DPO) training."""

    policy_model: Union[ModelConfig, kithara.Model]
    optimizer: Union[
        OptimizerConfig,
        keras.Optimizer,
        "optax.GradientTransformation",
        "optax.GradientTransformationExtraArgs",
    ]
    beta: float = 0.1
    label_smoothing: float = 0.0
    run_mpmd: bool = False


class DPOTrainer(ValidationMixin):
    """Direct Preference Optimization (DPO) Trainer.

    Can run in either MPMD (multiple program, multiple data) mode on CPU with tasks dispatched to TPUs,
    or SPMD (single program, multiple data) mode directly on TPU.
    """

    def __init__(
        self,
        dpo_config: DPOConfig,
        train_dataloader: kithara.Dataloader,
        eval_dataloader: kithara.Dataloader = None,
        steps: int = 100,
        epochs=None,
        log_steps_interval=1,
        eval_steps_interval=None,
        eval_epochs_interval=None,
        max_eval_samples=sys.maxsize,
        tensorboard_dir=None,
        profiler: Profiler = None,
        checkpointer: Checkpointer = None,
    ):
        """Initialize the DPO trainer.

        Args:
            dpo_config: Configuration for DPO training
            train_dataloader: Dataloader for training data
            eval_dataloader: Dataloader for evaluation data (optional)
            steps: Number of training steps to run (either steps or epochs must be specified)
            epochs: Number of training epochs to run
            log_steps_interval: How often to log training progress (in steps)
            eval_steps_interval: How often to run evaluation (in steps)
            eval_epochs_interval: How often to run evaluation (in epochs)
            max_eval_samples: Maximum number of samples to use for evaluation
            tensorboard_dir: Directory for TensorBoard logs
            profiler: Profiler callback
            checkpointer: Checkpointer callback
        """
        # Handle default values
        if steps is None and epochs is None:
            epochs = 1
        if (
            (eval_dataloader is not None)
            and (eval_steps_interval is None)
            and (eval_epochs_interval is None)
        ):
            eval_epochs_interval = 1

        # Core components
        self.dpo_config = dpo_config
        self.policy_model, self.ref_model = self.init_models()
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.run_mpmd = dpo_config.run_mpmd

        # Training parameters
        self.steps = steps
        self.epochs = epochs
        self.step_count = 0
        self.epoch_count = 0
        self.eval_steps_interval = eval_steps_interval
        self.eval_epochs_interval = eval_epochs_interval
        self.max_eval_samples = max_eval_samples
        self.log_steps_interval = log_steps_interval
        self.global_batch_size = train_dataloader.global_batch_size
        self.device_count = jax.device_count()
        self._validate_setup()

        # Callbacks
        self.profiler = profiler
        self.checkpointer = checkpointer
        self.tensorboard_dir = tensorboard_dir
        self.callbacks = self._create_callbacks()

        # Print summary and validate memory
        self._print_run_summary()
        if not dpo_config.run_mpmd:
            self._validate_memory_usage()

    def init_models(self):
        """Initialize policy and reference models based on configuration."""
        # Initialize policy model
        if self.dpo_config.run_mpmd:
            resources = {"TPU": 4}
            policy_model = RayDPOPolicyModel.options(**resources).remote(
                self.dpo_config
            )
        else:
            policy_model = DPOPolicyModel(self.dpo_config)

        # Initialize reference model (or use policy model with LoRA disabled if using PEFT)
        if self.dpo_config.policy_model.lora_rank is None:
            if self.dpo_config.run_mpmd:
                resources = {"TPU": 4}
                ref_model = RayDPOReferenceModel.options(**resources).remote(
                    self.dpo_config
                )
            else:
                ref_model = DPOReferenceModel(self.dpo_config)
        else:
            ref_model = None

        return policy_model, ref_model

    def spmd_train_step(self, batch):
        """Run a single training step in SPMD mode."""
        if self.ref_model:
            ref_logits = self.ref_model.get_logits(batch)
        else:
            ref_logits = self.policy_model.get_ref_logits(batch)
        loss, metrics = self.policy_model.compute_loss_and_update(batch, ref_logits)
        return loss, metrics

    def mpmd_train_step(self, batch):
        """Run a single training step in MPMD mode using Ray."""
        # Get logits from policy and reference models in parallel
        policy_future = self.policy_model.get_logits.remote(batch)
        ref_future = self.ref_model.get_logits.remote(batch)

        policy_logits = ray.get(policy_future)
        ref_logits = ray.get(ref_future)

        # Compute loss and update policy model
        loss_future = self.policy_model.compute_loss_and_update.remote(
            batch, ref_logits
        )
        loss, metrics = ray.get(loss_future)
        return loss, metrics

    def spmd_eval_step(self, batch):
        """Run a single evaluation step in SPMD mode."""
        if self.ref_model:
            ref_logits = self.ref_model.get_logits(batch)
        else:
            ref_logits = self.policy_model.get_ref_logits(batch)
        loss, metrics = self.policy_model.compute_loss(batch, ref_logits)
        return loss, metrics

    def mpmd_eval_step(self, batch):
        """Run a single evaluation step in MPMD mode using Ray."""
        # Get logits from policy and reference models in parallel
        policy_future = self.policy_model.get_logits.remote(batch)
        ref_future = self.ref_model.get_logits.remote(batch)

        policy_logits = ray.get(policy_future)
        ref_logits = ray.get(ref_future)

        # Compute loss without updating policy model
        loss_future = self.policy_model.compute_loss.remote(batch, ref_logits)
        loss, metrics = ray.get(loss_future)
        return loss, metrics

    def train(self):
        """Run the training loop."""
        self.callbacks.on_train_begin()

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

                # Run appropriate training step based on mode
                if self.run_mpmd:
                    loss, metrics = self.mpmd_train_step(batch_input)
                else:
                    loss, metrics = self.spmd_train_step(batch_input)
                    # Wait for computation to complete for accurate timing
                    jax.block_until_ready(loss)

                epoch_loss += loss
                batches_seen_in_epoch += 1

                # Calculate training statistics
                step_time = time.time() - start_time
                tokens_per_second_per_device = (
                    self.global_batch_size
                    * self.train_dataloader.dataset.max_seq_len
                    / (step_time * self.device_count)
                )
                samples_per_second = self.global_batch_size / step_time

                # Compile step statistics
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
                    **metrics,
                }

                # Log progress at specified intervals
                if (
                    self.step_count == 1
                    or self.step_count % self.log_steps_interval == 0
                ):
                    print(step_stats)

                self.callbacks.on_train_batch_end(self.step_count, step_stats)

                # Run step-based evaluation if scheduled
                if (
                    (self.eval_dataloader is not None)
                    and (self.eval_steps_interval is not None)
                    and (self.step_count % self.eval_steps_interval == 0)
                ):
                    self.evaluate()

            # Compute and log epoch statistics
            epoch_loss = (
                epoch_loss / batches_seen_in_epoch if batches_seen_in_epoch > 0 else 0
            )
            self.callbacks.on_epoch_end(self.epoch_count, {"epoch_loss": epoch_loss})
            print(
                f"Train epoch {self.epoch_count} (epoch may be incomplete) loss: {epoch_loss}"
            )

            # Run epoch-based evaluation if scheduled
            if (
                (self.eval_dataloader is not None)
                and (self.eval_epochs_interval is not None)
                and (self.epoch_count % self.eval_epochs_interval == 0)
            ):
                self.evaluate()

            # Check termination conditions
            if self.steps and self.step_count >= self.steps:
                break
            if self.epochs and self.epoch_count >= self.epochs:
                break

        self.callbacks.on_train_end()

    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        self.callbacks.on_test_begin()
        eval_loss = 0
        eval_batches_seen = 0
        eval_start_time = time.time()

        # Process each batch in evaluation dataset
        for step_i, batch_input in enumerate(self.eval_dataloader):
            if (eval_batches_seen + 1) * self.global_batch_size > self.max_eval_samples:
                break

            start_time = time.time()

            # Run appropriate evaluation step based on mode
            if self.run_mpmd:
                loss, metrics = self.mpmd_eval_step(batch_input)
            else:
                loss, metrics = self.spmd_eval_step(batch_input)

            # Accumulate metrics
            eval_loss += loss
            eval_batches_seen += 1

            # Log progress at specified intervals
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
                    **metrics,
                }
                print(step_stats)

        # Compute final metrics
        if eval_batches_seen > 0:
            eval_loss = eval_loss / eval_batches_seen
            eval_time = time.time() - eval_start_time

            tokens_per_second_per_device = (
                eval_batches_seen
                * self.global_batch_size
                * self.train_dataloader.dataset.max_seq_len
            ) / (eval_time * self.device_count)

            samples_per_second = eval_batches_seen * self.global_batch_size / eval_time

            # Report evaluation results
            self.callbacks.on_test_end(
                {
                    "eval_loss": eval_loss,
                    "eval_samples_seen": eval_batches_seen * self.global_batch_size,
                    "eval_time": eval_time,
                    "tokens_per_second_per_device": tokens_per_second_per_device,
                    "tokens_per_second": tokens_per_second_per_device
                    * self.device_count,
                    "samples_per_second": samples_per_second,
                    "eval_steps_per_second": eval_batches_seen / eval_time,
                }
            )

            print(f"Eval loss after {self.step_count} training steps: {eval_loss}")

        return eval_loss

    def _print_run_summary(self):
        """Print a summary of the training run configuration."""
        training_duration = (
            f"Steps = {self.steps:,}" if self.steps else f"Epochs = {self.epochs}"
        )

        # Get model stats based on mode
        if self.dpo_config.run_mpmd:
            trainable_params, total_params, trainable_params_percent = ray.get(
                self.policy_model.trainable_params_stats.remote()
            )
        else:
            trainable_params, total_params, trainable_params_percent = (
                self.policy_model.trainable_params_stats()
            )

        # Include reference model stats if using separate reference model
        if self.ref_model:
            if self.dpo_config.run_mpmd:
                _, ref_total_params, _ = ray.get(
                    self.ref_model.trainable_params_stats.remote()
                )
            else:
                _, ref_total_params, _ = self.ref_model.trainable_params_stats()
            total_params += ref_total_params
        else:
            ref_total_params = "Using PEFT Model"

        # Build and print log banner
        logo_with_key_stats = (
            f"       '==='\n"
            f"        |||\n"
            f"     '- ||| -'\n"
            f"    /  |||||  \\   Kithara DPO| Device Count = {self.device_count}\n"
            f"   |   (|||)   |  {training_duration} | Batch size per device = {self.global_batch_size // self.device_count}\n"
            f"   |   |◕‿◕|   |  Global batch size = {self.global_batch_size} | Total policy parameters = {total_params:.3f}(GB) | Total reference parameters = {ref_total_params} (GB)\n"
            f"    \\  |||||  /   Trainable parameters = {trainable_params:.3f}(GB) ({trainable_params_percent}%) | Non-trainable = {total_params - trainable_params:.3f}(GB)\n"
            f"     --|===|--   "
        )
        print(logo_with_key_stats)

    def _create_callbacks(self):
        """Create and return a list of callbacks for training."""
        callbacks = []
        # if self.tensorboard_dir:
        #     callbacks.append(
        #         keras.callbacks.TensorBoard(
        #             log_dir=self.tensorboard_dir,
        #             update_freq="batch",
        #             write_steps_per_second=True,
        #         )
        #     )
        # if self.profiler:
        #     callbacks.append(self.profiler)
        if self.checkpointer and not self.dpo_config.run_mpmd:
            callbacks.append(self.checkpointer)
        return keras.callbacks.CallbackList(callbacks, model=self.policy_model)

    def _validate_setup(self):
        """Validate that the training configuration is valid."""
        assert (
            self.max_eval_samples >= self.global_batch_size
        ), "Number of eval examples must be greater or equal to global batch size"

        assert not (
            (
                self.eval_steps_interval is not None
                or self.eval_epochs_interval is not None
            )
            and self.eval_dataloader is None
        ), "Evaluation interval is set but no evaluation dataloader is provided"

        assert (
            self.steps is None or self.epochs is None
        ), "Specify either steps or epochs, not both"

        assert (
            self.eval_steps_interval is None or self.eval_epochs_interval is None
        ), "Specify either eval_steps_interval or eval_epochs_interval, not both"
