import os

os.environ["KERAS_BACKEND"] = "jax"
import kithara

import jax
import ray
from kithara.trainer.loss_fn.dpo_loss import dpo_loss_fn
from kithara.model.model import (
    create_model_from_config,
    create_optimizer_from_config,
    ModelConfig,
    OptimizerConfig,
)
from dataclasses import dataclass
from typing import Union
from kithara.distributed.sharding.utils import (
    entire_tree_is_sharded,
    is_not_sharded_and_is_large,
    get_size_in_mb,
    get_size_in_gb,
)
from kithara.optimizers import convert_to_kithara_optimizer
from kithara.callbacks import Profiler, Checkpointer
from typing import Any, Union, List, Tuple
import numpy as np
import keras
import sys
from functools import partial
import optax
from kithara.trainer.rlhf.raymodel import RayModel
import time


@dataclass
class DPOConfig:
    policy_model: Union[ModelConfig, kithara.Model]
    optimizer: Union[
        OptimizerConfig,
        keras.Optimizer,
        optax.GradientTransformation,
        optax.GradientTransformationExtraArgs,
    ]
    beta: float = 0.1
    label_smoothing: float = 0.0
    run_mpmd: bool = False


class DPOPolicyModel(RayModel):
    def __init__(self, dpo_config: DPOConfig):
        self.dpo_config = dpo_config

        self.model, self.mask_key, self.token_key, self.checkpointer = (
            self.create_model_and_checkpointer()
        )
        self.optimizer = self.create_optimizer()
        self.model.optimizer = self.optimizer

        if dpo_config.run_mpmd:
            self._validate_memory_usage()

    def create_model_and_checkpointer(self):
        if isinstance(self.dpo_config.policy_model, ModelConfig):
            model, mask_key, token_key, checkpointer = create_model_from_config(
                self.dpo_config.policy_model
            )
        else:
            model, mask_key, token_key, checkpointer = (
                self.dpo_config.policy_model,
                self.dpo_config.policy_model.mask_key,
                self.dpo_config.policy_model.token_key,
                self.dpo_config.checkpointer,
            )
        return model, mask_key, token_key, checkpointer

    def create_optimizer(self):
        if isinstance(self.dpo_config.optimizer, OptimizerConfig):
            optimizer = create_optimizer_from_config(self.dpo_config.optimizer)
        else:
            optimizer = self.dpo_config.optimizer

        if isinstance(optimizer, keras.optimizers.Optimizer):
            optimizer.build(self.model.trainable_variables)
        else:
            optimizer = convert_to_kithara_optimizer(
                optimizer, self.model.trainable_variables
            )
        return optimizer

    def get_logits(self, batch):
        logits = self.model.forward(batch["x"])
        return logits

    def get_ref_logits(self, batch):
        self.model.disable_lora()
        logits = self.model.forward(batch["x"])
        self.model.enable_lora()
        return logits

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _compute_loss_and_update(self, state, ref_logits, batch):
        """Stateless update"""

        trainable_variables, non_trainable_variables, optimizer_variables = state

        def loss_fn(trainable_variables, non_trainable_variables, ref_logits, batch):

            logits, non_trainable_variables = self.model.stateless_call(
                trainable_variables,
                non_trainable_variables,
                batch["x"],
            )
            loss, metrics = dpo_loss_fn(
                logits,
                ref_logits,
                batch["x"][self.mask_key],
                batch["x"][self.token_key],
                beta=self.dpo_config.beta,
                label_smoothing=self.dpo_config.label_smoothing,
            )
            return loss, (metrics, non_trainable_variables)

        (loss, (metrics, non_trainable_variables)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(trainable_variables, non_trainable_variables, ref_logits, batch)

        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables,
            grads,
            trainable_variables,
        )

        return (
            loss,
            metrics,
            (trainable_variables, non_trainable_variables, optimizer_variables),
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _compute_loss(self, state, ref_logits, batch):
        """Stateless update"""

        trainable_variables, non_trainable_variables, optimizer_variables = state

        def loss_fn(trainable_variables, non_trainable_variables, ref_logits, batch):

            logits, non_trainable_variables = self.model.stateless_call(
                trainable_variables,
                non_trainable_variables,
                batch["x"],
            )
            loss, metrics = dpo_loss_fn(
                logits,
                ref_logits,
                batch["x"][self.mask_key],
                batch["x"][self.token_key],
                beta=self.dpo_config.beta,
                label_smoothing=self.dpo_config.label_smoothing,
            )
            return loss, (metrics, non_trainable_variables)

        loss, (metrics, non_trainable_variables) = loss_fn(
            trainable_variables, non_trainable_variables, ref_logits, batch
        )

        return loss, metrics

    def compute_loss_and_update(self, batch, ref_logits):
        """stateful update"""
        state = (
            self.model.trainable_variables,
            self.model.non_trainable_variables,
            self.optimizer.variables,
        )
        self._validate_sharding_correctness(batch, state)
        loss, metrics, state = self._compute_loss_and_update(state, ref_logits, batch)
        self.model.update_model_state(*state)
        return loss, metrics

    def compute_loss(self, batch, ref_logits):
        """stateful update"""
        state = (
            self.model.trainable_variables,
            self.model.non_trainable_variables,
            self.optimizer.variables,
        )
        loss, metrics = self._compute_loss(state, ref_logits, batch)
        return loss, metrics

    def generate(self, *args, **kwargs):
        """Generate text using the model"""
        return self.model.generate(*args, **kwargs)

    def save_checkpoint(self, step):
        self.checkpointer.save(step)

    def trainable_params_stats(self):
        return self.model.trainable_params_stats()

    def _validate_sharding_correctness(self, data, state):
        """This method performs several sharding correctness checks and prints
        warnings for any sharding issues detected.

        1. Checks if data is properly sharded
        2. Validates sharding of trainable variables
        3. Validates sharding of non-trainable variables
        4. Validates sharding of optimizer variables

        Args:
            data: Input batch to validate
            state: Current model state tuple

        """
        try:
            if not entire_tree_is_sharded(data):
                print(
                    "Warning: data is not sharded",
                    data["y"].shape,
                    data["y"].sharding,
                )
            for variable, value in zip(self.model.trainable_variables, state[0]):
                if is_not_sharded_and_is_large(value):
                    print(
                        f"Step {self.step_count}: trainable variable is not sharded",
                        f"{get_size_in_mb(value)}mb",
                        variable.path,
                        value.shape,
                        value.sharding,
                    )
            for variable, value in zip(self.model.non_trainable_variables, state[1]):
                if is_not_sharded_and_is_large(value):
                    print(
                        f"Step {self.step_count}: nontrainable variable is not sharded",
                        f"{get_size_in_mb(value)}mb",
                        variable.path,
                        value.shape,
                        value.sharding,
                    )

            _ = jax.tree.map(
                lambda variable, value: (
                    print(
                        f"Step {self.step_count}: optimizer variable is not sharded",
                        f"{get_size_in_mb(value)}mb",
                        variable.path,
                        value.shape,
                        value.sharding,
                    )
                    if is_not_sharded_and_is_large(value)
                    else None
                ),
                self.optimizer.variables,
                state[2],
            )

        except Exception as e:
            print(f"Error during sharding correctness validation: {e}")

    def _validate_memory_usage(self):
        """This method checks the current HBM usage matches the expected HBM
        usage.

        Current HBM usage is calculated by summing the size of all live arrays,
        expected HBM usage is calculated by summing the size of all model and
        optimizer variables.
        """

        total_size = 0
        for v in self.model.variables:
            total_size += get_size_in_mb(v.value)

        total_size += jax.tree.reduce(
            lambda agg, leaf: jax.numpy.add(agg, get_size_in_mb(leaf.value)),
            self.optimizer.variables,
            initializer=0,
        )

        live_arrays = jax.live_arrays()
        live_arrays_size = 0
        for v in live_arrays:
            live_arrays_size += get_size_in_mb(v)

        if not np.isclose(total_size, live_arrays_size, atol=1.0):
            print(
                f"WARNING: Potential memory leakage. HBM usage is {live_arrays_size:.3f} MB "
                f"but model and optimizer are only {total_size:.3f} MB in size."
            )
        else:
            print(
                f"✅ No memory leakage detected. HBM usage ({live_arrays_size:.3f} MB) "
                f"matches model and optimizer size ({total_size:.3f} MB)."
            )

        try:
            memory_info = jax.local_devices()[0].memory_stats()
            memory_per_device_mb = memory_info["bytes_limit"] / (1024**2)
            total_memory = memory_per_device_mb * jax.device_count()
            print(
                f"Total memory available is {total_memory:.3f} MB, if you run into "
                "errors, check if your memory usage is close to the limit, and either "
                "reduce your per-device batch size or sequence length."
            )
        except Exception as e:
            # memory_info is not available on some TPUs
            pass


class DPOReferenceModel(RayModel):
    def __init__(self, dpo_config: DPOConfig):

        if isinstance(dpo_config.policy_model, ModelConfig):
            self.model, *_ = create_model_from_config(dpo_config.policy_model)
        else:
            self.model = dpo_config.policy_model

    def get_logits(self, batch):
        logits = self.model.get_logits(batch["x"])
        return logits

    def generate(self, *args, **kwargs):
        """Generate text using the model"""
        return self.model.generate(*args, **kwargs)

    def trainable_params_stats(self):
        return self.model.trainable_params_stats()


RayDPOPolicyModel = ray.remote(DPOPolicyModel)
RayDPOReferenceModel = ray.remote(DPOReferenceModel)


class DPOTrainer:
    """
    DPOTrainer may be initialized on a TPU or CPU.
    When running in mpmd mode, DPOTrainer is initialized on CPU, and dispatches tasks to TPUs
    When running in spmd mode, DPOTrainer is initialized and run on TPU.

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
        self.profiler = profiler  # TODO: support profiling
        self.checkpointer = checkpointer  # TODO: support checkpointer
        self.tensorboard_dir = tensorboard_dir  # TODO: support tensorboard_dir
        self.callbacks = self._create_callbacks()

        # Print summary
        self._print_run_summary()
        if dpo_config.run_mpmd is False:
            self._validate_memory_usage()

    def init_models(self):
        if self.dpo_config.run_mpmd:
            resources = {"TPU": 4}
            policy_model = RayDPOPolicyModel.options(**resources).remote(
                self.dpo_config
            )
        else:
            policy_model = DPOPolicyModel(self.dpo_config)

        if self.dpo_config.policy_model.lora_rank is None:
            if self.dpo_config.run_mpmd:
                resources = {"TPU": 4}
                ref_model = DPOReferenceModel.options(**resources).remote(
                    self.dpo_config
                )
            else:
                ref_model = DPOReferenceModel(self.dpo_config)
        else:
            ref_model = None
        return policy_model, ref_model

    def spmd_train_step(self, batch):
        if self.ref_model:
            ref_logits = self.ref_model.get_logits(batch)
        else:
            ref_logits = self.policy_model.get_ref_logits(batch)
        loss, metrics = self.policy_model.compute_loss_and_update(batch, ref_logits)
        return loss, metrics

    def mpmd_train_step(self, batch):
        # Right now:
        # |policy_logits||loss_and_update||policy_logits||loss_and_update|------
        # |ref_logits***|-----------------|ref_logits***|------------------------

        # Ideally:
        #                |logits_loss_and_update||logits_loss_and_update|------
        # |ref_logits***|---------|ref_logits***|------------------------|ref_logits***|

        policy_future = self.policy_model.get_logits.remote(batch)
        ref_future = self.ref_model.get_logits.remote(batch)

        policy_logits = ray.get(policy_future)
        ref_logits = ray.get(ref_future)

        loss_future = self.policy_model.compute_loss_and_update.remote(
            batch,
            ref_logits,
        )
        loss, metrics = ray.get(loss_future)
        return loss, metrics

    def spmd_eval_step(self, batch):
        if self.ref_model:
            ref_logits = self.ref_model.get_logits(batch)
        else:
            ref_logits = self.policy_model.get_ref_logits(batch)
        loss, metrics = self.policy_model.compute_loss(batch, ref_logits)
        return loss, metrics

    def mpmd_eval_step(self, batch):

        policy_future = self.policy_model.get_logits.remote(batch)
        ref_future = self.ref_model.get_logits.remote(batch)

        policy_logits = ray.get(policy_future)
        ref_logits = ray.get(ref_future)

        loss_future = self.policy_model.compute_loss.remote(
            batch,
            ref_logits,
        )
        loss, metrics = ray.get(loss_future)
        return loss, metrics

    def train(self):
        self.callbacks.on_train_begin()
        # for batch in self.train_dataloader:
        #     if self.step_count >= self.steps:
        #         break

        #     if self.run_mpmd:
        #         loss, metrics = self.mpmd_train_step(batch)
        #     else:
        #         loss, metrics = self.spmd_train_step(batch)

        #     self.step_count += 1
        #     print("loss", loss)
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

                if self.run_mpmd:
                    loss, metrics = self.mpmd_train_step(batch_input)
                else:
                    loss, metrics = self.spmd_train_step(batch_input)
                    # Wait for computation to complete for accurate step time
                    jax.block_until_ready(loss)

                epoch_loss += loss
                batches_seen_in_epoch += 1

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
                    **metrics,
                    # "learning_rate": (round(float(self.optimizer.learning_rate.value),7)
                    #                   if self.optimizer.learning_rate is not None else None),
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
                    self.evaluate()

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
                self.evaluate()

            # Check termination conditions
            if self.steps and self.step_count >= self.steps:
                break
            if self.epochs and self.epoch_count >= self.epochs:
                break

        self.callbacks.on_train_end()

    def evaluate(self):
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

            if self.run_mpmd:
                loss, metrics = self.mpmd_train_step(batch_input)
            else:
                loss, metrics = self.spmd_train_step(batch_input)

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
                    **metrics,
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

    def _print_run_summary(self):

        training_duration = (
            f"Steps = {self.steps:,}" if self.steps else f"Epochs = {self.epochs}"
        )
        if self.dpo_config.run_mpmd:
            trainable_params, total_params, trainable_params_percent = ray.get(
                self.policy_model.trainable_params_stats.remote()
            )
        else:
            trainable_params, total_params, trainable_params_percent = (
                self.policy_model.trainable_params_stats()
            )

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

        # TODO: Implement more structured logging
        for attr_name, attr_value in vars(self).items():
            print(attr_name, attr_value)

    def _create_callbacks(self):
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

        assert (self.eval_steps_interval is None) or (
            self.eval_epochs_interval is None
        ), "Specify either eval_steps_interval or eval_epochs_interval, not both"

    def _validate_memory_usage(self):
        """This method checks the current HBM usage matches the expected HBM
        usage.

        Current HBM usage is calculated by summing the size of all live arrays,
        expected HBM usage is calculated by summing the size of all model and
        optimizer variables.
        """

        total_size = 0
        for v in self.policy_model.model.variables:
            total_size += get_size_in_mb(v.value)
        if self.ref_model:
            for v in self.ref_model.model.variables:
                total_size += get_size_in_mb(v.value)

        total_size += jax.tree.reduce(
            lambda agg, leaf: jax.numpy.add(agg, get_size_in_mb(leaf.value)),
            self.policy_model.optimizer.variables,
            initializer=0,
        )

        live_arrays = jax.live_arrays()
        live_arrays_size = 0
        for v in live_arrays:
            live_arrays_size += get_size_in_mb(v)

        if not np.isclose(total_size, live_arrays_size, atol=1.0):
            print(
                f"WARNING: Potential memory leakage. HBM usage is {live_arrays_size:.3f} MB "
                f"but model and optimizer are only {total_size:.3f} MB in size."
            )
        else:
            print(
                f"✅ No memory leakage detected. HBM usage ({live_arrays_size:.3f} MB) "
                f"matches model and optimizer size ({total_size:.3f} MB)."
            )

        try:
            memory_info = jax.local_devices()[0].memory_stats()
            memory_per_device_mb = memory_info["bytes_limit"] / (1024**2)
            total_memory = memory_per_device_mb * jax.device_count()
            print(
                f"Total memory available is {total_memory:.3f} MB, if you run into "
                "errors, check if your memory usage is close to the limit, and either "
                "reduce your per-device batch size or sequence length."
            )
        except Exception as e:
            # memory_info is not available on some TPUs
            pass
