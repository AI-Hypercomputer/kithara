import os

os.environ["KERAS_BACKEND"] = "jax"

from functools import partial

import jax
import ray
import keras
from kithara.model.model import (
    create_model_from_config,
    create_optimizer_from_config,
    ModelConfig,
    OptimizerConfig,
)
from kithara.optimizers import convert_to_kithara_optimizer
from kithara.model.mpmd import RayModel
from kithara.trainer.validation_mixin import ValidationMixin
from kithara.trainer.dpo.dpo_loss import dpo_loss_fn

class DPOPolicyModel(RayModel, ValidationMixin):
    """Policy model for DPO training that can be run locally or distributed with Ray."""

    def __init__(self, dpo_config: "DPOConfig"):
        """Initialize the DPO policy model.

        Args:
            dpo_config: Configuration for DPO training
        """
        self.dpo_config = dpo_config
        self.model, self.mask_key, self.token_key = (
            self.create_model()
        )
        self.checkpointer = self.create_checkpointer()
        self.optimizer = self.create_optimizer()
        self.model.optimizer = self.optimizer

        if dpo_config.run_mpmd:
            self._validate_memory_usage(
                models=[self.model], optimizers=[self.optimizer]
            )

    def create_model(self):
        """Create model and checkpointer from config or use existing ones."""
        if isinstance(self.dpo_config.policy_model, ModelConfig):
            model, mask_key, token_key = create_model_from_config(
                self.dpo_config.policy_model
            )
        else:
            model, mask_key, token_key = (
                self.dpo_config.policy_model,
                self.dpo_config.policy_model.mask_key,
                self.dpo_config.policy_model.token_key,
            )
        return model, mask_key, token_key
    def create_checkpointer(self):
        return None
    def create_optimizer(self):
        """Create optimizer from config or use existing one."""
        if isinstance(self.dpo_config.policy_model.optimizer, OptimizerConfig):
            optimizer = create_optimizer_from_config(self.dpo_config.optimizer)
        else:
            optimizer = self.dpo_config.policy_model.optimizer

        if isinstance(optimizer, keras.optimizers.Optimizer):
            optimizer.build(self.model.trainable_variables)
        else:
            optimizer = convert_to_kithara_optimizer(
                optimizer, self.model.trainable_variables
            )
        return optimizer

    def get_logits(self, batch):
        """Get logits from the model for the given batch."""
        return self.model.forward(batch["x"])

    def get_ref_logits(self, batch):
        """Get reference logits with LoRA disabled."""
        self.model.disable_lora()
        logits = self.model.forward(batch["x"])
        self.model.enable_lora()
        return logits

    def loss_fn(self, trainable_variables, non_trainable_variables, ref_logits, batch):
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

    def compute_loss_and_update(self, batch, ref_logits):
        """Compute loss and update model weights (stateful wrapper)."""
        state = (
            self.model.trainable_variables,
            self.model.non_trainable_variables,
            self.optimizer.variables,
        )
        self._validate_sharding_correctness(
            data=batch,
            model=self.model,
            optimizer=self.optimizer,
        )
        loss, metrics, state = self._compute_loss_and_update(state, ref_logits, batch)
        self.model.update_model_state(*state)
        return loss, metrics

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _compute_loss_and_update(self, state, ref_logits, batch):
        """Compute loss and update model weights (stateless implementation).

        Args:
            state: Tuple of (trainable_variables, non_trainable_variables, optimizer_variables)
            ref_logits: Logits from reference model
            batch: Input batch

        Returns:
            Tuple of (loss, metrics, updated_state)
        """
        trainable_variables, non_trainable_variables, optimizer_variables = state

        (loss, (metrics, non_trainable_variables)), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True
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
    
    def compute_loss(self, batch, ref_logits):
        """Compute loss without updating weights (stateful wrapper)."""
        state = (
            self.model.trainable_variables,
            self.model.non_trainable_variables,
        )
        loss, metrics = self._compute_loss(state, ref_logits, batch)
        return loss, metrics
    
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _compute_loss(self, state, ref_logits, batch):
        """Compute loss without updating model weights (stateless implementation).

        Args:
            state: Tuple of (trainable_variables, non_trainable_variables)
            ref_logits: Logits from reference model
            batch: Input batch

        Returns:
            Tuple of (loss, metrics)
        """
        trainable_variables, non_trainable_variables = state

        loss, (metrics, _) = self.loss_fn(
            trainable_variables, non_trainable_variables, ref_logits, batch
        )

        return loss, metrics

    def generate(self, *args, **kwargs):
        """Generate text using the model."""
        return self.model.generate(*args, **kwargs)

    def trainable_params_stats(self):
        """Get statistics about trainable parameters."""
        return self.model.trainable_params_stats()

# Create Ray remote versions of the models
RayDPOPolicyModel = ray.remote(DPOPolicyModel)
