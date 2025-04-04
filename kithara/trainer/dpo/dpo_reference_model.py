import os

os.environ["KERAS_BACKEND"] = "jax"

import ray
from kithara.model.model import (
    create_model_from_config,
    ModelConfig,
)
from kithara.model.mpmd import RayModel
from kithara.trainer.validation_mixin import ValidationMixin


class DPOReferenceModel():
    """Reference model for DPO training."""

    def __init__(self, dpo_config: "DPOConfig"):
        """Initialize the DPO reference model.

        Args:
            dpo_config: Configuration for DPO training
        """
        if isinstance(dpo_config.policy_model, ModelConfig):
            self.model, *_ = create_model_from_config(dpo_config.policy_model)
        else:
            self.model = dpo_config.policy_model

    def get_logits(self, batch):
        """Get logits from the model for the given batch."""
        return self.model.get_logits(batch["x"])

    def generate(self, *args, **kwargs):
        """Generate text using the model."""
        return self.model.generate(*args, **kwargs)

    def trainable_params_stats(self):
        """Get statistics about trainable parameters."""
        return self.model.trainable_params_stats()


RayDPOReferenceModel = ray.remote(DPOReferenceModel)
