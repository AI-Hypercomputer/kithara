import subprocess
import sys

subprocess.run(["pip", "install", "setuptools==61.0"])
sys.path.append("kithra/model/maxtext/JetStream/jetstream")

import kithara
import os

os.environ["KERAS_BACKEND"] = "jax"
import keras

import ray
import jax
import jax.numpy as jnp
from kithara.model.maxtext.conversion_utils import MaxTextConversionMixin
from kithara.dataset.binary_preference_dataset import BinaryPreferenceDataset
from kithara.dataset import Dataloader

# logits_a, vjp_a = jax.vjp(model_a, x)
# vjp_a stores intermediate gradients --> is this too large?

# loss_val, logits_a_grad = jax.value_and_grad(loss)(logits_a, logits_b)

# x_grad_a, = vjp_a(logits_a_grad)


class PolicyModel:
    def __init__(self, beta=0.1):

        self.model = kithara.MaxTextModel.from_random(
            "gemma2-2b", per_device_batch_size=1, seq_len=100
        )

        # DPO temperature parameter
        self.beta = beta

        # Initialize optimizer
        self.optimizer = keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=0.01)
        self.optimizer.build(self.model.trainable_variables)
        self.optimizer_variables = self.optimizer.variables

        self.vjp_a = None
        self.logits_grad = None

    def get_logits(self, batch):

        @jax.jit
        def _forward(trainable_vars, inputs):
            logits, _ = self.model.stateless_call(
                trainable_vars,
                [var.value for var in self.model.non_trainable_variables],
                inputs,
            )
            return logits

        logits, vjp_a = jax.vjp(
            _forward,
            [var.value for var in self.model.trainable_variables],
            batch["x"],
        )
        total_sum = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, vjp_a, 0.0)
        print(f"Sum of all elements in VJP: {total_sum*2/(1024*1024*1024)} (GB)")
        # Sum of all elements in VJP: 13.285257892683148 (GB)
        
        self.vjp_a = vjp_a
        return logits

    def compute_loss_and_update(self, policy_logits, ref_logits):
        def _dpo_loss_impl(policy_logits, ref_logits):
            # fake loss
            return jnp.mean(policy_logits - ref_logits)

        loss, self.logits_grad = jax.value_and_grad(_dpo_loss_impl)(
            policy_logits, ref_logits
        )

        print("policy logits grad", self.logits_grad.shape) 
        # (8, 1024, 256128)

        # Compute loss and gradients
        # ! => OOM at this step
        (trainable_params_grad,) = self.vjp_a(self.logits_grad)

        # Apply gradients
        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            self.optimizer_variables,
            trainable_params_grad,
            self.model.trainable_variables,
        )

        self.model.trainable_variables = trainable_variables
        self.optimizer_variables = optimizer_variables

        return {"loss": float(loss)}

    def generate(self, *args, **kwargs):
        """Generate text using the model"""
        return self.model.generate(*args, **kwargs)


def toy_dpo():

    dpo_dataset_dict = [
        {
            "prompt": "hello",
            "chosen": " hi nice to meet you",
            "rejected": " leave me alone",
        }
        for _ in range(100)
    ]
    dataset = ray.data.from_items(dpo_dataset_dict)
    dataset = BinaryPreferenceDataset(
        dataset, tokenizer_handle="google/gemma-2-2b", model_type="MaxText"
    )

    dataloader = Dataloader(dataset, per_device_batch_size=1)

    policy_model = PolicyModel()

    for i, batch in enumerate(dataloader):
        print(f"Iteration {i+1}")

        policy_logits = policy_model.get_logits(batch)

        loss = policy_model.compute_loss_and_update(policy_logits, policy_logits)

        print("-" * 30)
        if i == 1:
            break


if __name__ == "__main__":
    toy_dpo()
