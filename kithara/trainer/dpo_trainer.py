import os
import subprocess
import sys

os.environ["KERAS_BACKEND"] = "jax"
subprocess.run(["pip", "install", "setuptools==61.0"])
sys.path.append("kithra/model/maxtext/JetStream/jetstream")

import keras
import kithara

import jax
from kithara.dataset.binary_preference_dataset import BinaryPreferenceDataset
from kithara.dataset import Dataloader
import ray
from kithara.rlhf.dpo_loss import dpo_loss_fn
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class ModelConfig:
    preset_handle: str
    model_type: str
    lora_rank: int
    precision: str
    per_device_batch_size: int
    seq_len: int
    optimizer_name: str
    learning_rate: float 
    

@dataclass
class DPOConfig:
    policy_model: ModelConfig
    beta: float = 0.1


def create_model_from_config(config):
    if config.model_type == "KerasHub":
        model = kithara.KerasHubModel.from_preset(
            config.preset_handle,
            lora_rank=config.lora_rank,
            precision=config.precision,
        )
        mask_key = "padding_mask"
        token_key = "token_ids"
    elif config.model_type == "MaxText":
        model = kithara.MaxTextModel.from_preset(
            config.preset_handle,
            precision=config.precision,
            seq_len=config.seq_len,
            per_device_batch_size=config.per_device_batch_size,
        )
        mask_key = "segment_ids"
        token_key = "tokens"
    else:
        raise ValueError(
            "Model type not supported. Must be one of MaxText and KerasHub"
        )
    return model, mask_key, token_key


def create_optimizer_from_config(config):
    if config.optimizer_name == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=config.learning_rate, weight_decay=0.01)
    return optimizer


class DPOPolicyModel:
    def __init__(self, dpo_config: DPOConfig):
        self.dpo_config = dpo_config
        model_config = dpo_config.policy_model
        self.model, self.mask_key, self.token_key = create_model_from_config(model_config)
        self.optimizer = create_optimizer_from_config(model_config)
        self.optimizer.build(self.model.trainable_variables)
        self.model.optimizer = self.optimizer
        self._compute_loss_and_update_fn = jax.jit(
            self._compute_loss_and_update, donate_argnums=(0,)
        )

    def get_logits(self, batch):
        logits = self.model.get_logits(batch["x"])
        return logits

    def get_ref_logits(self, batch):

        self.model.disable_lora()
        logits = self.model.get_logits(batch["x"])
        self.model.enable_lora()
        return logits

    def _compute_loss_and_update(self, state, ref_logits, batch):
        """Stateless update"""

        trainable_variables, non_trainable_variables, optimizer_variables = state

        def loss_fn(trainable_variables, non_trainable_variables, ref_logits, batch):

            logits, non_trainable_variables = self.model.stateless_call(
                trainable_variables,
                non_trainable_variables,
                batch["x"],
            )
            loss = dpo_loss_fn(
                logits,
                ref_logits,
                batch["x"][self.mask_key],
                batch["x"][self.token_key],
                beta = self.dpo_config.beta
            )
            return loss, non_trainable_variables

        (loss, non_trainable_variables), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(trainable_variables, non_trainable_variables, ref_logits, batch)

        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables,
            grads,
            trainable_variables,
        )

        return loss, (trainable_variables, non_trainable_variables, optimizer_variables)

    def compute_loss_and_update(self, batch, ref_logits):
        state = (
            self.model.trainable_variables,
            self.model.non_trainable_variables,
            self.optimizer.variables,
        )
        loss, state = self._compute_loss_and_update_fn(state, ref_logits, batch)
        self.model.update_model_state(*state)
        return loss

    def generate(self, *args, **kwargs):
        """Generate text using the model"""
        return self.model.generate(*args, **kwargs)


class DPOReferenceModel:
    def __init__(self, dpo_config: DPOConfig):
        
        self.model, *_ = create_model_from_config(dpo_config.policy_model)

    def get_logits(self, batch):
        logits = self.model.get_logits(batch["x"])
        return logits

    def generate(self, *args, **kwargs):
        """Generate text using the model"""
        return self.model.generate(*args, **kwargs)


RayDPOPolicyModel = ray.remote(DPOPolicyModel)
RayDPOReferenceModel = ray.remote(DPOReferenceModel)


class DPOTrainer:
    def __init__(
        self,
        dpo_config: DPOConfig,
        train_dataloader: kithara.Dataloader,
        eval_dataloader: kithara.Dataloader = None,
        run_mpmd: bool = False,
        steps: int = 100,
    ):
        self.dpo_config = dpo_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.run_mpmd = run_mpmd
        self.steps = steps
        self.step_count = 0
        self.policy_model, self.ref_model = self.init_models()

    def init_models(self):
        if self.run_mpmd:
            resources = {"TPU": 4}
            policy_model = RayDPOPolicyModel.options(**resources).remote(
                self.dpo_config
            )
        else:
            policy_model = DPOPolicyModel(self.dpo_config)

        if self.dpo_config.policy_model.lora_rank is None:
            if self.run_mpmd:
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
        loss = self.policy_model.compute_loss_and_update(batch, ref_logits)
        return loss

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
        loss = ray.get(loss_future)
        return loss

    def train(self):

        for batch in self.train_dataloader:
            if self.step_count >= self.steps:
                break

            if self.run_mpmd:
                loss = self.mpmd_train_step(batch)
            else:
                loss = self.spmd_train_step(batch)
            
            self.step_count += 1
            print("loss", loss)

