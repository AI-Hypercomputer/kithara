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

import keras
import jax
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Union
from keras.src.backend.common import global_state
from keras.distribution import set_distribution
from kithara.callbacks.checkpointer import Checkpointer
from kithara.distributed.sharding import ShardingStrategy
from kithara.distributed.sharding.utils import (
    print_elements_that_are_unsharded_and_large_in_pytree,
)
from kithara.dataset.utils import initialize_tokenizer
from dataclasses import dataclass
from kithara.distributed.sharding.utils import get_size_in_gb
from functools import partial


class ModelImplementationType(str, Enum):

    KERASHUB = "KerasHub"
    MAXTEXT = "MaxText"

    @classmethod
    def list_supported_types(cls) -> list[str]:
        """Returns a list of all supported model implementation types."""
        return [impl.value for impl in cls]


class ModelValidationMixin:
    """Mixin providing common model validation functionality."""

    def validate_sharding(self, model: Any) -> None:
        if model is None:
            raise ValueError("Model has not been successfully created.")
        print_elements_that_are_unsharded_and_large_in_pytree(model)


class Model(ABC, ModelValidationMixin):
    """
    Base class for all models in Kithara. This class serves as a thin
    wrapper around the underlying model instance, providing a uniform
    interface for Kithara workloads. Currently supported underlying model
    implementations include MaxText and KerasHub models.

    Attributes:
        sharding_strategy(kithara.ShardingStrategy): Strategy used for
            distributing model, optimizer, and data tensors.
            E.g. `kithara.PredefinedShardingStrategy("fsdp", "gemma2-2b")`.
        model(Keras.Model): The underlying Keras model instance.
        model_name(str, optional): Optional name of the model.
        precision(str, optional): Optional mixed-precision policy for
            model weights and activations.
            Default is "mixed_bfloat16". Supported policies include
            "float32", "float16", "bfloat16", "mixed_float16", and "mixed_bfloat16".
            Mixed precision policies load model weight in float32 and casts
            activations to the specified dtype.
        scan_layers: Boolean indicating whether to scan layers using
            jax.lax.scan, which speeds up training compilation.
            Currently only MaxText models support this feature.
        lora_rank: Int indicating the rank of the LoRA weights. Currently
            only KerasHub models support LoRA. KerasHub models apply LoRA
            to the v_proj and q_proj weights.
    Key Methods:
        __init__():
            Initializes the Model instance with the given parameters.
        __getattr__():
            Delegates any unknown attributes/methods to the underlying model.
        generate():
            Generate text tokens using the model based on the input prompt.
        stateless_call():
            Runs the forward pass of the model in a stateless fashion. This
            function is handled by keras.model.stateless_call().
    """

    def __init__(
        self,
        model: keras.Model,
        sharding_strategy: ShardingStrategy,
        model_name: str = None,
        precision: str = "mixed_bfloat16",
        scan_layers: bool = False,
        lora_rank: int = None,
    ):

        self.sharding_strategy = sharding_strategy
        self.model = model
        self.scan_layers = scan_layers
        self.model_name = model_name
        self.precision = precision
        self.lora_rank = lora_rank
        self.weight_dtype = self._weight_dtype(precision)
        self.activation_dtype = self._activation_dtype(precision)
        # Tensorboard requires `model.optimizer`.
        # This will be automatically set during training.
        self._optimizer = None

    @property
    def variables(self):
        return self.model.variables

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def non_trainable_variables(self):
        return self.model.non_trainable_variables

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def __getattr__(self, name):
        try:
            # Try to get the attribute from the Model class first
            return object.__getattribute__(self, name)
        except AttributeError:
            # If not found, delegate to _model
            model = object.__getattribute__(self, "model")
            return getattr(model, name, None)

    @staticmethod
    def _weight_dtype(precision: Optional[str] = None) -> str:
        if "mixed" in precision:
            return "float32"
        return precision

    @staticmethod
    def _activation_dtype(precision: Optional[str] = None) -> str:
        if "mixed" in precision:
            return precision.split("_")[1]
        return precision

    def trainable_params_stats(self):
        trainable_params = sum(
            get_size_in_gb(v.value) for v in self.trainable_variables
        )
        total_params = trainable_params + sum(
            get_size_in_gb(v.value) for v in self.non_trainable_variables
        )
        trainable_params_percent = round((trainable_params / total_params) * 100, 2)
        return trainable_params, total_params, trainable_params_percent

    @abstractmethod
    def save_in_hf_format(
        self, output_dir: str, dtype: str = "auto", parallel_threads=8
    ):
        """Save the model in HuggingFace format.

        Args:
            output_dir (str): Directory path where the model should be saved.
                Directory could be local or a Google cloud storage path, and
                will be created if it doesn't exist.
            dtype (str, optional): Data type for saved weights. Defaults to "auto".
            parallel_threads (int, optional): Number of parallel threads to use for saving.
        """

    def update_model_state(
        self,
        trainable_variables=None,
        non_trainable_variables=None,
        optimizer_variables=None,
    ):
        """Update model internal parameters with the provided state."""
        if trainable_variables:
            for variable, value in zip(
                self.model.trainable_variables, trainable_variables
            ):
                value = jax.lax.with_sharding_constraint(value, variable._layout)
                variable.assign(value)

        if non_trainable_variables:
            for variable, value in zip(
                self.model.non_trainable_variables, non_trainable_variables
            ):
                value = jax.lax.with_sharding_constraint(value, variable._layout)
                variable.assign(value)

        if optimizer_variables:
            _ = jax.tree.map(
                lambda variable, value: variable.assign(
                    jax.lax.with_sharding_constraint(value, variable._layout)
                ),
                self.optimizer.variables,
                optimizer_variables,
            )

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def generate(
        self,
        inputs: Union[
            str, List[str], List[int], np.ndarray, List[np.ndarray], List[List[int]]
        ],
        max_length: int = 100,
        stop_token_ids: Union[str | List[int]] = "auto",
        strip_prompt: bool = False,
        tokenizer: Optional["AutoTokenizer"] = None,
        tokenizer_handle: Optional[str] = None,
        return_decoded: bool = True,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[str] | List[List[int]]]:
        """Generate text tokens using the model.
        Args:
            inputs (str|list[str]|list[np.ndarray]|list[list[int]]): Inputs can be
                either string or integer tokens. String inputs can be a single string,
                or a list of strings. Token inputs can be a numpy array,
                a list of numpy arrays, an integer array, or a list of integer arrays.
            max_length (int, optional): Maximum total sequence length
                (prompt + generated tokens).
            stop_token_ids (List[int], optional): List of token IDs that stop
                generation. Defaults to "auto", which extracts the end token id
                from the tokenizer.
            strip_prompt (bool, optional): If True, returns only the generated
                tokens without the input prompt. If False, returns the full sequence
                including the prompt. Defaults to False.
            tokenizer (AutoTokenizer, optional): A HuggingFace AutoTokenizer instance
            tokenizer_handle (str, optional): A HuggingFace Tokenizer handle, e.g.
                "google/gemma-2-2b"
            return_decoded (bool, optional): If True, returns the decoded text using
                the tokenizer, otherwise return the predicted tokens. Default to True.
                This option must be set to False if no tokenizer is provided.
            skip_special_tokens(bool, optional): If True, skip special tokens during
                decoding. E.g. will not include <bos> and <eos> in the final output.
                This option is ignore when return_decoded=False. Default to True.

        Returns:
            A list of string if return_decoded=True, or a list[list[int]] containing
            generated token IDs for each prompt

        Example:
            ```
            # Return tokens
            prompt= "what is your name?"
            pred_tokens = model.generate(prompt, max_length=100, tokenizer_handle="hf://google/gemma-2-2b")
            print(pred_tokens)

            # Return text
            pred_text = model.generate(prompt, max_length=100, tokenizer_handle="hf://google/gemma-2-2b", return_decoded=True, strip_prompt=True)
            print(pred_text)

            # Use an initialized tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("hf://google/gemma-2-2b")
            pred_text = model.generate(prompt, max_length=100, tokenizer=tokenizer, return_decoded=True, strip_prompt=True)
            ```
        """

        # Type checking
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        elif isinstance(inputs, list) and isinstance(inputs[0], int):
            inputs = [inputs]
        elif isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list):
            raise TypeError("Non-list input must be a str or a np.ndarray.")
        if isinstance(inputs[0], str) or return_decoded:
            assert (
                tokenizer or tokenizer_handle
            ) is not None, "Either tokenizer or tokenizer_handle must be provided when inputs are strings."

        if tokenizer or tokenizer_handle:
            tokenizer = initialize_tokenizer(tokenizer_handle, tokenizer)

        if stop_token_ids == "auto":
            stop_token_ids = []
            token_attributes = [
                "end_token_id",
                "eos_token_id",
                "end_token2_id",
                "eos_token2_id",
            ]
            for attr in token_attributes:
                if hasattr(tokenizer, attr):
                    stop_token_ids.append(getattr(tokenizer, attr))

        if isinstance(inputs[0], list):
            inputs = [np.array(l) for l in inputs]
        tokens: list[list[int]] = self._generate(
            inputs,
            max_length=max_length,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
            tokenizer=tokenizer,
        )
        if return_decoded:
            tokenizer = initialize_tokenizer(tokenizer_handle, tokenizer)
            text = [
                tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
                for token_ids in tokens
            ]
            return text
        return tokens

    @abstractmethod
    def _generate(
        self,
        inputs: Union[List[str], List[np.ndarray]],
        max_length: int = None,
        stop_token_ids: Optional[List] = None,
        strip_prompt: str = False,
        tokenizer: Optional["AutoTokenizer"] = None,
    ) -> List[List[int]]:
        """Generate tokens using the model. This function is intended to generate
        tokens for custom models. MaxTextModel and KerasHubModel have provided
        their own `_generate()` implementation.

        Args:
            inputs (list[str]|list[np.ndarray]): A list of strings, or a list
                of numpy arrays containing integer token ids.
            max_length (int, optional): Maximum total sequence length
                (prompt + generated tokens).
            stop_token_ids (List[int], optional): List of token IDs that stop
                generation.
            strip_prompt (bool, optional): If True, returns only the generated
                tokens without the input prompt tokens. If False, returns all
                tokens, including the prompt tokens. Defaults to False.
            tokenizer (AutoTokenizer, optional): A HuggingFace AutoTokenizer instance.
                This is guaranteed to be provided when inputs are strings.

        Returns:
            list[list[int]]: Generated token IDs for each prompt
        """
        pass


def set_precision(
    precision: Optional[str] = None,
    weight_dtype: Optional[str] = None,
    activation_dtype: Optional[str] = None,
) -> None:
    """
    Sets the precision policy for mixed precision training. This function overrides the
    default precision policy and must be called before loading the model. Note you do
    not need to manually call this function unless you are defining a custom model.

    Args:
        precision (Optional[str]): The precision policy to set. Can be one of
            'float32', 'float16', 'mixed_float16', or 'mixed_bfloat16'. If None, the
            precision will be inferred from `weight_dtype` and `activation_dtype`.
        weight_dtype (Optional[str]): The data type for weights. Used to infer
            the precision policy if `precision` is None. Must be one of
            'float32', 'float16', or 'bfloat16'.
        activation_dtype (Optional[str]): The data type for activations. Used to
            infer the precision policy if `precision` is None. Must be one of
            'float32', 'float16', or 'bfloat16'.

    Returns:
        precision (str): The precision policy that was set.
    """

    assert (
        precision is None
        and (weight_dtype is not None)
        and (activation_dtype is not None)
    ) or (
        (precision is not None)
        and (weight_dtype is None)
        and (activation_dtype is None)
    ), "Please only specify either weight and activation dtype, or precision, but not both."

    if precision is None:
        if weight_dtype == activation_dtype:
            precision = weight_dtype
        elif weight_dtype == "float32" and activation_dtype == "float16":
            precision = "mixed_float16"
        elif weight_dtype == "float32" and activation_dtype == "bfloat16":
            precision = "mixed_bfloat16"
        else:
            raise ValueError(
                "Weight dtype and activation dtype combination is not valid."
            )

    policy = global_state.get_global_attribute("dtype_policy", None)
    if policy:
        print(f"Overriding existing policy: {policy}")
    keras.mixed_precision.set_global_policy(precision)
    return precision


def set_global_sharding_strategy(strategy: Optional[ShardingStrategy]) -> None:
    """
    Sets the sharding strategy for the model and batch input.This function
    overrides the existing sharding strategy and must be called before loading
    the model.

    Args:
        strategy (Optional[kithara.ShardingStrategy]): The sharding strategy to be set
            globally. If None, no changes are made to the global state.
    """
    if strategy:
        if global_state.get_global_attribute("distribution") is not None:
            print("WARNING: Distribution strategy is being overridden.")
        set_distribution(strategy.distribution)
        global_state.set_global_attribute("DATA_SHARDING", strategy.data_sharding)


def set_global_model_implementation_type(model_type) -> None:
    """
    Sets the global variable representing the model implementation type (MAXTEXT or KERASHUB).
    This global variable is used during the pre- and post-processing for correctly
    formatting model input.

    Args:
        model_type (str): Either MODEL_IMPLEMENTATION.MAXTEXT or MODEL_IMPLEMENTATION.KERASHUB
    """
    if model_type not in ModelImplementationType.list_supported_types():
        raise ValueError(
            f"{model_type} must be one of {ModelImplementationType.list_supported_types()}"
        )
    global_state.set_global_attribute("MODEL_IMPLEMENTATION", model_type)



@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float


@dataclass
class ModelConfig:
    preset_handle: str
    model_type: str = "KerasHub"  # or "MaxText"
    lora_rank: int = 16
    precision: str = "mixed_bfloat16"
    per_device_batch_size: int = 1
    seq_len: int = 1024
    optimizer: Union[
        OptimizerConfig,
        keras.Optimizer,
        "optax.GradientTransformation",
        "optax.GradientTransformationExtraArgs",
    ] = None


# TODO: Complete ModelConfig
def create_model_from_config(config: ModelConfig):
    if config.model_type == "KerasHub":
        from kithara.model.kerashub import KerasHubModel

        model = KerasHubModel.from_preset(
            config.preset_handle,
            lora_rank=config.lora_rank,
            precision=config.precision,
        )
        mask_key = "padding_mask"
        token_key = "token_ids"
    elif config.model_type == "MaxText":
        from kithara.model.maxtext import MaxTextModel

        model = MaxTextModel.from_preset(
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


def create_optimizer_from_config(config: OptimizerConfig):
    if config.name == "adamw":
        optimizer = keras.optimizers.AdamW(
            learning_rate=config.learning_rate, weight_decay=0.01
        )
    return optimizer
