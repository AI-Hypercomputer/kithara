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
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Union, Dict
from keras.src.backend.common import global_state
from keras.distribution import set_distribution
from kithara.distributed.sharding import ShardingStrategy
from kithara.distributed.sharding.utils import (
    print_elements_that_are_unsharded_and_large_in_pytree,
)
from keras.src.backend.common import global_state
from kithara.distributed.sharding._mesh import Axis
from jax.experimental import multihost_utils
import time
from enum import Enum
from transformers import AutoTokenizer
from kithara.dataset.utils import initialize_tokenizer


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

    def make_generate_step(self):
        """Create a JIT-compiled function for single-step token generation.

        Returns:
            function: Compiled function that performs one step of token generation.
        """

        def fn(trainable_variables, non_trainable_variables, x):
            logits, non_trainable_variables = self.model.stateless_call(
                trainable_variables, non_trainable_variables, x
            )
            return logits

        return jax.jit(fn)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def _convert_text_input_to_model_input(
        self,
        prompts: Union[str | List[str]],
        max_length: int = 100,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_handle: Optional[str] = None,
    ):
        assert (tokenizer is not None) or (tokenizer_handle is not None), (
            "Cannot convert text input to model input because tokenizer and tokenizer_handle"
            " are both not specified."
        )

        assert (
            max_length is not None
        ), "max_length must be provided to generate() when inputs are strings."

        tokenizer = initialize_tokenizer(tokenizer_handle, tokenizer)

        tokens: Dict[str, np.ndarray] = tokenizer(
            prompts,
            max_length=max_length,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="np",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        return {
            "token_ids": input_ids,
            "padding_mask": attention_mask,
        }

    def generate(
        self,
        inputs: Union[str | List[str] | Dict[str, np.ndarray]],
        max_length: int = 100,
        stop_token_ids: Union[str | List[int]] = "auto",
        strip_prompt: bool = False,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_handle: Optional[str] = None,
        return_decoded: bool = True,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[str] | Dict[str, np.ndarray]]:
        """Generate text tokens using the model.
        Args:
            inputs (list, dict): A single string, a list of strings, or a
                dictionary with tokens as expected by the underlying model
                during the forward pass. If strings are provided, one of
                `tokenizer` and `tokenizer_handle` must be provided.
            max_length (int, optional): Maximum total sequence length
                (prompt + generated tokens). If `tokenizer` and `tokenizer_handle`
                are `None`, `inputs` should be should be padded to the desired
                maximum length and this argument will be ignored. When `inputs` is
                string, this value must be provided.
            stop_token_ids (List[int], optional): List of token IDs that stop
                generation. Defaults to "auto", which extracts the end token id
                from the tokenizer.
            strip_prompt (bool, optional): If True, returns only the generated
                tokens without the input prompt. If False, returns the full sequence
                including the prompt. Defaults to False.
            return_decoded (bool, optional): If Ture, returns the decoded text using
                the tokenizer, otherwise return the predicted tokens. Defautl to True.
                This option must be set to False if no tokenizer is provided.
        Returns:
            A list of string if input is text, or a dictionary containing the following
                keys if the input is tokens.
                - 'token_ids': Generated token IDs (numpy.ndarray) of shape [B, S]
                - 'padding_mask': Attention mask for the generated sequence (numpy.ndarray) of shape [B, S]

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

        if isinstance(inputs, str) or isinstance(inputs, list) or return_decoded:
            assert (tokenizer or tokenizer_handle) is not None

        if isinstance(inputs, str) or isinstance(inputs, list):
            inputs = self._convert_text_input_to_model_input(
                inputs, max_length, tokenizer, tokenizer_handle
            )

        if stop_token_ids == "auto":
            stop_token_ids = []
            if tokenizer or tokenizer_handle:
                tokenizer = initialize_tokenizer(tokenizer_handle, tokenizer)

                token_attributes = [
                    "end_token_id",
                    "eos_token_id",
                    "end_token2_id",
                    "eos_token2_id",
                ]

                for attr in token_attributes:
                    if hasattr(tokenizer, attr):
                        stop_token_ids.append(getattr(tokenizer, attr))

        tokens: Dict[str, Any] = self._generate(
            inputs,
            max_length=max_length,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
        )
        if return_decoded:
            tokenizer = initialize_tokenizer(tokenizer_handle, tokenizer)

            text = [
                tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
                for token_ids in tokens["token_ids"]
            ]
            return text
        return tokens

    def _generate(
        self,
        inputs: Dict[str, np.ndarray],
        max_length: int = None,
        stop_token_ids: Optional[List] = None,
        strip_prompt: str = False,
        tokens_key: str = "token_ids",
        padding_mask_key: str = "padding_mask",
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Generate tokens using the model.

        Args:
            tokens_key (str, optional): Key in the inputs dictionary for token IDs.
                Defaults to "token_ids".
            padding_mask_key (str, optional): Key in the inputs dictionary for padding
                mask. Defaults to "padding_mask".
            For the rest of the args, please refer to `generate()`.

        Returns:
            dict: Dictionary containing:
                - 'token_ids': Generated token IDs (numpy.ndarray) of shape [B, S]
                - 'padding_mask': Attention mask for the generated sequence (numpy.ndarray) of shape [B, S]
        """
        if stop_token_ids is None:
            stop_token_ids = []
        if max_length < 1:
            raise ValueError("Please either a positive max_length.")
        if max_length == None and len(stop_token_ids) == 0:
            raise ValueError("Please either specify max_length or stop_token_ids.")

        jitted_generate_fn = self.make_generate_step()
        batch_size = inputs[tokens_key].shape[0]

        # Pad batch to be a multiple of fsdp dimension
        mesh = self.sharding_strategy.data_sharding.mesh
        devices_in_data_fsdp = (
            mesh.shape[Axis.FSDP] if Axis.FSDP in mesh.shape else mesh.shape["fsdp"]
        )
        remainder = batch_size % devices_in_data_fsdp
        if remainder != 0:
            pad_size = devices_in_data_fsdp - remainder
            for key in inputs.keys():
                inputs[key] = np.pad(
                    inputs[key],
                    ((0, pad_size), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

        def next_token(current_inputs):
            current_inputs = jax.device_put(
                current_inputs, self.sharding_strategy.data_sharding
            )
            logits = jitted_generate_fn(
                [v.value for v in self.model.trainable_variables],
                [v.value for v in self.model.non_trainable_variables],
                current_inputs,
            )
            jax.block_until_ready(logits)
            return logits

        tokens = inputs[tokens_key]
        segment_ids = inputs[padding_mask_key]

        # Calculate initial number of tokens (where segment_ids == 1)
        num_tokens = int(np.sum(segment_ids[0] == 1))
        seq_len = segment_ids.shape[1]

        # Calculate how many tokens we can/should generate
        max_length = min(seq_len, max_length) if max_length else seq_len
        generate_steps = max_length - num_tokens

        # Track which sequences have reached EOS
        reached_eos = [False for _ in range(batch_size)]

        for s in range(generate_steps):
            current_inputs = {
                **inputs,
                tokens_key: tokens,
                padding_mask_key: segment_ids,
            }

            # Get next token predictions
            logits = next_token(current_inputs)

            next_token_logits = logits[:, num_tokens - 1, :]
            next_tokens = keras.ops.argmax(next_token_logits, axis=-1)
            next_tokens = multihost_utils.process_allgather(next_tokens)
            # Update the tokens array with predictions
            tokens[:, num_tokens] = next_tokens

            # Update attention mask (segment_ids)
            segment_ids = np.roll(segment_ids, 1, axis=1)
            segment_ids[:, 0] = 1

            # Increment number of tokens
            num_tokens += 1
            # Check for EOS tokens
            for i, token in enumerate(next_tokens.flatten()[:batch_size]):
                if token in stop_token_ids:
                    reached_eos[i] = True
            if np.all(reached_eos):
                generate_steps = s + 1
                break

        token_ids = tokens[:batch_size, :num_tokens]
        padding_mask = segment_ids[:batch_size, :num_tokens]
        if strip_prompt:
            token_ids = tokens[:batch_size, num_tokens - generate_steps : num_tokens]
            padding_mask = segment_ids[
                :batch_size, num_tokens - generate_steps : num_tokens
            ]

        return {
            "padding_mask": padding_mask,
            "token_ids": token_ids,
        }


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
