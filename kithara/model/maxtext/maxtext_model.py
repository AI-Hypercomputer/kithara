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

from typing import Optional, Union, List, Dict
import numpy as np
from transformers import AutoTokenizer
from kithara.dataset.utils import initialize_tokenizer
from kithara.model.hf_compatibility import get_model_name_from_preset_handle
from kithara.model.maxtext.conversion_utils import MaxTextConversionMixin
from kithara.model.maxtext.ckpt_compatibility import (
    save_maxtext_model_in_hf_format,
    load_hf_weights_into_maxtext_model,
)
from kithara.model import (
    Model,
    set_precision,
    set_global_model_implementation_type,
    ModelImplementationType,
)


class MaxTextModel(Model, MaxTextConversionMixin):
    """
    MaxTextModel is class that represents a MaxText model via the
    Kithara.Model interface. It is a thin wrapper around the underlying
    MaxText model instance.

    Methods
    -------
    from_random: Create a randomly initialized MaxText model with the
        given configuration.
    from_preset: Create a MaxText model initialized with weights from
        HuggingFace Hub.
    generate: Generate text based on the input tokens, with an option to
        stop at specific token IDs.
    save_in_hf_format: Save the MaxText model in HuggingFace format.
    """

    @classmethod
    def from_random(
        cls,
        model_name: str,
        seq_len: int = 2048,
        per_device_batch_size: int = 1,
        precision: str = "mixed_bfloat16",
        scan_layers: bool = False,
        maxtext_config_args: Optional[dict] = None,
    ) -> "MaxTextModel":
        """Create a randomly initialized MaxText model with the given configuration.

        Args:
            model_name (str): Name of the MaxText model configuration to use.
            seq_len (int, optional): Maximum sequence length. Defaults to 2048.
            per_device_batch_size (int, optional): Batch size per device.
                Defaults to 1.
            precision (str, optional): Precision mode for computations.
                Defaults to "mixed_bfloat16".
            scan_layers (bool, optional): Whether to use scan layers for memory efficiency.
                Defaults to False.
            maxtext_config_args (Optional[dict], optional): Additional configuration arguments.
                Defaults to None.

        Returns:
            MaxTextModel: A new instance of MaxTextModel with random initialization.
        """
        set_global_model_implementation_type(ModelImplementationType.MAXTEXT)

        set_precision(precision)
        weight_dtype = cls._weight_dtype(precision)
        activation_dtype = cls._activation_dtype(precision)

        sharding_strategy, model = cls.initialize_random_maxtext_model(
            model_name,
            seq_len,
            per_device_batch_size,
            weight_dtype,
            activation_dtype,
            scan_layers,
            maxtext_config_args,
        )

        return cls(
            model,
            sharding_strategy,
            model_name=model_name,
            precision=precision,
            scan_layers=scan_layers,
        )

    @classmethod
    def from_preset(
        cls,
        preset_handle: str,
        seq_len: int = 2048,
        per_device_batch_size: int = 1,
        precision: str = "mixed_bfloat16",
        scan_layers: bool = False,
        maxtext_config_args: Optional[dict] = None,
    ) -> "MaxTextModel":
        """Create a MaxText model initialized with weights from HuggingFace Hub.

        Args:
            preset_handle (str): HuggingFace model identifier. This could be a
                HuggingFace Hub path (e.g "gs://google/gemma-2-2b), or
                a local HuggingFace checkpoint path (e.g. tmp/my_model/checkpoint), or
                a GCS HuggingFace checkpoint path (e.g. gs://bucket_name/my_model/checkpoint)
            seq_len (int): Maximum sequence length.
            per_device_batch_size (int): Batch size per device.
            precision (str, optional): Precision mode for computations.
                Defaults to "mixed_bfloat16".
            scan_layers (bool, optional): Whether to use scan layers. Defaults to False.
            maxtext_config_args (Optional[dict], optional): Additional configuration arguments.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            MaxTextModel: A new instance of MaxTextModel initialized with pretrained weights.
        """
        set_global_model_implementation_type(ModelImplementationType.MAXTEXT)

        set_precision(precision)
        weight_dtype = cls._weight_dtype(precision)
        activation_dtype = cls._activation_dtype(precision)

        model_name = get_model_name_from_preset_handle(preset_handle)
        sharding_strategy, model = cls.initialize_random_maxtext_model(
            model_name,
            seq_len,
            per_device_batch_size,
            weight_dtype,
            activation_dtype,
            scan_layers,
            maxtext_config_args,
        )
        model = load_hf_weights_into_maxtext_model(preset_handle, model, scan_layers)

        return cls(
            model,
            sharding_strategy,
            model_name=model_name,
            precision=precision,
            scan_layers=scan_layers,
        )

    def _convert_text_input_to_model_input(
        self,
        prompts: Union[str | List[str]],
        max_length: int = 100,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_handle: Optional[str] = None,
    ):
        
        # MaxText requires max length to be a multiple of 128
        max_length = ((max_length + 127) // 128) * 128

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

        B, S = input_ids.shape
        positions = np.arange(S, dtype=np.int32)[None, :]
        positions = np.repeat(positions, B, axis=0)
        return {
            "tokens": input_ids,
            "segment_ids": attention_mask,
            "positions": positions,
        }

    def _generate(
        self,
        inputs,
        **kwargs,
    ):
        """Note: Currently running inference MaxText models is slow
        since it does not perform kv caching. This should improve with integration
        with MaxEngine.
        """
        return super()._generate(
            inputs, **kwargs, tokens_key="tokens", padding_mask_key="segment_ids"
        )

    def save_in_hf_format(
        self, output_dir: str, dtype: str = "auto", parallel_threads=8
    ):
        """Save the model in HuggingFace format, including the model configuration file (`config.json`),
            the model weights file (`model.safetensors` for models smaller than
            `DEFAULT_MAX_SHARD_SIZE` and `model-x-of-x.safetensors` for larger models),
            and the safe tensors index file (`model.safetensors.index.json`).

        Args:
            output_dir (str): Directory path where the model should be saved.
                Directory could be local or a Google cloud storage path, and
                will be created if it doesn't exist.
            dtype (str, optional): Data type for saved weights. Defaults to "auto".
            parallel_threads (int, optional): Number of parallel threads to use for saving.
                Defaults to 8. Make sure the local system has at least
                `parallel_threads * DEFAULT_MAX_SHARD_SIZE` free disk space,
                as each thread will maintain a local cache of size `DEFAULT_MAX_SHARD_SIZE`.
        """
        save_maxtext_model_in_hf_format(
            self, output_dir, dtype=dtype, parallel_threads=parallel_threads
        )
