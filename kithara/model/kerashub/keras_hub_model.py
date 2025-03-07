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

from typing import Optional, Dict
import numpy as np
from keras_hub.models import CausalLM
from kithara.distributed.sharding import ShardingStrategy, PredefinedShardingStrategy
from kithara.model.hf_compatibility import get_model_name_from_preset_handle
from kithara.model.model import (
    Model,
    set_precision,
    set_global_sharding_strategy,
    set_global_model_implementation_type,
    ModelImplementationType,
)
from kithara.model.kerashub.ckpt_compatibility.to_huggingface import (
    save_kerashub_model_in_hf_format,
)


class KerasHubModel(Model):
    """
    A Kithara model wrapper for KerasHub models.

    Attributes:
        preset_handle (str): Model identifier, e.g., "hf://google/gemma-2-2b".
        lora_rank (Optional[int]): Rank for LoRA adaptation (disabled if None), applied to q_proj and v_proj
        sharding_strategy(kithara.ShardingStrategy): Strategy used for distributing model, optimizer,
            and data tensors. E.g. `kithara.PredefinedShardingStrategy("fsdp", "gemma2-2b")`.
        precision (str): Default is "mixed_bfloat16". Supported policies include "float32", "float16", "bfloat16",
            "mixed_float16", and "mixed_bfloat16". Mixed precision policies load model weight in float32
            and casts activations to the specified dtype.

    Example Usage:
        model = KerasHubModel.from_preset("hf://google/gemma-2-2b", lora_rank=4)
    """

    @classmethod
    def from_preset(
        cls,
        preset_handle: str,
        lora_rank: Optional[int] = None,
        precision: str = "mixed_bfloat16",
        sharding_strategy: Optional[ShardingStrategy] = None,
        **kwargs,
    ) -> "KerasHubModel":
        """Load a KerasHub model, optionally apply LoRA, and configure precision and sharding.

        Args:
            preset_handle (str): Identifier for the model preset. This can be:
                - A built-in KerasHub preset identifier (e.g., `"bert_base_en"`).
                - A Kaggle Models handle (e.g., `"kaggle://user/bert/keras/bert_base_en"`).
                - A Hugging Face handle (e.g., `"hf://user/bert_base_en"`).
                - A local directory path (e.g., `"./bert_base_en"`).
            lora_rank (Optional[int]): Rank for LoRA adaptation. If None, LoRA is disabled.
                Defaults to None. When enabled, LoRA is applied to the `q_proj` and `v_proj` layers.
            precision (str): Precision policy for the model. Defaults to "mixed_bfloat16".
                Supported options include: "float32", "float16", "bfloat16", "mixed_float16",
                and "mixed_bfloat16". Mixed precision policies load weights in float32 and cast
                activations to the specified dtype.
            sharding_strategy (Optional[ShardingStrategy]): Strategy for distributing model parameters,
                optimizer states, and data tensors. If None, tensors will be sharded using FSDP.
                You can use `kithara.ShardingStrategy` to configure custom sharding strategies.

        Returns:
            KerasHubModel: An instance of the `KerasHubModel` class.

        Example:
            ```
            model = KerasHubModel.from_preset(
                "hf://google/gemma-2-2b",
                lora_rank=4
            )
            ```
        """

        model_name = get_model_name_from_preset_handle(preset_handle)

        if sharding_strategy is None:
            sharding_strategy = PredefinedShardingStrategy(
                parallelism="fsdp", model=model_name
            )

        set_global_model_implementation_type(ModelImplementationType.KERASHUB)
        set_precision(precision)
        set_global_sharding_strategy(sharding_strategy)

        model = CausalLM.from_preset(preset_handle, preprocessor=None, **kwargs)
        if lora_rank:
            model.backbone.enable_lora(rank=lora_rank)

        return cls(
            model,
            model_name=model_name,
            sharding_strategy=sharding_strategy,
            precision=precision,
            lora_rank=lora_rank,
        )

    def _generate(
        self,
        model_input,
        stop_token_ids=None,
        strip_prompt=False,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Fall back to https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/causal_lm.py"""

        # stop_token_ids cannot be an empty list
        stop_token_ids = stop_token_ids if stop_token_ids else None

        tokens = self.model.generate(
            model_input,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
        )

        # Return output but first stripped prompt
        is_token = tokens["padding_mask"] == True
        B, _ = tokens["token_ids"].shape
        return {
            "token_ids": tokens["token_ids"][is_token][None, :].reshape(B, -1),
            "padding_mask": tokens["padding_mask"][is_token][None, :].reshape(B, -1),
        }

    def save_in_hf_format(
        self,
        output_dir: str,
        dtype: str = "auto",
        only_save_adapters=False,
        save_adapters_separately=False,
        parallel_threads=8,
    ):
        """Save the model in HuggingFace format, including the model configuration file (`config.json`),
            the model weights file (`model.safetensors` for models smaller than
            `DEFAULT_MAX_SHARD_SIZE` and `model-x-of-x.safetensors` for larger models),
            and the safe tensors index file (`model.safetensors.index.json`).

        Args:
            output_dir (str): Directory path where the model should be saved.
                Directory could be local or a Google cloud storage path, and will be created if
                it doesn't exist.
            dtype (str, optional): Data type for saved weights. Defaults to "auto".
            only_save_adapters (bool): If set to True, only adapter weights will be saved. If
                set to False, both base model weights and adapter weights will be saved. Default
                to False.
            save_adapters_separately (bool): If set to False, adapter weights will be merged with base model.
                If set to True, adapter weights will be saved separately in HuggingFace's peft format.
                Default to False.
            parallel_threads (int, optional): Number of parallel threads to use for saving.
                Defaults to 8. Make sure the local system has at least
                `parallel_threads * DEFAULT_MAX_SHARD_SIZE` free disk space,
                as each thread will maintain a local cache of size `DEFAULT_MAX_SHARD_SIZE`.
        """
        save_kerashub_model_in_hf_format(
            self,
            output_dir,
            dtype=dtype,
            only_save_adapters=only_save_adapters,
            save_adapters_separately=save_adapters_separately,
            parallel_threads=parallel_threads,
        )
