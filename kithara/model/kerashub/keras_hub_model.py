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

from typing import Optional, Dict, Union, List
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

    def _pad_tokens_to_max_length(self, tokens, max_length):
        """
        Pad each sequence in the list of token sequences to max_length.
        
        Args:
            tokens: List of numpy arrays, where each array is a sequence of token IDs
            max_length: The target length to pad sequences to
            
        Returns:
            Dict containing padded token_ids and corresponding padding_mask
        """
        # Initialize arrays for padded tokens and attention masks
        batch_size = len(tokens)
        padded_tokens = np.zeros((batch_size, max_length), dtype=np.int64)
        padding_mask = np.zeros((batch_size, max_length), dtype=np.int64)
        
        # Fill the arrays with the tokens and create corresponding masks
        for i, seq in enumerate(tokens):
            seq_len = min(len(seq), max_length)
            padded_tokens[i, :seq_len] = seq[:seq_len]
            padding_mask[i, :seq_len] = 1
        
        return {
            "token_ids": padded_tokens,
            "padding_mask": padding_mask,
        }
    
    def _convert_text_input_to_model_input(
        self,
        prompts: Union[str | List[str]],
        tokenizer:"AutoTokenizer",
        max_length: int,
    ):
        assert (
            max_length is not None
        ), "max_length must be provided to generate() when inputs are strings."

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

    def _generate(
        self,
        inputs: Union[List[str], List[np.ndarray]],
        max_length: int = None,
        stop_token_ids: Optional[List] = None,
        strip_prompt: str = False,
        tokenizer: Optional["AutoTokenizer"] = None,
        **kwargs,
    ) -> List[List[int]]:
        """Generate tokens using the model. This function falls back to KerasHub model's
        native generation function: 
        https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/causal_lm.py

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
            list[np.ndarray]: Generated token IDs (numpy.ndarray) for each prompt        
        """

        if isinstance(inputs[0], str):
            inputs = self._convert_text_input_to_model_input(
                inputs, tokenizer, max_length
            )
        else:
            inputs = self._pad_tokens_to_max_length(inputs, max_length)

        # stop_token_ids cannot be an empty list
        stop_token_ids = stop_token_ids if stop_token_ids else None

        tokens = self.model.generate(
            inputs,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
        )
                
        results = []
        for i in range(len(inputs["token_ids"])):
            is_token = tokens["padding_mask"][i, :] == True
            generated_tokens = tokens["token_ids"][i, :][is_token]
            results.append(generated_tokens.tolist())
        return results

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
                Directory could be a local folder (e.g. "foldername/"), 
                HuggingFaceHub repo (e.g. "hf://your_hf_id/repo_name") or a 
                Google cloud storage path (e.g. "gs://your_bucket/folder_name), 
                and will be created if it doesn't exist. 
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
    
    def disable_lora(self):
        target_names = ["query_dense", "value_dense", "query", "value"]
        for layer in self.model.backbone._flatten_layers(include_self=False):
            if layer.name in target_names and hasattr(layer, "enable_lora"):
                layer.lora_enabled = False
                layer.lora_enabled = False


    def enable_lora(self):
        target_names = ["query_dense", "value_dense", "query", "value"]
        for layer in self.model.backbone._flatten_layers(include_self=False):
            if layer.name in target_names and hasattr(layer, "enable_lora"):
                layer.lora_enabled = True
                layer.lora_enabled = True

