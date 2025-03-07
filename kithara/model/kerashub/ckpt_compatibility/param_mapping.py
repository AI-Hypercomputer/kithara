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

from kithara.model import supported_models
import jax
import numpy as np


def GEMMA2_KERASHUB_TO_HF_PARAM_MAPPING(config):
    """Returns mapping between KerasHub and HuggingFace Gemma2 weight paths.

    Args:
        config (dict): Model configuration dictionary containing at least 'num_hidden_layers'.

    Returns:
        dict: A mapping where:
            - Keys are KerasHubModel weights paths obtained from `model.weights`
            - Values are either:
                - Single strings (HF parameter path) for unscanned parameters
                - Lists of strings (HF parameter paths) for stacked layers when scan_layers=True

    How to obtain this mapping for a new model:
        ```
        model = KerasHubModel.from_preset(
            "hf://google/gemma-2-2b"
        )

        # print out all keras model weights

        for v in model.weights:
            print(v.path)

        # print out all modules of the huggingface model

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

        print(model.state_dict().keys())

        # Manually create the mapping between these weight keys.
        ```
    """

    nlayers = config["num_hidden_layers"]
    mapping = {
        "token_embedding/embeddings": "model.embed_tokens.weight",
        "final_normalization/scale": "model.norm.weight",
    }
    for layer_idx in range(nlayers):
        layer_mapping = {
            f"decoder_block_{layer_idx}/pre_attention_norm/scale": f"model.layers.{layer_idx}.input_layernorm.weight",
            f"decoder_block_{layer_idx}/ffw_linear/kernel": f"model.layers.{layer_idx}.mlp.down_proj.weight",
            f"decoder_block_{layer_idx}/ffw_gating_2/kernel": f"model.layers.{layer_idx}.mlp.up_proj.weight",
            f"decoder_block_{layer_idx}/ffw_gating/kernel": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"decoder_block_{layer_idx}/post_attention_norm/scale": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            f"decoder_block_{layer_idx}/post_ffw_norm/scale": f"model.layers.{layer_idx}.post_feedforward_layernorm.weight",
            f"decoder_block_{layer_idx}/pre_ffw_norm/scale": f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight",
            f"decoder_block_{layer_idx}/attention/key/kernel": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"decoder_block_{layer_idx}/attention/attention_output/kernel": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            f"decoder_block_{layer_idx}/attention/query/kernel": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"decoder_block_{layer_idx}/attention/value/kernel": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        }
        mapping = {**mapping, **layer_mapping}
    return mapping


def GEMMA2_KERASHUB_TO_HF_PARAM_HOOK_FN(config):
    """Creates parameter transformation functions for converting between KerasHub and
    HuggingFace formats.

    This function generates a mapping of transformation functions that handle the necessary
    conversions between KerasHub and HuggingFace parameter formats, including operations like
    padding, reshaping, and scaling.

    Args:
        config (dict): Model configuration dictionary that must contain:
            - num_hidden_layers (int): Number of layers in the model
            - head_dim (int): Dimension of attention heads
            - hidden_size (int): Model's hidden dimension size

        scan_layers (bool, optional): Controls the output format for layer parameters:
            - True: Returns transformation functions for batched layer parameters
            - False: Returns transformation functions for individual layer parameters
            Defaults to False.

        saving_to_hf (bool, optional): Determines the direction of transformation:
            - True: KerasHub → HuggingFace conversion
            - False: HuggingFace → KerasHub conversion
            Defaults to False.

    Returns:
        dict: Parameter transformation mapping where:
            - Keys: KerasHub parameter names (str)
            - Values: Either:
                - callable: Single transformation function
                - list[callable]: List of transformation functions to be applied in sequence

    Transformation Details:
        The function handles several types of parameter transformations:
        1. Embedding layer padding:
            - HF shape: [256000, d_model]
            - KerasHub shape: [256128, d_model] (padded for performance)
        2. Layer normalization scaling:
            - Adds/subtracts 1.0 depending on direction
        3. Attention query scaling:
            - Scales by sqrt(head_dim) or its inverse

        4. Kernel reshaping:
            - Handles dimension transposition and reshaping between formats
    """
    nlayers = config["num_hidden_layers"]

    def reshape_embedding_layer(input_tensor, target_shape):
        return input_tensor[: target_shape[0], : target_shape[1]]

    def transpose(input_tensor, target_shape):
        return input_tensor.transpose()

    def out_proj_reshape(input_tensor, target_shape):
        target_shape = (target_shape[1], target_shape[0])
        return input_tensor.reshape(target_shape).transpose()

    def qkv_proj_reshape(input_tensor, target_shape):
        return input_tensor.transpose(0, 2, 1).reshape(target_shape)

    mapping = {
        "token_embedding/embeddings": reshape_embedding_layer,
    }
    for layer_idx in range(nlayers):
        mapping = {
            **mapping,
            f"decoder_block_{layer_idx}/ffw_linear/kernel": transpose,
            f"decoder_block_{layer_idx}/ffw_gating_2/kernel": transpose,
            f"decoder_block_{layer_idx}/ffw_gating/kernel": transpose,
            f"decoder_block_{layer_idx}/attention/key/kernel": qkv_proj_reshape,
            f"decoder_block_{layer_idx}/attention/attention_output/kernel": out_proj_reshape,
            f"decoder_block_{layer_idx}/attention/query/kernel": qkv_proj_reshape,
            f"decoder_block_{layer_idx}/attention/value/kernel": qkv_proj_reshape,
        }
    return mapping


def LLAMA31_KERASHUB_TO_HF_PARAM_MAPPING(config):
    """Returns mapping between KerasHub and HuggingFace Gemma2 weight paths.

    Args:
        config (dict): Model configuration dictionary containing at least 'num_hidden_layers'.

    Returns:
        dict: A mapping where:
            - Keys are KerasHubModel weights paths obtained from `model.weights`
            - Values are either:
                - Single strings (HF parameter path) for unscanned parameters
                - Lists of strings (HF parameter paths) for stacked layers when scan_layers=True

    How to obtain this mapping for a new model:
        ```
        model = KerasHubModel.from_preset(
            "hf://google/gemma-2-2b",
            sharding_strategy=PredefinedShardingStrategy(
                parallelism="fsdp", model=config["model"]
            ),
        )

        # print out all keras model weights

        for v in model.weights:
            print(v.path)

        # print out all modules of the huggingface model

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

        print(model.state_dict().keys())

        # Manually create the mapping between these weight keys.
        ```
    """

    nlayers = config["num_hidden_layers"]
    layer_mapping = {
        "token_embedding/embeddings": "model.embed_tokens.weight",
        "token_embedding/reverse_embeddings": "lm_head.weight",
        "sequence_output_layernorm/scale": "model.norm.weight",
    }

    for layer_idx in range(nlayers):
        layer_mapping.update(
            {
                f"transformer_layer_{layer_idx}/self_attention/query/kernel": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                f"transformer_layer_{layer_idx}/self_attention/key/kernel": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                f"transformer_layer_{layer_idx}/self_attention/value/kernel": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                f"transformer_layer_{layer_idx}/self_attention/attention_output/kernel": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                f"transformer_layer_{layer_idx}/feedforward_intermediate_dense/kernel": f"model.layers.{layer_idx}.mlp.up_proj.weight",
                f"transformer_layer_{layer_idx}/feedforward_gate_dense/kernel": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                f"transformer_layer_{layer_idx}/feedforward_output_dense/kernel": f"model.layers.{layer_idx}.mlp.down_proj.weight",
                f"transformer_layer_{layer_idx}/self_attention_layernorm/scale": f"model.layers.{layer_idx}.input_layernorm.weight",
                f"transformer_layer_{layer_idx}/feedforward_layernorm/scale": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            }
        )
    return layer_mapping


def LLAMA31_KERASHUB_TO_HF_PARAM_HOOK_FN(config):
    """Creates parameter transformation functions for converting between KerasHub and
    HuggingFace formats.

    This function generates a mapping of transformation functions that handle the necessary
    conversions between KerasHub and HuggingFace parameter formats, including operations like
    padding, reshaping, and scaling.

    Args:
        config (dict): Model configuration dictionary that must contain:
            - num_hidden_layers (int): Number of layers in the model
            - head_dim (int): Dimension of attention heads
            - hidden_size (int): Model's hidden dimension size

        scan_layers (bool, optional): Controls the output format for layer parameters:
            - True: Returns transformation functions for batched layer parameters
            - False: Returns transformation functions for individual layer parameters
            Defaults to False.

        saving_to_hf (bool, optional): Determines the direction of transformation:
            - True: KerasHub → HuggingFace conversion
            - False: HuggingFace → KerasHub conversion
            Defaults to False.

    Returns:
        dict: Parameter transformation mapping where:
            - Keys: KerasHub parameter names (str)
            - Values: Either:
                - callable: Single transformation function
                - list[callable]: List of transformation functions to be applied in sequence

    Transformation Details:
        The function handles several types of parameter transformations:
        1. Embedding layer padding:
            - HF shape: [256000, d_model]
            - KerasHub shape: [256128, d_model] (padded for performance)
        2. Layer normalization scaling:
            - Adds/subtracts 1.0 depending on direction
        3. Attention query scaling:
            - Scales by sqrt(head_dim) or its inverse

        4. Kernel reshaping:
            - Handles dimension transposition and reshaping between formats
    """
    nlayers = config["num_hidden_layers"]

    def reshape_kernel(input_tensor, target_shape):
        flipped_target_shape = np.flip(np.array(target_shape))
        return input_tensor.reshape(flipped_target_shape).transpose()

    def transpose(input_tensor, target_shape):
        return input_tensor.transpose()

    mapping = {}
    mapping = {
        "token_embedding/reverse_embeddings": transpose,
    }
    for layer_idx in range(nlayers):
        mapping = {
            **mapping,
            f"transformer_layer_{layer_idx}/self_attention/query/kernel": [
                reshape_kernel,
            ],
            f"transformer_layer_{layer_idx}/self_attention/key/kernel": [
                reshape_kernel,
            ],
            f"transformer_layer_{layer_idx}/self_attention/key/kernel": reshape_kernel,
            f"transformer_layer_{layer_idx}/self_attention/value/kernel": reshape_kernel,
            f"transformer_layer_{layer_idx}/self_attention/attention_output/kernel": reshape_kernel,
            f"transformer_layer_{layer_idx}/feedforward_intermediate_dense/kernel": reshape_kernel,
            f"transformer_layer_{layer_idx}/feedforward_gate_dense/kernel": reshape_kernel,
            f"transformer_layer_{layer_idx}/feedforward_output_dense/kernel": reshape_kernel,
            f"transformer_layer_{layer_idx}/self_attention_layernorm/scale": reshape_kernel,
            f"transformer_layer_{layer_idx}/feedforward_layernorm/scale": reshape_kernel,
        }
    return mapping


PARAM_MAPPING = {
    supported_models.GEMMA2_2B: GEMMA2_KERASHUB_TO_HF_PARAM_MAPPING,
    supported_models.GEMMA2_9B: GEMMA2_KERASHUB_TO_HF_PARAM_MAPPING,
    supported_models.GEMMA2_27B: GEMMA2_KERASHUB_TO_HF_PARAM_MAPPING,
    supported_models.LLAMA31_8B: LLAMA31_KERASHUB_TO_HF_PARAM_MAPPING,
    supported_models.LLAMA31_70B: LLAMA31_KERASHUB_TO_HF_PARAM_MAPPING,
    supported_models.LLAMA31_405B: LLAMA31_KERASHUB_TO_HF_PARAM_MAPPING,
    supported_models.LLAMA32_3B: LLAMA31_KERASHUB_TO_HF_PARAM_MAPPING,
    supported_models.LLAMA32_1B: LLAMA31_KERASHUB_TO_HF_PARAM_MAPPING,
}

HOOK_FNS = {
    supported_models.GEMMA2_2B: GEMMA2_KERASHUB_TO_HF_PARAM_HOOK_FN,
    supported_models.GEMMA2_9B: GEMMA2_KERASHUB_TO_HF_PARAM_HOOK_FN,
    supported_models.GEMMA2_27B: GEMMA2_KERASHUB_TO_HF_PARAM_HOOK_FN,
    supported_models.LLAMA31_8B: LLAMA31_KERASHUB_TO_HF_PARAM_HOOK_FN,
    supported_models.LLAMA31_70B: LLAMA31_KERASHUB_TO_HF_PARAM_HOOK_FN,
    supported_models.LLAMA31_405B: LLAMA31_KERASHUB_TO_HF_PARAM_HOOK_FN,
    supported_models.LLAMA32_3B: LLAMA31_KERASHUB_TO_HF_PARAM_HOOK_FN,
    supported_models.LLAMA32_1B: LLAMA31_KERASHUB_TO_HF_PARAM_HOOK_FN,
}

LORA_A_SUFFIX = "lora_kernel_a"
LORA_B_SUFFIX = "lora_kernel_b"

LORA_BASE_SUFFIX = "kernel"

# HF LORA A shape is [lora_r, dim1]
HF_LORA_A_SUFFIX = ".lora_A.weight"
# HF LORA B Shape is [dim2, lora_r]
HF_LORA_B_SUFFIX = ".lora_B.weight"
