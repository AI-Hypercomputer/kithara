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

def GEMMA2_HF_WEIGHTS_TO_SHAPE_MAPPING(config):
    """Returns mapping between HuggingFace weights path and weights shape.

    Args:
        config (dict): Model configuration dictionary, defined in `model_configs.py`

    Returns:
        dict: A mapping where:
            - Keys are HuggingFace model parameter paths
            - Values are parameter shape as a List
    """

    mapping = {
        "model.embed_tokens.weight": [config["vocab_size"], config["hidden_size"]],
        "model.norm.weight": [config["hidden_size"]],
    }
    for layer_idx in range(config["num_hidden_layers"]):
        layer_mapping = {
            f"model.layers.{layer_idx}.input_layernorm.weight": [config["hidden_size"]],
            f"model.layers.{layer_idx}.mlp.down_proj.weight": [
                config["hidden_size"],
                config["intermediate_size"],
            ],
            f"model.layers.{layer_idx}.mlp.up_proj.weight": [
                config["intermediate_size"],
                config["hidden_size"],
            ],
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": [
                config["intermediate_size"],
                config["hidden_size"],
            ],
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": [
                config["hidden_size"]
            ],
            f"model.layers.{layer_idx}.post_feedforward_layernorm.weight": [
                config["hidden_size"]
            ],
            f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight": [
                config["hidden_size"]
            ],
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": [
                config["num_key_value_heads"] * config["head_dim"],
                config["hidden_size"],
            ],
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": [
                config["hidden_size"],
                config["num_attention_heads"] * config["head_dim"],
            ],
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": [
                config["num_attention_heads"] * config["head_dim"],
                config["hidden_size"],
            ],
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": [
                config["num_key_value_heads"] * config["head_dim"],
                config["hidden_size"],
            ],
        }
        mapping = {**mapping, **layer_mapping}
    return mapping


def LLAMA31_HF_WEIGHTS_TO_SHAPE_MAPPING(config):
    """Returns mapping between HuggingFace weights path and weights shape.

    Args:
        config (dict): Model configuration dictionary, defined in `model_configs.py`

    Returns:
        dict: A mapping where:
            - Keys are HuggingFace model parameter paths
            - Values are parameter shape as a List
    """

    mapping = {
        "model.embed_tokens.weight": [config["vocab_size"], config["hidden_size"]],
        "model.norm.weight": [config["hidden_size"]],
        "lm_head.weight": [config["vocab_size"], config["hidden_size"]]
    }
    for layer_idx in range(config["num_hidden_layers"]):
        layer_mapping = {
            f"model.layers.{layer_idx}.input_layernorm.weight": [config["hidden_size"]],
            f"model.layers.{layer_idx}.mlp.down_proj.weight": [
                config["hidden_size"],
                config["intermediate_size"],
            ],
            f"model.layers.{layer_idx}.mlp.up_proj.weight": [
                config["intermediate_size"],
                config["hidden_size"],
            ],
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": [
                config["intermediate_size"],
                config["hidden_size"],
            ],
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": [
                config["hidden_size"]
            ],
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": [
                config["num_key_value_heads"] * config["head_dim"],
                config["hidden_size"],
            ],
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": [
                config["hidden_size"],
                config["num_attention_heads"] * config["head_dim"],
            ],
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": [
                config["num_attention_heads"] * config["head_dim"],
                config["hidden_size"],
            ],
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": [
                config["num_key_value_heads"] * config["head_dim"],
                config["hidden_size"],
            ],
        }
        mapping = {**mapping, **layer_mapping}
    return mapping

def QWEN25_HF_WEIGHTS_TO_SHAPE_MAPPING(config):
    """Returns mapping between HuggingFace weights path and weights shape.

    Args:
        config (dict): Model configuration dictionary, defined in `qwen_model_configs.py`

    Returns:
        dict: A mapping where:
            - Keys are HuggingFace model parameter paths
            - Values are parameter shape as a List
    """
    vocab_size, hidden_size = config["vocab_size"], config["hidden_size"]
    num_key_value_heads = config["num_key_value_heads"]
    intermediate_size = config["intermediate_size"]

    #  Differentiate .5B and others
    multihead_size = 128 if hidden_size == 896 else num_key_value_heads * 128

    mapping = {
        "model.embed_tokens.weight": [vocab_size, hidden_size],
    }

    for layer in range(config["num_hidden_layers"]):
        mapping |= {
            # multi-head attention layer
            f"model.layers.{layer}.self_attn.q_proj.weight": [hidden_size, hidden_size],
            f"model.layers.{layer}.self_attn.k_proj.weight": [multihead_size, hidden_size],
            f"model.layers.{layer}.self_attn.v_proj.weight": [multihead_size, hidden_size],
            f"model.layers.{layer}.self_attn.o_proj.weight": [hidden_size, hidden_size],

            #mlp layer
            f"model.layers.{layer}.mlp.gate_proj.weight": [intermediate_size, hidden_size],
            f"model.layers.{layer}.mlp.up_proj.weight": [intermediate_size, hidden_size],
            f"model.layers.{layer}.mlp.down_proj.weight": [hidden_size, intermediate_size],
            # f"model.layers.{layer}.mlp.act_fn.weight": [],  //SiLU does not have weight

            f"model.layers.{layer}.input_layernorm.weight": [hidden_size],
            f"model.layers.{layer}.post_attention_layernorm.weight": [hidden_size],
        }

    mapping |= {
        "model.norm.weight": [hidden_size],
        #  "model.rotary_emb": []  //Qwen2RotaryEmbedding does not have weight
        "lm_head.weight": [vocab_size, hidden_size]
    }
    return mapping

SHAPE_MAPPING = {
    supported_models.GEMMA2_2B: GEMMA2_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.GEMMA2_9B: GEMMA2_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.GEMMA2_27B: GEMMA2_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.LLAMA31_8B: LLAMA31_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.LLAMA31_70B: LLAMA31_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.LLAMA31_405B: LLAMA31_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.LLAMA32_1B: LLAMA31_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.LLAMA32_3B: LLAMA31_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.QWEN25_D5B: QWEN25_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.QWEN25_1D5B: QWEN25_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.QWEN25_3B: QWEN25_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.QWEN25_7B: QWEN25_HF_WEIGHTS_TO_SHAPE_MAPPING,
    supported_models.QWEN25_14B: QWEN25_HF_WEIGHTS_TO_SHAPE_MAPPING,
}
