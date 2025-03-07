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

from kithara.distributed.sharding._mesh import Axis

LLAMA_FSDP = {
    ".*token_embedding.embeddings.*": (None, Axis.FSDP),
    ".*token_embedding.reverse_embeddings.*": (Axis.FSDP, None),
    ".*transformer_layer.*self_attention.*(query|key|value).kernel*": (
        None,
        Axis.FSDP,
    ),
    ".*transformer_layer.*self_attention.attention_output.kernel*": (
        None,
        None,
        Axis.FSDP,
    ),
    ".*transformer_layer.*feedforward_gate_dense.kernel*": (None, Axis.FSDP),
    ".*transformer_layer.*feedforward_intermediate_dense.kernel*": (None, Axis.FSDP),
    ".*transformer_layer.*feedforward_output_dense.kernel*": (None, Axis.FSDP),
    # Lora layers
    ".*transformer_layer.*self_attention.*(query|key|value).lora_kernel.*": (
        None,
        Axis.FSDP,
    ),
}

LLAMA_LAYOUT = {
    "fsdp": LLAMA_FSDP,
}
