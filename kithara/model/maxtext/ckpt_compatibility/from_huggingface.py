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

"""
This module provides functionality to load weights from HuggingFace's 
checkpoint into MaxText models.
"""

import time
from typing import Union, List, Optional, Callable
from huggingface_hub import snapshot_download
from keras_hub.src.utils.preset_utils import jax_memory_cleanup, load_json
from kithara.model.maxtext.ckpt_compatibility.param_mapping import (
    PARAM_MAPPING,
    HOOK_FNS,
)
from kithara.model.hf_compatibility import (
    get_model_name_from_preset_handle,
)
from kithara.utils.safetensor_utils import MaxTextSafetensorLoader


def port_weight(
    loader: MaxTextSafetensorLoader,
    keras_variable: "keras.Variable",
    hf_weight_key: Union[str | List[str]],
    hook_fn=Optional[Union[List | Callable]],
    scan_layers=False,
    expected_dtype=None
):
    target_shape = list(keras_variable.shape)
    target_is_stacked = scan_layers and isinstance(hf_weight_key, list)
    if target_is_stacked:
        target_shape = (target_shape[0], *target_shape[2:])

    if hook_fn:
        if not isinstance(hook_fn, list):
            hook_fn = [hook_fn]
    else:
        hook_fn = []

    start_time = time.time()
    if isinstance(hf_weight_key, str):
        hf_tensor = loader.get_tensor(hf_weight_key)
        for fn in hook_fn:
            hf_tensor = fn(hf_tensor, target_shape)
    else:
        hf_tensor = loader.get_tensors(hf_weight_key, hook_fn, target_shape)

    if expected_dtype:
        hf_tensor = hf_tensor.astype(expected_dtype)
    keras_variable.assign(hf_tensor)
    print(
        f"✅ Successfully loaded weight {keras_variable.path} into model in {time.time() - start_time:.3f}s"
    )


def load_hf_weights_into_maxtext_model(
    preset_handle: str, maxtext_model: "kithara.MaxtextModel", scan_layers=False
):
    """
    Loads weights from HuggingFace Hub into a MaxText model.

    Args:
        preset_handle: HuggingFace model preset identifier
        maxtext_model: A randomly initialized kithara.MaxTextModel instance
        scan_layers: Whether the MaxText model is initialize with the
            scan_layer option

    Returns:
        The loaded MaxText model

    Note: Loading the model will be slower the first time. Subsequent runs
    should be significantly faster as weights will be cached under
    the `~/.cache/huggingface/hub` or `HF_HOME` directory. If you are on a
    multi-host setup, the cache directory is `/home/ubuntu/.cache/huggingface/hub`,
    or `/home/ubuntu/HF_HOME`.
    """

    model_name = get_model_name_from_preset_handle(preset_handle)

    if model_name not in PARAM_MAPPING:
        raise ValueError(
            f"Model {model_name} is not supported. "
            f"Supported models are {list(PARAM_MAPPING.keys())}"
        )
    # Load config.json from HuggingFace Hub
    config = load_json(preset_handle)
    params_mapping = PARAM_MAPPING[model_name](config, scan_layers)
    hook_fn_mapping = HOOK_FNS[model_name](config, scan_layers, saving_to_hf=False)
    if params_mapping is None:
        raise ValueError(f"Model type {config['model_type']} is not current supported.")

    jax_memory_cleanup(maxtext_model)

    print(f"-> Downloading HuggingFace weights ({preset_handle})...")
    start_time = time.time()
    #snapshot_download(repo_id=preset_handle.removeprefix("hf://"))
    print(f"✅ Downloaded HuggingFace weights in {time.time() - start_time}s")

    with MaxTextSafetensorLoader(preset_handle) as loader:
        for variable in maxtext_model.weights:
            # Get the final path from the absolution path 
            variable_path = variable.path
            if variable.path.startswith("max_text_layer"):
                variable_path = variable.path.split("/")[-1]
            
            if variable_path not in params_mapping:
                raise ValueError(
                    f"Variable path {variable_path} does not exist in the provided weight_mapping"
                )

            try:
                expected_dtype = variable.value.dtype
                hook_fn = (
                    hook_fn_mapping.get(variable_path) if hook_fn_mapping else None
                )
                port_weight(
                    loader,
                    keras_variable=variable,
                    hf_weight_key=params_mapping[variable_path],
                    hook_fn=hook_fn,
                    scan_layers=scan_layers,
                    expected_dtype=expected_dtype,
                )

                assert (
                    variable.value.dtype == expected_dtype
                ), f"Expected weight dtype is {expected_dtype}, but weight dtype is {variable.value.dtype}"

            except Exception as e:
                raise ValueError(
                    f"Failed to load HF weight ({params_mapping[variable_path]}) into MaxText model({variable_path}). "
                    f"Error: {e}"
                )
    print(
        f"✅ Successfully loaded {preset_handle} into {model_name} in {time.time() - start_time:.3f}s..."
    )
    return maxtext_model
