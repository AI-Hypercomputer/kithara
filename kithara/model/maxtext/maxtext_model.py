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
    supported_models,
)
from kithara.model.maxtext.inference_engine import MaxtextInferenceEngine as MaxEngine
from tqdm import tqdm


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

    def __init__(
        self,
        model,
        sharding_strategy,
        model_name: str,
        precision: str,
        scan_layers: bool,
        maxtext_config: dict,
    ):
        """Initialize a MaxTextModel instance.

        Args:
            model: The underlying MaxText model instance.
            sharding_strategy: Strategy for sharding the model across devices.
            model_name (str): Name of the MaxText model.
            precision (str): Precision mode used for computations.
            scan_layers (bool): Whether scan layers are used for memory efficiency.
            maxtext_config (dict): Configuration parameters for the MaxText model.
        """
        super().__init__(
            model,
            sharding_strategy,
            model_name,
            precision,
            scan_layers,
        )

        self.maxtext_config = maxtext_config
        self.maxengine = None
        self.decode_state = None

    @classmethod
    def from_random(
        cls,
        model_name: str,
        seq_len: int = 2048,
        per_device_batch_size: int = 1,
        precision: str = "mixed_bfloat16",
        scan_layers: bool = True,
        max_prefill_predict_length: int = "auto",
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
                Defaults to True.
            max_prefill_predict_length (int, optional): The maximum length of prompt tokens supported at
                inferenece time. The number of generated tokens is capped to max_seq_len -
                max_prefill_length. This number is required by the inference server. It is
                defaulted to "auto" which sets to `seq_len//2`. You should set it a larger number
                if you are running inference with this model on larger prompts; or set it to a
                smaller number if you are working with short prompts and what to generate more tokens.
            maxtext_config_args (Optional[dict], optional): Additional configuration arguments.
                Defaults to None.

        Returns:
            MaxTextModel: A new instance of MaxTextModel with random initialization.
        """
        set_global_model_implementation_type(ModelImplementationType.MAXTEXT)

        set_precision(precision)
        weight_dtype = cls._weight_dtype(precision)
        activation_dtype = cls._activation_dtype(precision)

        if max_prefill_predict_length == "auto":
            max_prefill_predict_length = seq_len // 2

        maxtext_config, sharding_strategy, model = cls.initialize_random_maxtext_model(
            model_name,
            seq_len,
            per_device_batch_size,
            weight_dtype,
            activation_dtype,
            scan_layers,
            max_prefill_predict_length,
            maxtext_config_args,
        )

        return cls(
            model,
            sharding_strategy,
            model_name=model_name,
            precision=precision,
            scan_layers=scan_layers,
            maxtext_config=maxtext_config,
        )

    @classmethod
    def from_preset(
        cls,
        preset_handle: str,
        seq_len: int = 2048,
        per_device_batch_size: int = 1,
        precision: str = "mixed_bfloat16",
        scan_layers: bool = "auto",
        max_prefill_predict_length: int = "auto",
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
            scan_layers (bool, optional): Whether to use scan layers. Defaults to "auto".
                Scan layer means to stack the weights of sequential layers into one logical
                array, which will significantly speed up compilation time. However, when
                converting weights from HuggingFace into MaxText, layer scanning may led to
                OOM if the stacked weights is too large to fix in HBM.
            max_prefill_predict_length (int, optional): The maximum length of prompt tokens supported at
                inferenece time. The number of generated tokens is capped to max_seq_len -
                max_prefill_length. This number is required by the inference server. It is
                defaulted to "auto" which sets to `seq_len//2`. You should set it a larger number
                if you are running inference with this model on larger prompts; or set it to a
                smaller number if you are working with short prompts and what to generate more tokens.
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
        if scan_layers == "auto":
            scan_layers = does_model_support_scanning(model_name)
        if max_prefill_predict_length == "auto":
            max_prefill_predict_length = seq_len // 2
        maxtext_config, sharding_strategy, model = cls.initialize_random_maxtext_model(
            model_name,
            seq_len,
            per_device_batch_size,
            weight_dtype,
            activation_dtype,
            scan_layers,
            max_prefill_predict_length,
            maxtext_config_args,
        )
        model = load_hf_weights_into_maxtext_model(preset_handle, model, scan_layers)

        return cls(
            model,
            sharding_strategy,
            model_name=model_name,
            precision=precision,
            scan_layers=scan_layers,
            maxtext_config=maxtext_config,
        )

    def _convert_text_to_tokens(
        self,
        prompts: Union[str | List[str]],
        tokenizer: "AutoTokenizer",
    ) -> list[np.ndarray]:
        tokens: Dict[str, np.ndarray] = tokenizer(
            prompts,
            padding="do_not_pad",
            return_tensors="np",
        )

        return tokens["input_ids"]

    def _generate(
        self,
        inputs: Union[List[str], List[np.ndarray]],
        max_length: int = None,
        stop_token_ids: Optional[List] = None,
        strip_prompt: str = False,
        tokenizer: Optional["AutoTokenizer"] = None,
        **kwargs,
    ) -> List[List[int]]:
        """Generate tokens using the model.

        Args:
            inputs (list[str]|list[np.ndarray]): A non-empty list of strings
                or numpy arrays.
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

        self._maybe_initialize_maxengine()

        if isinstance(inputs[0], str):
            inputs: list[np.ndarray] = self._convert_text_to_tokens(inputs, tokenizer)

        params = MaxTextConversionMixin.get_maxtext_params(self.model)
        if max_length is None:
            max_length = self.maxtext_config.max_target_length
        max_length = min(max_length, self.maxtext_config.max_target_length)

        def run_inference_on_one_batch(batch_prompts: list[np.ndarray]):
            # Start result buffer with the prompt tokens, later we will strip
            # prompt tokens if strip_prompt=True
            batch_results = [
                prompt.tolist() if isinstance(prompt, np.ndarray) else prompt
                for prompt in batch_prompts
            ]
            finished = [False for _ in batch_prompts]
            # Prefill with the prompt tokens
            for i, prompt_tokens in enumerate(batch_prompts):
                if len(prompt_tokens) >= min(
                    self.maxtext_config.max_prefill_predict_length, max_length
                ):
                    print(
                        f"Skipping {i}'th input because it's length "
                        "({len(prompt_tokens)}) exceeds max_length({max_length}) "
                        f"or max_prefill_predict_length({self.maxtext_config.max_prefill_predict_length})."
                    )
                    finished[i] = True
                    continue
                prefill_result, first_token = self.maxengine.prefill(
                    params=params,
                    padded_tokens=prompt_tokens,
                    true_length=len(prompt_tokens),
                )
                batch_results[i].append(int(prefill_result["tokens"][0][0]))
                self.decode_state = self.maxengine.insert(
                    prefill_result, self.decode_state, slot=i
                )

            # Generate the remaining tokens
            for _ in range(max_length):
                if all(finished):
                    break
                self.decode_state, _ = self.maxengine.generate(
                    params,
                    self.decode_state,
                )
                for i in range(len(batch_prompts)):
                    if finished[i]:
                        continue
                    token = int(self.decode_state["tokens"][i][0])
                    batch_results[i].append(token)
                    # Check if generation should continue
                    if token in stop_token_ids or len(batch_results[i]) >= max_length:
                        finished[i] = True
                if all(finished):
                    break
            return batch_results

        results = []
        total_num_prompts = len(inputs)

        for batch_start in tqdm(
            range(0, total_num_prompts, self.maxengine.max_concurrent_decodes),
            desc="Running batch inference",
            unit="batch",
        ):
            results.extend(
                run_inference_on_one_batch(
                    inputs[
                        batch_start : batch_start
                        + self.maxengine.max_concurrent_decodes
                    ]
                )
            )
        if strip_prompt:
            for i in range(total_num_prompts):
                results[i] = results[i][len(inputs[i]) :]

        return results

    def save_in_hf_format(
        self, output_dir: str, dtype: str = "auto", parallel_threads=8
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
            parallel_threads (int, optional): Number of parallel threads to use for saving.
                Defaults to 8. Make sure the local system has at least
                `parallel_threads * DEFAULT_MAX_SHARD_SIZE` free disk space,
                as each thread will maintain a local cache of size `DEFAULT_MAX_SHARD_SIZE`.
        """
        save_maxtext_model_in_hf_format(
            self, output_dir, dtype=dtype, parallel_threads=parallel_threads
        )

    def _maybe_initialize_maxengine(self):
        if self.maxengine is None:
            self.maxengine = MaxEngine(self.maxtext_config)
            params = MaxTextConversionMixin.get_maxtext_params(self.model)
            self.maxengine.load_existing_params(params)
            self.decode_state = self.maxengine.init_decode_state()

    def disable_lora(self):
        return
    
    def enable_lora(self):
        return

def does_model_support_scanning(model_name):
    return model_name in [
        supported_models.GEMMA2_2B,
        supported_models.GEMMA2_9B,
        supported_models.LLAMA31_8B,
        supported_models.LLAMA32_1B,
        supported_models.LLAMA32_3B,
    ]
