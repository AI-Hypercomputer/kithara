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

from kithara.dataset.utils import HFtokenize
import ray
import numpy as np
from kithara.dataset.utils import HFtokenize
from typing import Dict, Any, Optional
from kithara.dataset.text_completion import TextCompletionDataset
from transformers import AutoTokenizer


class BinaryPreferenceDataset(TextCompletionDataset):
    """A dataset class for Direct Preference Optimization (DPO) tasks.

    Args:
        source (ray.data.Dataset): The source Ray dataset containing the training data.
        tokenizer (Optional[AutoTokenizer]): HuggingFace tokenizer instance.
        tokenizer_handle (Optional[str]): Handle/name of the tokenizer to load if not provided.
        column_mapping (Optional[Dict[str, str]]): Mapping of source column names to expected
            column names ("prompt", "chosen", "rejected").
        model_type (Optional[ModelImplementationType]): Type of model implementation to use.
            Please specify model_type or set MODEL_IMPLEMENTATION in global state. Global
            state is automatically set upon model initialization. Supported types:
            ModelImplementationType.KERASHUB, ModelImplementationType.MAXTEXT
        max_seq_len (int): Maximum sequence length for tokenization (default: 1024). Sequences
            will be padded to this length.
        custom_formatting_fn（callable): A custom formatting function to apply to the raw
            sample before any other transformation steps.
    """

    def __init__(
        self,
        source: ray.data.Dataset,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_handle: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        model_type: Optional["ModelImplementationType"] = "auto",
        max_seq_len: int = 1024,
        custom_formatting_fn: Optional[callable] = None,
    ):
        super().__init__(
            source=source,
            tokenizer=tokenizer,
            tokenizer_handle=tokenizer_handle,
            model_type=model_type,
            max_seq_len=max_seq_len,
            custom_formatting_fn=custom_formatting_fn,
        )
        self.column_mapping = {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}
        if column_mapping:
            self.column_mapping = {**self.column_mapping, **column_mapping}

    def task_transform(self, sample: Dict[str, str]) -> Dict[str, str]:
        """Transform the raw sample into a standardized prompt-chosen-rejected format.

        Args:
            sample (Dict[str, str]): Raw sample containing prompt, chosen answer,
                and rejected answer.

        Returns:
            Dict[str, str]: Transformed sample with standardized keys.

        """
        if self.custom_formatting_fn:
            sample = self.custom_formatting_fn(sample)
        return {
            "prompt": sample[self.column_mapping["prompt"]] if self.column_mapping["prompt"] is not None else "",
            "chosen": sample[self.column_mapping["chosen"]],
            "rejected": sample[self.column_mapping["rejected"]],
        }

    def model_transform(self, sample: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Transform the prompt-chosen-rejected triple into model inputs.

        Args:
            sample (Dict[str, str]): Sample containing "prompt", "chosen", and "rejected" keys.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing input_ids,
            attention_mask, and label_ids.
            The chosen and rejected tokens will be interleaved: 
                input_ids : [[prompt+chosen], [prompt+rejected], [prompt+chosen],[prompt+rejected]]
                padding_mask: [[prompt+chosen],  [prompt+rejected], [prompt+chosen],[prompt+rejected]]
                labels_ids: [[prompt+chosen], [prompt+rejected], [prompt+chosen],[prompt+rejected]]
        """
        prompt, chosen, rejected = (
            sample["prompt"],
            sample["chosen"],
            sample["rejected"],
        )
        full_chosen_seq = HFtokenize(
            f"<bos>{prompt}{chosen}<eos>", self.tokenizer, seq_len=self.max_seq_len
        )
        full_rejected_seq = HFtokenize(
            f"<bos>{prompt}{rejected}<eos>", self.tokenizer, seq_len=self.max_seq_len
        )
        prompt_seq = HFtokenize(
            f"<bos>{prompt}",
            self.tokenizer,
            seq_len=self.max_seq_len,
            padding="do_not_pad",
        )
        num_prompt_tokens = len(prompt_seq["input_ids"][0])

        input_ids = np.concatenate(
            [full_chosen_seq["input_ids"], full_rejected_seq["input_ids"]]
        )  # [2, S]
        attention_mask = np.concatenate(
            [full_chosen_seq["attention_mask"], full_rejected_seq["attention_mask"]]
        )

        label_ids = np.roll(input_ids, -1)
        label_ids[:, -1] = self.tokenizer.pad_token_id
        label_ids[:, : num_prompt_tokens - 1] = self.tokenizer.pad_token_id
        return input_ids, attention_mask, label_ids