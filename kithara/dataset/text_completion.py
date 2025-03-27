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

from kithara.dataset.dataset import Dataset
from kithara.dataset.packed_dataset import PackedDataset
from kithara.dataset.utils import initialize_tokenizer, HFtokenize
import ray
from keras.src.backend.common import global_state
import numpy as np
from kithara.dataset.utils import HFtokenize
from typing import Dict, Optional, Union
from transformers import AutoTokenizer


class TextCompletionDataset(Dataset):
    """
    A dataset class for standard text completion tasks.

    Args:
        source (ray.data.Dataset| datasets.Dataset): The source Ray or HuggingFace 
            dataset containing the text data.
        tokenizer: The tokenizer instance to use.
        tokenizer_handle: Handle/name of the tokenizer to load if not provided.
        column_mapping (Optional[Dict]): Mapping of source column name to expected
            column name ("text")
        model_type (ModelImplementationType | "auto"): Type of model implementation to use.
            MaxText and KerasHub models expect input formats, the dataset needs to know
            which model it is feeding data to. This field can be set to "auto" or a specific
            model type. Supported mode_type: ModelImplementationType.KERASHUB,
            ModelImplementationType.MAXTEXT. When set to "auto", model_type will be inferred
            from the `MODEL_IMPLEMENTATION` global state. `MODEL_IMPLEMENTATION` will be
            automatically set upon model initialization. You should only need to manually
            specify this argument when you are creating a dataset without creating a model.
        max_seq_len (int): Maximum sequence length for tokenization (default: 1024).
        custom_formatting_fn（callable): A custom formatting function to apply to the raw
            sample before any other transformation steps.
    """

    def __init__(
        self,
        source: Union[ray.data.Dataset, "datasets.Dataset"],
        tokenizer: AutoTokenizer = None,
        tokenizer_handle: str = None,
        column_mapping: Dict[str, str] = None,
        model_type: "ModelImplementationType" = "auto",
        max_seq_len: int = 1024,
        custom_formatting_fn: Optional[callable] = None,
        packing=False,
    ):
        super().__init__(source)

        assert (tokenizer is not None) or (
            tokenizer_handle is not None
        ), "Either a HF Tokenizer or a HF tokenizer handle must be provided"

        self.max_seq_len = max_seq_len
        self.tokenizer = initialize_tokenizer(tokenizer_handle, tokenizer)
        self.column_mapping = {"text": "text"}
        self._model_type = model_type
        self.custom_formatting_fn = custom_formatting_fn
        self.packing = packing
        if column_mapping:
            self.column_mapping = {**self.column_mapping, **column_mapping}

    @property
    def model_type(self):
        # Avoid circular import
        from kithara.model import ModelImplementationType

        if self._model_type == "auto":
            model_type = global_state.get_global_attribute("MODEL_IMPLEMENTATION", None)
        else:
            model_type = self._model_type

        if model_type not in ModelImplementationType.list_supported_types():
            raise ValueError(
                "Did you forget to specify model_type during Dataset creation? Please specify model_type or set MODEL_IMPLEMENTATION "
                "in global state. Supported types: `KerasHub`, `MaxText`. You don't need to specify model type if you have already created"
                " a model. "
            )

        return model_type

    def task_transform(self, sample):
        """Transform the raw sample with standardized key."""
        if self.custom_formatting_fn:
            sample = self.custom_formatting_fn(sample)
        return {"text": sample[self.column_mapping["text"]]}

    def model_transform(self, sample: Dict[str, str]) -> Dict[str, str]:
        """Transform the text sample into model inputs.

        Returns:
            tuple: Tuple containing input_ids, attention_mask, and label_ids.
        """
        text = sample["text"]
        full_seq = HFtokenize(
            f"<bos>{text}<eos>", self.tokenizer, seq_len=self.max_seq_len
        )
        input_ids = full_seq["input_ids"]
        attention_mask = full_seq["attention_mask"]

        label_ids = np.roll(input_ids, -1)
        label_ids[:, -1] = self.tokenizer.pad_token_id
        return input_ids, attention_mask, label_ids

    def convert_to_model_specific_input(
        self, input_ids: np.ndarray, attention_mask: np.ndarray, label_ids: np.ndarray
    ):
        """
        Convert tokenized inputs into format expected by either the MaxText model
        for the KerasHub model.

        Args:
            input_ids (np.ndarray): Token IDs.
            attention_mask (np.ndarray): Attention mask.
            label_ids (np.ndarray): Label IDs.

        Returns:
            Dict[str, ndarray]: Model-specific formatted inputs.
        """
        # Avoid circular import
        from kithara.model import ModelImplementationType

        input_formats = {
            ModelImplementationType.MAXTEXT: lambda: {
                "x": {
                    "tokens": input_ids,
                    "segment_ids": attention_mask,
                    "positions": np.tile(
                        np.arange(self.max_seq_len, dtype=np.int32), (len(input_ids), 1)
                    ),
                },
                "y": label_ids,
            },
            ModelImplementationType.KERASHUB: lambda: {
                "x": {"token_ids": input_ids, "padding_mask": attention_mask},
                "y": label_ids,
            },
        }

        if self.model_type not in input_formats:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return input_formats[self.model_type]()

    def process_sample(self, sample: int) -> Dict[str, np.ndarray]:
        """Overrides the base class implementation. This function
        process a single sample through the complete transformation pipeline.

        Args:
            sample: Raw sample from the dataset.

        Returns:
            Dict[str, np.ndarray]: Processed sample in model-specific format.
        """
        sample = self.task_transform(sample)
        input_ids, attention_mask, label_ids = self.model_transform(sample)
        sample = self.convert_to_model_specific_input(
            input_ids, attention_mask, label_ids
        )
        return sample

    def to_packed_dataset(self) -> PackedDataset:
        """Converts the current dataset to a PackedDataset for more efficient processing.

        The PackedDataset combines multiple sequences into single fixed-length sequences
        to maximize computational efficiency during training. This is particularly useful
        for handling variable-length sequences by packing them together with proper
        segmentation and position information.

        Packing currently only works for MaxText models.

        Returns:
            PackedDataset: A new dataset instance that packs sequences from this dataset
                together, using the tokenizer's pad token ID for padding.
        """
        return PackedDataset(self, pad_value=self.tokenizer.pad_token_id)
