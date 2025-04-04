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

from typing import Dict, Any, Optional, Tuple, Callable, Union

import numpy as np
import ray
from kithara.dataset.text_completion import TextCompletionDataset
from kithara.dataset.utils import HFtokenize


class BinaryPreferenceDataset(TextCompletionDataset):
    """A dataset class for binary preference optimization tasks, e.g. DPO.
    
    This class handles binary preference data where each sample contains a prompt,
    a chosen response, and a rejected response. It prepares the data for training
    models with preference optimization objectives.
    
    Args:
        source (ray.data.Dataset): The source Ray dataset containing the training data.
        tokenizer (Optional[AutoTokenizer]): HuggingFace tokenizer instance. If not provided, will be loaded
            from tokenizer_handle.
        tokenizer_handle (Optional[str]): Handle/name of the tokenizer to load if not provided.
        column_mapping (Optional[Dict[str, str]]): Mapping of source column names to expected column names
            ("prompt", "chosen", "rejected").
        model_type (Optional["ModelImplementationType"]): Type of model implementation to use. Please specify model_type or 
            set MODEL_IMPLEMENTATION in global state. Global state is automatically 
            set upon model initialization. Supported types: ModelImplementationType.KERASHUB, 
            ModelImplementationType.MAXTEXT
        max_prompt_length (int): Maximum length for the prompt portion of the input. Prompts
            exceeding this length will be truncated. Default: 512.
        max_seq_len (int): Maximum sequence length for tokenization. Sequences will be 
            padded to this length. Default: 1024.
        custom_formatting_fn (Optional[Callable]): A custom formatting function to apply to the raw
            sample before any other transformation steps.
    """

    def __init__(
        self,
        source: Union[ray.data.Dataset, "datasets.Dataset"],
        tokenizer: Optional["AutoTokenizer"] = None,
        tokenizer_handle: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        model_type: Optional["ModelImplementationType"] = "auto",
        max_prompt_length: int = 512,
        max_seq_len: int = 1024,
        custom_formatting_fn: Optional[Callable] = None,
    ):
        super().__init__(
            source=source,
            tokenizer=tokenizer,
            tokenizer_handle=tokenizer_handle,
            model_type=model_type,
            max_seq_len=max_seq_len,
            custom_formatting_fn=custom_formatting_fn,
        )
        self.max_prompt_length = max_prompt_length
        self.column_mapping = {
            "prompt": "prompt",
            "chosen": "chosen",
            "rejected": "rejected",
        }
        if column_mapping:
            self.column_mapping.update(column_mapping)

    def task_transform(self, sample: Dict[str, str]) -> Dict[str, str]:
        """Transform the raw sample into a standardized prompt-chosen-rejected format.
        
        This method applies any custom formatting and maps input columns to the
        standardized format expected by the model transformation step.

        Args:
            sample (Dict[str, str]): Raw sample containing prompt, chosen answer, and rejected answer.

        Returns:
            Dict[str, str]: Transformed sample with standardized keys.
        """
        if self.custom_formatting_fn:
            sample = self.custom_formatting_fn(sample)
            
        return {
            "prompt": (
                sample[self.column_mapping["prompt"]]
                if self.column_mapping["prompt"] is not None
                else ""
            ),
            "chosen": sample[self.column_mapping["chosen"]],
            "rejected": sample[self.column_mapping["rejected"]],
        }

    def model_transform(self, sample: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform the prompt-chosen-rejected triple into model inputs.
        
        This method tokenizes and formats the data for training, ensuring proper
        sequence lengths and creating the necessary input tensors.

        Args:
            sample (Dict[str, str]): Sample containing "prompt", "chosen", and "rejected" keys.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - input_ids: Array of token IDs with shape [2, sequence_length]
                - attention_mask: Array indicating non-padding positions
                - label_ids: Array of target token IDs for computing loss
            
            The chosen and rejected tokens will be interleaved as:
                input_ids: [[prompt+chosen], [prompt+rejected]]
                attention_mask: [[prompt+chosen], [prompt+rejected]]
                labels_ids: [[prompt+chosen], [prompt+rejected]]
                
            Labels are shifted to predict the next token, with padding in prompt positions.
        """
        prompt, chosen, rejected = (
            sample["prompt"],
            sample["chosen"],
            sample["rejected"],
        )
        
        # Tokenize the prompt with a length limit
        prompt_seq = HFtokenize(
            f"<bos>{prompt}", self.tokenizer, seq_len=self.max_prompt_length, padding="do_not_pad"
        )
        num_prompt_tokens = len(prompt_seq["input_ids"][0])

        # Tokenize chosen and rejected responses, leaving room for prompt tokens

        chosen_seq = HFtokenize(
            f"{chosen}<eos>",
            self.tokenizer,
            seq_len=self.max_seq_len - num_prompt_tokens,
        )
        rejected_seq = HFtokenize(
            f"{rejected}<eos>",
            self.tokenizer,
            seq_len=self.max_seq_len - num_prompt_tokens,
        )

        # Concatenate prompt with chosen and rejected sequences
        full_chosen_seq_input_ids = np.concatenate(
            [prompt_seq["input_ids"], chosen_seq["input_ids"]], axis=1
        )

        full_rejected_seq_input_ids = np.concatenate(
            [prompt_seq["input_ids"], rejected_seq["input_ids"]], axis=1
        )
        full_chosen_seq_mask = np.concatenate(
            [prompt_seq["attention_mask"], chosen_seq["attention_mask"]], axis=1
        )
        full_rejected_seq_mask = np.concatenate(
            [prompt_seq["attention_mask"], rejected_seq["attention_mask"]], axis=1
        )
        
        # Stack chosen and rejected sequences
        input_ids = np.concatenate(
            [full_chosen_seq_input_ids, full_rejected_seq_input_ids]
        )  # [2, sequence_length]
        attention_mask = np.concatenate(
            [full_chosen_seq_mask, full_rejected_seq_mask]
        )
        
        label_ids = np.roll(input_ids, -1)
        label_ids[:, -1] = self.tokenizer.pad_token_id
        label_ids[:, : num_prompt_tokens - 1] = self.tokenizer.pad_token_id
        
        return input_ids, attention_mask, label_ids