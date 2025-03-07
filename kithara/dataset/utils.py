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

from typing import List, Dict
from transformers import AutoTokenizer as HFTokenizer
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=5) 
def initialize_tokenizer(tokenizer_handle, tokenizer=None):
    """
    Creates an HuggingFace AutoTokenizer with the tokenizer_handle.
    Try to specify a pad_token for the tokenizer if not specified.
    """
    if tokenizer is None:
        if tokenizer_handle.startswith("hf://"):
            tokenizer_handle = tokenizer_handle.removeprefix("hf://")
        try:
            tokenizer = HFTokenizer.from_pretrained(tokenizer_handle)
        except ValueError as e:
            print(f"Tokenizer handle {tokenizer_handle} is not a valid HuggingFace tokenizer handle.")
    
    # Llama tokenizers don't have a default pad_token, we must add it here
    if tokenizer.pad_token is None: 
        # Gemma 2 
        if tokenizer.get_vocab().get("<pad>") is not None: 
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
        # Llama 3 uses "<|finetune_right_pad_id|>"
        elif tokenizer.get_vocab().get("<|finetune_right_pad_id|>") is not None: 
            tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
        # Llama 2 doesn't have a padding token, fall back to "<unk>"
        elif tokenizer.get_vocab().get("<unk>") is not None: 
            tokenizer.add_special_tokens({'pad_token': '<unk>'})
        else:
            print("WARNING: Tokenizer doesn't have the attribute pad_token")
    return tokenizer


def HFtokenize(
    text: List[str],
    tokenizer: HFTokenizer,
    seq_len: int,
    padding: str = "max_length",
) -> Dict[str, np.ndarray]:
    """Tokenizes text using a HuggingFace tokenizer with specific parameters.

    This function handles the tokenization of input text, applying padding and
    truncation as needed. It converts text into a format suitable for model input.

    Args:
        text (Union[str, List[str]]): Input text or list of texts to tokenize
        tokenizer (HFTokenizer): HuggingFace tokenizer instance
        seq_len (int): Maximum sequence length for tokenization
        padding (str, optional): Padding strategy. Defaults to "max_length".
            Options include:
            - "max_length": Pad to seq_len
            - "longest": Pad to longest sequence in batch
            - "do_not_pad": No padding

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'input_ids': Token IDs (shape: [batch_size, seq_len])
            - 'attention_mask': Mask indicating real vs padded tokens
                              (shape: [batch_size, seq_len])

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> result = HFtokenize(["hello world"], tokenizer, seq_len=10)
        >>> print(result['input_ids'].shape)
        (1, 10)
    """

    return tokenizer(
        text,
        max_length=seq_len,
        padding=padding,
        padding_side="right",
        truncation=True,
        add_special_tokens=False,
        return_tensors="np",
    )
