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

import transformers

# From https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
qwen25_d5_config = transformers.Qwen2Config(
    attention_dropout = 0, initializer_range=.02,
    vocab_size=151936, hidden_size=896,
    intermediate_size=4864, num_hidden_layers=24,
    num_attention_heads=14, num_key_value_heads=2,
    max_position_embeddings=32768, max_window_layers=24,
    rms_norm_eps=1e-06, rope_theta=1000000.0, sliding_window=32768,
    tie_word_embeddings=True, use_cache=True, use_mrope=False,
    bos_token_id=151643, eos_token_id=151643,
    use_sliding_window=False
)

# From https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json
qwen25_1d5_config = transformers.Qwen2Config(
    attention_dropout=.0, hidden_act="silu",
    hidden_size=1536, initializer_range=.02,
    intermediate_size=8960, max_position_embeddings=131072,
    max_window_layers=28, num_attention_heads=12,
    num_hidden_layers=28, num_key_value_heads=2,
    rms_norm_eps=1e-06, rope_theta=1000000.0,
    sliding_window=131072, tie_word_embeddings=True,
    use_cache=True, use_mrope=False, use_sliding_window=False,
    bos_token_id=151643, eos_token_id=151643,
    vocab_size=151936
)

# From https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/config.json
qwen25_3b_config = transformers.Qwen2Config(
    attention_dropout=0.0, hidden_act="silu",
    hidden_size=2048, initializer_range=0.02,
    intermediate_size=11008, max_position_embeddings=32768,
    max_window_layers=36, num_attention_heads=16,
    num_hidden_layers=36, num_key_value_heads=2,
    rms_norm_eps=1e-06, rope_theta=1000000.0,
    sliding_window=32768, tie_word_embeddings=True,
    use_cache=True, use_mrope=False, use_sliding_window=False,
    bos_token_id=151643, eos_token_id=151643,
    vocab_size=151936
)

# From https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json
qwen25_7b_config = transformers.Qwen2Config(
    attention_dropout=0.0, hidden_act="silu",
    hidden_size=3584, initializer_range=0.02,
    intermediate_size=18944, max_position_embeddings=131072,
    max_window_layers=28, num_attention_heads=28,
    num_hidden_layers=28, num_key_value_heads=4,
    rms_norm_eps=1e-06, rope_theta=1000000.0,
    sliding_window=131072, tie_word_embeddings=False,
    use_cache=True, use_mrope=False, use_sliding_window=False,
    bos_token_id=151643, eos_token_id=151643,
    vocab_size=152064
)

# From https://huggingface.co/Qwen/Qwen2.5-14B/blob/main/config.json
qwen25_14b_config = transformers.Qwen2Config(
    attention_dropout=0.0, hidden_act="silu",
    hidden_size=5120, initializer_range=0.02,
    intermediate_size=13824, max_position_embeddings=131072,
    max_window_layers=48, num_attention_heads=40,
    num_hidden_layers=48, num_key_value_heads=8,
    rms_norm_eps=1e-05, rope_theta=1000000.0,
    sliding_window=131072, tie_word_embeddings=False,
    use_cache=True, use_sliding_window=False,
    bos_token_id=151643, eos_token_id=151643,
    vocab_size=152064
)

# From https://huggingface.co/Qwen/Qwen2.5-32B/blob/main/config.json
qwen25_32b_config = transformers.Qwen2Config(
    attention_dropout=0.0, hidden_act="silu",
    hidden_size=5120, initializer_range=0.02,
    intermediate_size=27648, max_position_embeddings=131072,
    max_window_layers=64, num_attention_heads=40,
    num_hidden_layers=64, num_key_value_heads=8,
    rms_norm_eps=1e-05, rope_theta=1000000.0,
    sliding_window=131072, tie_word_embeddings=False,
    use_cache=True, use_sliding_window=False,
    bos_token_id=151643, eos_token_id=151643,
    vocab_size=152064
)

# From https://huggingface.co/Qwen/Qwen2.5-72B/blob/main/config.json
qwen25_72b_config = transformers.Qwen2Config(
    attention_dropout=0.0, hidden_act="silu",
    hidden_size=8192, initializer_range=0.02,
    intermediate_size=29568, max_position_embeddings=131072,
    max_window_layers=80, num_attention_heads=64,
    num_hidden_layers=80, num_key_value_heads=8,
    rms_norm_eps=1e-05, rope_theta=1000000.0,
    sliding_window=131072, tie_word_embeddings=False,
    use_cache=True, use_sliding_window=False,
    bos_token_id=151643, eos_token_id=151643,
    vocab_size=152064
)
