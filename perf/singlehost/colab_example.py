import os

os.environ["KERAS_BACKEND"] = "jax"
import keras_hub
import keras
import numpy as np
from keras.src.backend.common.remat import RematScope
import jax 
OUTPUT_PATH = "gs://wenxindong-vm/mar04/perf/singlehost/enable_remat"

# Enable Flash Attention
keras.config.enable_flash_attention()

# Enable Remat         
with RematScope(mode="full"):
    model = keras_hub.models.GemmaCausalLM.from_preset("hf://google/gemma-2-2b")

batch_size = 1
seq_length = 1024
model_input =  {
        "token_ids": np.ones((batch_size, seq_length), dtype=np.int32),
        "padding_mask": np.ones((batch_size, seq_length)),
        }

print("running forward pass...")

for i in range(10): 
    if i == 5:
        jax.profiler.start_trace(OUTPUT_PATH)

    logits, _ = model.stateless_call(
        model.trainable_variables,
        model.non_trainable_variables,
        model_input,
    )

jax.profiler.stop_trace()

print("Done.")
