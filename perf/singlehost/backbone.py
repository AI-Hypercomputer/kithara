import os

os.environ["KERAS_BACKEND"] = "jax"

import keras_hub
import numpy as np
from keras.src.backend.common.remat import RematScope

input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

with RematScope(mode=None):
    model = keras_hub.models.GemmaBackbone.from_preset("hf://google/gemma-2-2b")

model(input_data)

