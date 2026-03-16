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

from ctypes import cdll
import subprocess
import shutil
from keras.src.callbacks.callback import Callback
import jax
import os
import wandb


class Wandb(Callback):
    """Callbacks to send data to Weights and Biases.

    Args:
        settings (wandb.Settings): Settings to init Weights and Biases with.
        learning_rate (float, optional): Training learning rate. Defaults to None.
        epochs (int, optional): Training epochs. Defaults to None.
    """

    def __init__(
        self,
        settings,
        learning_rate=None,
        epochs=None,
    ):
        super().__init__()
        wandb.login()
        config = {}
        if learning_rate:
            config["learning_rate"] = learning_rate
        if epochs:
            config["epochs"] = epochs
        wandb.init(
            settings=settings,
            config=config,
        )

    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        wandb.finish()

    def on_train_batch_begin(self, batch, logs=None):
        return

    def on_train_batch_end(self, batch, logs=None):
        if logs != None:
            wandb.log(logs)
