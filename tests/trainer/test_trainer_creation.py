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

"""Unit tests for running SFT on singlehost

Run test on a TPU VM: python -m unittest tests/trainer/test_trainer_creation.py
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

import optax

import unittest
from kithara import (
    MaxTextModel,
    TextCompletionDataset,
    Dataloader,
    Trainer
)
import unittest.result

import ray
from kithara.utils.gcs_utils import find_cache_root_dir
import shutil
import keras


class TestRunningSFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")
        cls.MODEL_HANDLE = "hf://google/gemma-2-2b"
        cls.TOKENIZER_HANDLE = "hf://google/gemma-2-2b"
        cls.SEQ_LEN = 100
        cls.PRECISION = "mixed_bfloat16"
        cls.LOG_STEPS_INTERVAL = 5
        cls.PER_DEVICE_BATCH_SIZE = 1
        cls.LEARNING_RATE = 5e-5

    def setUp(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)

    def _create_model(self):
        return MaxTextModel.from_random(
            "default",
            seq_len=self.SEQ_LEN,
            precision=self.PRECISION,
            per_device_batch_size=self.PER_DEVICE_BATCH_SIZE,
            scan_layers=True,
        )

    def _create_dataloaders(self):
        dataset_items = [
            {"text": f"{i} What is your name? My name is Mary."} for i in range(100)
        ]
        dataset = ray.data.from_items(dataset_items)
        train_source, eval_source = dataset.train_test_split(test_size=50)

        train_dataset = TextCompletionDataset(
            train_source,
            tokenizer_handle=self.TOKENIZER_HANDLE,
            max_seq_len=self.SEQ_LEN,
        )
        eval_dataset = TextCompletionDataset(
            eval_source,
            tokenizer_handle=self.TOKENIZER_HANDLE,
            max_seq_len=self.SEQ_LEN,
        )
        train_dataloader = Dataloader(
            train_dataset,
            per_device_batch_size=self.PER_DEVICE_BATCH_SIZE,
            dataset_is_sharded_per_host=False,
        )
        eval_dataloader = Dataloader(
            eval_dataset,
            per_device_batch_size=self.PER_DEVICE_BATCH_SIZE,
            dataset_is_sharded_per_host=False,
        )

        return train_dataloader, eval_dataloader

    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
    def test_train_with_epochs(self):

        train_dataloader, eval_dataloader = self._create_dataloaders()
        model = self._create_model()

        # Create Keras optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.LEARNING_RATE,
            weight_decay=0.01,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            epochs=1,
            log_steps_interval=self.LOG_STEPS_INTERVAL,
            tensorboard_dir=os.path.join(self.TMP_DIR, "tensorboard")
        )

        # Start training
        trainer.train()

    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
    def test_train_with_optax_optimizer(self):

        train_dataloader, eval_dataloader = self._create_dataloaders()
        model = self._create_model()

        # Create Optax optimizer
        optimizer = optax.chain(
            optax.adamw(self.LEARNING_RATE, weight_decay=.01),
            optax.clip(1.0)
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            epochs=1,
            log_steps_interval=self.LOG_STEPS_INTERVAL,
            tensorboard_dir=os.path.join(self.TMP_DIR, "tensorboard")
        )

        # Start training
        trainer.train()

    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
    def test_train_with_steps(self):

        train_dataloader, eval_dataloader = self._create_dataloaders()
        model = self._create_model()

        # Create Keras optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.LEARNING_RATE,
            weight_decay=0.01,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            steps=10,
            log_steps_interval=self.LOG_STEPS_INTERVAL,
            tensorboard_dir=os.path.join(self.TMP_DIR, "tensorboard")
        )

        # Start training
        trainer.train()


    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
    def test_train_with_and_step_evals(self):

        train_dataloader, eval_dataloader = self._create_dataloaders()
        model = self._create_model()

        # Create Keras optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.LEARNING_RATE,
            weight_decay=0.01,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            steps=10,
            eval_steps_interval=2,
            log_steps_interval=self.LOG_STEPS_INTERVAL,
            tensorboard_dir=os.path.join(self.TMP_DIR, "tensorboard")
        )

        # Start training
        trainer.train()

    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
    def test_train_with_and_epoch_evals(self):

        train_dataloader, eval_dataloader = self._create_dataloaders()
        model = self._create_model()

        # Create Keras optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.LEARNING_RATE,
            weight_decay=0.01,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            epochs=2,
            eval_epochs_interval=1,
            log_steps_interval=self.LOG_STEPS_INTERVAL,
            tensorboard_dir=os.path.join(self.TMP_DIR, "tensorboard")
        )

        # Start training
        trainer.train()

if __name__ == "__main__":
    unittest.main(verbosity=2)
