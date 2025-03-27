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

"""Quick Start Example

This script demonstrates how to run LoRA SFT on a toy dataset

1. Load HuggingFace Gemma2 checkpoint
2. Configure data loading and preprocessing
3. Run training across TPU/GPU devices

This script can be run on both single-host and multi-host. 
For mulit-host set up, please follow https://kithara.readthedocs.io/en/latest/scaling_with_ray.html.

Singlehost: python examples/singlehost/quick_start.py 
Multihost:  python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/quick_start.py" --hf-token your_token
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import ray
from kithara import (
    KerasHubModel,
    Dataloader,
    Trainer,
    SFTDataset,
)
from wandb import Settings

config = {
    "model_handle": "hf://google/gemma-2-2b",
    "tokenizer_handle": "hf://google/gemma-2-2b",
    "seq_len": 4096,
    "lora_rank": 16,
    "precision": "mixed_bfloat16",
    "training_steps": 60,
    "eval_steps_interval": 10,
    "log_steps_interval": 1,
    "per_device_batch_size": 1,
    "max_eval_samples": 50,
    "learning_rate": 2e-4,
    "wandb_settings": Settings(project="project"),
}


def run_workload():

    # Create a toy dataset
    dataset_items = [
        {
            "prompt": "What is your name?",
            "answer": "My name is Kithara",
        }
        for _ in range(1000)
    ]
    dataset = ray.data.from_items(dataset_items)
    train_source, eval_source = dataset.train_test_split(test_size=500)

    # Create model
    model = KerasHubModel.from_preset(
        config["model_handle"],
        precision=config["precision"],
        lora_rank=config["lora_rank"],
    )

    # Creates datasets
    train_dataset = SFTDataset(
        train_source,
        tokenizer_handle=config["tokenizer_handle"],
        max_seq_len=config["seq_len"],
    )
    eval_dataset = SFTDataset(
        eval_source,
        tokenizer_handle=config["tokenizer_handle"],
        max_seq_len=config["seq_len"],
    )

    # Create optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=config["learning_rate"], weight_decay=0.01
    )

    # Create data loaders
    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=config["per_device_batch_size"],
    )
    eval_dataloader = Dataloader(
        eval_dataset,
        per_device_batch_size=config["per_device_batch_size"],
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=config["training_steps"],
        eval_steps_interval=config["eval_steps_interval"],
        max_eval_samples=config["max_eval_samples"],
        log_steps_interval=config["log_steps_interval"],
        wandb_settings=config["wandb_settings"],
    )

    # Start training
    trainer.train()

    print("Finished training. Prompting model...")

    # Test after tuning
    pred = model.generate(
        "What is your name?",
        max_length=30,
        tokenizer_handle=config["tokenizer_handle"],
        return_decoded=True,
    )
    print("Tuned model generates:", pred)


if __name__ == "__main__":

    run_workload()
