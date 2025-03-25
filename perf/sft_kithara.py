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

"""
Benchmark for SFT via Kithara.

This benchmark script runs supervised fine tuning using LoRA with the
specified model.

Metrics: tokens_per_second_per_device, tokens_per_second, samples_per_second

Purpose: Compare performance of SFT using Kithara.

Launch Script: python kithara/perf/sft_kithara.py
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import ray
from transformers import AutoTokenizer
from typing import Union, Optional, List
from kithara import KerasHubModel, Dataloader, Trainer, TextCompletionDataset
from datasets import load_dataset

config = {
    "model": "gemma",
    "model_handle": "google/gemma-2-2b",
    "seq_len": 4096,
    "use_lora": True,
    "lora_rank": 16,
    "precision": "mixed_bfloat16",
    "training_steps": 500,
    "eval_steps_interval": 100,
    "log_steps_interval": 10,
    "per_device_batch_size": 1,
    "packing": False,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "tensorboard": "unsloth_compare/",
}


def run_workload():
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    datasets = dataset.train_test_split(test_size=64, shuffle=False)
    train_source, eval_source = datasets["train"], datasets["test"]

    # Create model
    model = KerasHubModel.from_preset(
        f"hf://{config['model_handle']}",
        precision=config["precision"],
        lora_rank=config["lora_rank"] if config["use_lora"] else None,
    )

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_handle"])

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    def formatting_prompts_func(examples):
        EOS_TOKEN = tokenizer.eos_token
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    train_source = train_source.map(
        formatting_prompts_func,
        batched=True,
    )
    eval_source = eval_source.map(
        formatting_prompts_func,
        batched=True,
    )

    # Creates datasets
    train_dataset = TextCompletionDataset(
        train_source,
        tokenizer=tokenizer,
        max_seq_len=config["seq_len"],
    )

    eval_dataset = TextCompletionDataset(
        eval_source,
        tokenizer=tokenizer,
        max_seq_len=config["seq_len"],
    )

    if config["packing"]:
        train_dataset = train_dataset.to_packed_dataset()
        eval_dataset = eval_dataset.to_packed_dataset()

    # Create optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
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
        log_steps_interval=config["log_steps_interval"],
        tensorboard_dir=config["tensorboard"],
    )

    # Start training
    trainer.train()

    test_prompt = alpaca_prompt.format(
        "Continue the fibonnaci sequence.",  # instruction
        "1, 1, 2, 3, 5, 8",  # input
        "",  # output - leave this blank for generation!
    )

    # Test after tuning
    pred = model.generate(
        test_prompt,
        max_length=1000,
        tokenizer=tokenizer,
        return_decoded=True,
        strip_prompt=True,
    )
    print("Tuned model generates:", pred)


if __name__ == "__main__":
    run_workload()

#        '==='
#         |||
#      '- ||| -'
#     /  |||||  \   Kithara - Accelerated JAX Training | Device Count = 4
#    |   (|||)   |  Steps = 500 | Batch size per device = 1
#    |   |◕‿◕|   |  Total batch size = 4 | Total parameters = 2,626,056,448
#     \  |||||  /   Trainable parameters = 11,714,560 (0.45%) | Non-trainable = 2,614,341,888
#      --|===|--

