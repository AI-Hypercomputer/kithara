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
Benchmark for full parameter finetuning via Kithara.

Metrics: tokens_per_second_per_device, tokens_per_second, samples_per_second

Purpose: Compare performance of full parameter fine tunign using Kithara.

Launch Script: python kithara/perf/kithara_packing.py
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from transformers import AutoTokenizer
from kithara import MaxTextModel, Dataloader, Trainer, TextCompletionDataset
from datasets import load_dataset

config = {
    "model": "gemma",
    "model_handle": "google/gemma-2-2b",
    "seq_len": 8192,
    "precision": "mixed_bfloat16",
    "training_steps": 250,
    "eval_steps_interval": 100,
    "log_steps_interval": 10,
    "per_device_batch_size": 1,
    "packing": False,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "tensorboard": "kithara_packing/no_packing",
}


def run_workload():
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    datasets = dataset.train_test_split(test_size=200, shuffle=False)
    train_source, eval_source = datasets["train"], datasets["test"]

    # Create model
    model = MaxTextModel.from_preset(
        preset_handle=f"hf://{config['model_handle']}",
        seq_len=config["seq_len"],
        per_device_batch_size=config["per_device_batch_size"],
        precision=config["precision"],
        scan_layers=True,
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

    # Create optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
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
        max_length=1024,
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
#     /  |||||  \   Kithara | Device Count = 4
#    |   (|||)   |  Steps = 500 | Batch size per device = 1
#    |   |◕‿◕|   |  Total batch size = 4 | Total parameters = 9,974.047859191895(mb)
#     \  |||||  /   Trainable parameters = 9,974.0478515625(mb) (100.0%) | Non-trainable = 7.62939453125e-06(mb)
