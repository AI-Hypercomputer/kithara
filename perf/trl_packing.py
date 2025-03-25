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
Benchmark for full parameter finetuning via trl.

Metrics: train_samples_per_second

Purpose: Compare perforance of full parameter finetuning using trl.

Launch Script: python kithara/perf/trl_packing.py
"""


"""
Set up:
pip install trl peft transformers datasets hf_transfer  huggingface_hub[cli]

Run this script on single host:  HF_HUB_ENABLE_HF_TRANSFER=1 python kithara_trl_comparison/trl_train.py
"""

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
datasets = dataset.train_test_split(test_size=200, shuffle=False)
train_dataset, test_dataset = datasets["train"], datasets["test"]

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=dataset,
    eval_dataset = test_dataset,
    args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        max_steps=1000,
        logging_steps=10,
        # eval_strategy="steps",
        # eval_steps=100,
        learning_rate = 2e-4,
        optim = "adamw_torch",
        lr_scheduler_type="constant",
        bf16=True,
        seed = 3407,
        report_to = "tensorboard",
        max_seq_length = 8192,
        dataset_num_proc = 1,
        packing = False,
        output_dir = "trl_packing/no_packing"
    ),
)

trainer.train()

inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print("generated", tokenizer.batch_decode(outputs))

