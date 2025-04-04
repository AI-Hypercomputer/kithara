"""
This example demonstrates how to apply the DPO algorithm on Gemma2 and the UltraFeedback dataset.
Blog post: https://www.philschmid.de/dpo-align-llms-in-2024-with-trl
"""

from transformers import AutoTokenizer
from datasets import load_dataset
from kithara.dataset import Dataloader, BinaryPreferenceDataset
from kithara.model.model import ModelConfig, OptimizerConfig
from kithara.trainer.dpo import DPOConfig, DPOTrainer

# Load jsonl data from disk
train_dataset = load_dataset("json", data_files="train_dataset_toy.json", split="train")
eval_dataset = load_dataset("json", data_files="test_dataset_toy.json", split="train")


model_id = "google/gemma-2-2b"

policy_model_config = ModelConfig(
    preset_handle=f"hf://{model_id}",
    model_type="KerasHub",
    lora_rank=256,
    per_device_batch_size=1,
    seq_len=1024,
    optimizer=OptimizerConfig("adamw", learning_rate=5e-5),
)


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # to prevent errors with FA
tokenizer.truncation_side = 'left' # to prevent cutting off last generation

prompt_length = 1024
max_seq_length = 1512

dataset = BinaryPreferenceDataset(
    train_dataset,
    tokenizer_handle=model_id,
    max_prompt_length=prompt_length,
    max_seq_len=max_seq_length,
)

eval_dataset = BinaryPreferenceDataset(
    eval_dataset,
    tokenizer_handle=model_id,
    max_prompt_length=prompt_length,
    max_seq_len=max_seq_length,
)

dataloader = Dataloader(dataset, per_device_batch_size=1)
eval_dataloader = Dataloader(eval_dataset, per_device_batch_size=1)


dpo_config = DPOConfig(beta=0.1, policy_model=policy_model_config, run_mpmd=False)

dpo_trainer = DPOTrainer(
    dpo_config=dpo_config,
    train_dataloader=dataloader,
    eval_dataloader=eval_dataloader,
    epochs=1,
    log_steps_interval=1,
    eval_steps_interval=70000,
    max_eval_samples=10,
)

dpo_trainer.train()

prompts = [
    "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
    "It's Bengay for muscle relief, a combination of methyl salicylate, menthol, and what other active ingredient commonly found in aspirin?",
    "How can i get rid of llamas in my backyard?",
]

pred = dpo_trainer.policy_model.generate(
    prompts, tokenizer=tokenizer, max_length=1024
)
print("pred", pred)

