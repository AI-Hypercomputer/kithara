from datasets import load_dataset
 
# Load jsonl data from disk
train_dataset = load_dataset("json", data_files="kithara_trl_comparison/train_dataset.json", split="train")
eval_dataset = load_dataset("json", data_files="kithara_trl_comparison/test_dataset.json", split="train")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
 
# Hugging Face model id
model_id = "google/gemma-2-2b" # replace with your model id
 
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
 
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    use_cache=False,
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # to prevent errors with FA
tokenizer.truncation_side = 'left' # to prevent cutting off last generation

prompt_length = 1024
max_seq_length = 1512

from peft import LoraConfig
 
# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=256,
        lora_dropout=0.0,
        r=256,
        bias="none",
        target_modules=["v_proj", "q_proj"],
        task_type="CAUSAL_LM",
)

from transformers import TrainingArguments
from trl import DPOConfig
 
args = DPOConfig(
    output_dir="doplhin-dpo",               # directory to save and repository id
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=1,         # batch size per device during training
    per_device_eval_batch_size=1,           # batch size for evaluation
    gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch",              # use fused adamw optimizer
    learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
    logging_steps=2,                       # log every 25 steps
    evaluation_strategy="steps",            # evaluate every 1000 steps
    eval_steps=700,                         # when to evaluate
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    push_to_hub=False,                      # push model to hub
    max_length = max_seq_length,
    max_prompt_length = prompt_length,
    beta= 0.1,
    loss_type="sigmoid"
)
 
# dpo_args = {
#     "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
#     "loss_type": "sigmoid" ,                 # The loss type for DPO.
# }
# 
from trl import DPOTrainer
 
trainer = DPOTrainer(
    model,
    ref_model=None, # set to none since we use peft
    peft_config=peft_config,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
 
# save model at the end of training
trainer.save_model()


#The training with Flash Attention for 1 epochs with a dataset of ~10k samples took ~01:30:00 on 1x H100 GPU. You should be able to run the training on a g5.2xlarge instance by reducing the batch_size (est. to 1) and maybe the max_seq_length (est. to 1512).


del model
del trainer
torch.cuda.empty_cache()


import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
 
# Path to saved peft adapter model
# peft_model_id = args.output_dir # or
peft_model_id = "./doplhin-dpo"
 
# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  device_map="auto",
  torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompts = [
  "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
  "It's Bengay for muscle relief, a combination of methyl salicylate, menthol, and what other active ingredient commonly found in aspirin?",
  "How can i get rid of llamas in my backyard?"
]

for prompt in prompts:
  messages = pipe.tokenizer.apply_chat_template([{"role":"user", "content": prompt}], tokenize=False)
  outputs = pipe(prompt, max_new_tokens=2048, do_sample=True, temperature=1.0, top_k=50, top_p=0.9, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
  print(f"**Prompt**:\n{prompt}\n")
  print(f"**Generated Answer**:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
  print("===" * 10)
