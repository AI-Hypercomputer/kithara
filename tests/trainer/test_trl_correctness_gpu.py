from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from transformers import TrainingArguments
from trl import DPOConfig

# Load jsonl data from disk
train_dataset = load_dataset("json", data_files="kithara_trl_comparison/train_dataset.json", split="train")
eval_dataset = load_dataset("json", data_files="kithara_trl_comparison/test_dataset.json", split="train")
 
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
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # to prevent errors with FA
tokenizer.truncation_side = 'left' # to prevent cutting off last generation

prompt_length = 1024
max_seq_length = 1512
 
# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=256,
        lora_dropout=0.0,
        r=256,
        bias="none",
        target_modules=["v_proj", "q_proj"],
        task_type="CAUSAL_LM",
)

 
args = DPOConfig(
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=1,         # batch size per device during training
    per_device_eval_batch_size=1,           # batch size for evaluation
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch",              # use fused adamw optimizer
    learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
    logging_steps=1,                       # log every 25 steps
    evaluation_strategy="steps",            # evaluate every 1000 steps
    eval_steps=70000,                         # when to evaluate
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_length = max_seq_length,
    max_prompt_length = prompt_length,
    beta= 0.1,
    loss_type="sigmoid"
)
 

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
