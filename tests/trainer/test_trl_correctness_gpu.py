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
    output_dir = "temp/",
    num_train_epochs=1,
    max_steps=10, 
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

# On GPU
# {'loss': 0.6931, 
# 'grad_norm': 780.9299926757812, 
# 'learning_rate': 4.5e-05, 
# 'rewards/chosen': 0.0, 
# 'rewards/rejected': 0.0, 
# 'rewards/accuracies': 0.0, 
# 'rewards/margins': 0.0, 
# 'logps/chosen': -2225.393310546875, 
# 'logps/rejected': -617.556884765625, 
# 'logits/chosen': -3.687845230102539, 
# 'logits/rejected': -2.044302225112915, 
# 'epoch': 0.0}
# {'loss': 0.7896, 'grad_norm': 25.966228485107422, 'learning_rate': 4e-05, 'rewards/chosen': 0.3772842586040497, 'rewards/rejected': 0.5616249442100525, 'rewards/accuracies': 0.0, 'rewards/margins': -0.1843406856060028, 'logps/chosen': -53.942771911621094, 'logps/rejected': -47.29401397705078, 'logits/chosen': 1.0061376094818115, 'logits/rejected': 0.8444373607635498, 'epoch': 0.0}
# {'loss': 1.2726, 'grad_norm': 141.87716674804688, 'learning_rate': 3.5e-05, 'rewards/chosen': 7.152734279632568, 'rewards/rejected': 8.09666919708252, 'rewards/accuracies': 0.0, 'rewards/margins': -0.9439349174499512, 'logps/chosen': -352.88067626953125, 'logps/rejected': -217.74766540527344, 'logits/chosen': -2.577226161956787, 'logits/rejected': -1.5558576583862305, 'epoch': 0.0}
# {'loss': 0.0, 'grad_norm': 8.189210234377242e-07, 'learning_rate': 3e-05, 'rewards/chosen': 32.94903564453125, 'rewards/rejected': 11.424270629882812, 'rewards/accuracies': 1.0, 'rewards/margins': 21.524765014648438, 'logps/chosen': -821.6431884765625, 'logps/rejected': -272.1170349121094, 'logits/chosen': -4.15282678604126, 'logits/rejected': -0.7165324091911316, 'epoch': 0.0}
# {'loss': 0.0276, 'grad_norm': 3.067979335784912, 'learning_rate': 2.5e-05, 'rewards/chosen': 11.181021690368652, 'rewards/rejected': 7.606120586395264, 'rewards/accuracies': 1.0, 'rewards/margins': 3.5749011039733887, 'logps/chosen': -429.8357849121094, 'logps/rejected': -177.8030548095703, 'logits/chosen': -2.9004147052764893, 'logits/rejected': -3.855966567993164, 'epoch': 0.0}
# {'loss': 1.3442, 'grad_norm': 233.4412078857422, 'learning_rate': 2e-05, 'rewards/chosen': 18.96540641784668, 'rewards/rejected': 20.007471084594727, 'rewards/accuracies': 0.0, 'rewards/margins': -1.0420646667480469, 'logps/chosen': -703.53173828125, 'logps/rejected': -823.1295166015625, 'logits/chosen': -2.741118907928467, 'logits/rejected': 1.0434099435806274, 'epoch': 0.0}
# {'loss': 4.8927, 'grad_norm': 1063.8914794921875, 'learning_rate': 1.5e-05, 'rewards/chosen': 24.133350372314453, 'rewards/rejected': 29.01850700378418, 'rewards/accuracies': 0.0, 'rewards/margins': -4.885156631469727, 'logps/chosen': -872.611328125, 'logps/rejected': -1593.0496826171875, 'logits/chosen': -2.376319646835327, 'logits/rejected': 0.18929380178451538, 'epoch': 0.0}
# {'loss': 0.001, 'grad_norm': 0.139197438955307, 'learning_rate': 1e-05, 'rewards/chosen': 19.535497665405273, 'rewards/rejected': 12.591073989868164, 'rewards/accuracies': 1.0, 'rewards/margins': 6.944423675537109, 'logps/chosen': -449.90423583984375, 'logps/rejected': -336.0372314453125, 'logits/chosen': -3.8210296630859375, 'logits/rejected': -2.1282849311828613, 'epoch': 0.0}
# {'loss': 0.0, 'grad_norm': 0.004171309527009726, 'learning_rate': 5e-06, 'rewards/chosen': 11.86602783203125, 'rewards/rejected': 1.8350273370742798, 'rewards/accuracies': 1.0, 'rewards/margins': 10.031000137329102, 'logps/chosen': -227.44097900390625, 'logps/rejected': -86.51261901855469, 'logits/chosen': -1.9000743627548218, 'logits/rejected': -0.15733405947685242, 'epoch': 0.0}
# {'loss': 16.4419, 'grad_norm': 308.7105407714844, 'learning_rate': 0.0, 'rewards/chosen': 40.579647064208984, 'rewards/rejected': 57.0215950012207, 'rewards/accuracies': 0.0, 'rewards/margins': -16.44194793701172, 'logps/chosen': -811.4797973632812, 'logps/rejected': -1336.0142822265625, 'logits/chosen': -4.886911869049072, 'logits/rejected': -2.932751417160034, 'epoch': 0.0}
# {'train_runtime': 6.7274, 'train_samples_per_second': 1.486, 'train_steps_per_second': 1.486, 'train_loss': 2.546275564860598, 'epoch': 0.0}
# 100%|██████████████████████████████| 10/10 [00:06<00:00,  1.49it/s]

# TPU results:
# {'step': 1, 'loss': 0.691, -- seems correct 
# 'logits/chosen': Array(-1.94531, dtype=bfloat16),  vs -3.687845230102539, 
# 'logits/rejected': Array(-1.41406, dtype=bfloat16), vs -2.044302225112915, 
# 'logps/chosen': Array(-14.9375, dtype=bfloat16), vs -2225.393310546875, 
# 'logps/rejected': Array(-14, dtype=bfloat16), vs -617.556884765625, 
# 'rewards/accuracies': Array(0.75, dtype=float32), vs 0 
# 'rewards/chosen': Array(0.00108337, dtype=bfloat16),  -- seems correct 
# 'rewards/margins': Array(0.00112915, dtype=bfloat16),  -- seems correct 
# 'rewards/rejected': Array(-4.33922e-05, dtype=bfloat16)} -- seems correct 


{'step': 1, 'loss': 0.691, 
 'logits/chosen': Array(7.3125, dtype=bfloat16), 
 'logits/rejected': Array(8.9375, dtype=bfloat16), 
 'logps/chosen': Array(-15.5625, dtype=bfloat16), 
 'logps/rejected': Array(-14.25, dtype=bfloat16), 
 
 'rewards/accuracies': Array(0.75, dtype=float32), 
 'rewards/chosen': Array(0.00108337, dtype=bfloat16), 
 'rewards/margins': Array(0.00112915, dtype=bfloat16), 
 'rewards/rejected': Array(-4.33922e-05, dtype=bfloat16)}

# {'step': 2, 'loss': 0.695, 'step_time': 25.87, 'epoch': 1, 'tokens_per_second_per_device': 39.6, 'tokens_per_second': 158.3, 'samples_per_second': 0.15, 'train_steps_per_second': 0.04, 'samples_seen': 8, 'logits/chosen': Array(-0.710938, dtype=bfloat16), 'logits/rejected': Array(-0.730469, dtype=bfloat16), 'logps/chosen': Array(-13.25, dtype=bfloat16), 'logps/rejected': Array(-13.4375, dtype=bfloat16), 'rewards/accuracies': Array(0.25, dtype=float32), 'rewards/chosen': Array(0.00646973, dtype=bfloat16), 'rewards/margins': Array(-0.00151825, dtype=bfloat16), 'rewards/rejected': Array(0.00799561, dtype=bfloat16)}
# {'step': 4, 'loss': 0.68, 'step_time': 20.48, 'epoch': 1, 'tokens_per_second_per_device': 50.0, 'tokens_per_second': 200.0, 'samples_per_second': 0.2, 'train_steps_per_second': 0.05, 'samples_seen': 16, 'logits/chosen': Array(-1.39844, dtype=bfloat16), 'logits/rejected': Array(-1.17188, dtype=bfloat16), 'logps/chosen': Array(-15, dtype=bfloat16), 'logps/rejected': Array(-14.375, dtype=bfloat16), 'rewards/accuracies': Array(0.75, dtype=float32), 'rewards/chosen': Array(0.0252686, dtype=bfloat16), 'rewards/margins': Array(0.0256348, dtype=bfloat16), 'rewards/rejected': Array(-0.000318527, dtype=bfloat16)}
# {'eval_loss': 0.668, 'eval_step': 1, 'step_time': 12.24, 'tokens_per_second_per_device': 83.7, 'tokens_per_second': 334.7, 'eval_samples_per_second': 0.33, 'eval_steps_per_second': 0.08, 'logits/chosen': Array(-1.61719, dtype=bfloat16), 'logits/rejected': Array(-1.38281, dtype=bfloat16), 'logps/chosen': Array(-15.25, dtype=bfloat16), 'logps/rejected': Array(-15.375, dtype=bfloat16), 'rewards/accuracies': Array(0.75, dtype=float32), 'rewards/chosen': Array(-0.0209961, dtype=bfloat16), 'rewards/margins': Array(0.0512695, dtype=bfloat16), 'rewards/rejected': Array(-0.0722656, dtype=bfloat16)}
# Eval loss after 5 training steps: 0.679688
# {'step': 6, 'loss': 0.68, 'step_time': 14.4, 'epoch': 1, 'tokens_per_second_per_device': 71.1, 'tokens_per_second': 284.5, 'samples_per_second': 0.28, 'train_steps_per_second': 0.07, 'samples_seen': 24, 'logits/chosen': Array(-1.60938, dtype=bfloat16), 'logits/rejected': Array(-1.4375, dtype=bfloat16), 'logps/chosen': Array(-15.3125, dtype=bfloat16), 'logps/rejected': Array(-14.3125, dtype=bfloat16), 'rewards/accuracies': Array(0.75, dtype=float32), 'rewards/chosen': Array(-0.0238037, dtype=bfloat16), 'rewards/margins': Array(0.0281982, dtype=bfloat16), 'rewards/rejected': Array(-0.052002, dtype=bfloat16)}
# {'step': 8, 'loss': 0.676, 'step_time': 20.51, 'epoch': 1, 'tokens_per_second_per_device': 49.9, 'tokens_per_second': 199.7, 'samples_per_second': 0.2, 'train_steps_per_second': 0.05, 'samples_seen': 32, 'logits/chosen': Array(-1.07812, dtype=bfloat16), 'logits/rejected': Array(-1.16406, dtype=bfloat16), 'logps/chosen': Array(-13.8125, dtype=bfloat16), 'logps/rejected': Array(-14.125, dtype=bfloat16), 'rewards/accuracies': Array(0.75, dtype=float32), 'rewards/chosen': Array(0.0917969, dtype=bfloat16), 'rewards/margins': Array(0.0327148, dtype=bfloat16), 'rewards/rejected': Array(0.0593262, dtype=bfloat16)}
# {'step': 10, 'loss': 0.656, 'step_time': 21.0, 'epoch': 1, 'tokens_per_second_per_device': 48.8, 'tokens_per_second': 195.1, 'samples_per_second': 0.19, 'train_steps_per_second': 0.05, 'samples_seen': 40, 'logits/chosen': Array(-0.902344, dtype=bfloat16), 'logits/rejected': Array(-0.765625, dtype=bfloat16), 'logps/chosen': Array(-13.3125, dtype=bfloat16), 'logps/rejected': Array(-13.25, dtype=bfloat16), 'rewards/accuracies': Array(0.5, dtype=float32), 'rewards/chosen': Array(0.0908203, dtype=bfloat16), 'rewards/margins': Array(0.0766602, dtype=bfloat16), 'rewards/rejected': Array(0.0144653, dtype=bfloat16)}

# ======= toy dataset ========

#GPU
# {'loss': 0.6931, 
# 'grad_norm': 12.881969451904297, 
# 'learning_rate': 5e-05, 
# 'rewards/chosen': 0.0, 
# 'rewards/rejected': 0.0, 
# 'rewards/accuracies': 0.0, 
# 'rewards/margins': 0.0, 
# 'logps/chosen': -27.29616928100586, 
# 'logps/rejected': -31.401323318481445,
# 'logits/chosen': -5.121490955352783, 'logits/rejected': -6.006796360015869, 'epoch': 0.1}
# {'loss': 0.1206, 'grad_norm': 4.664772987365723, 'learning_rate': 5e-05, 'rewards/chosen': 1.1259502172470093, 'rewards/rejected': -0.9283022284507751, 'rewards/accuracies': 1.0, 'rewards/margins': 2.0542523860931396, 'logps/chosen': -16.036666870117188, 'logps/rejected': -40.68434524536133, 'logits/chosen': -6.098273754119873, 'logits/rejected': -7.0534987449646, 'epoch': 0.2}

# TPU

# {'step': 1, 'loss': 0.691, 
# 'logits/chosen': Array(7.3125, dtype=bfloat16), 
# 'logits/rejected': Array(8.9375, dtype=bfloat16), 
# 'logps/chosen': Array(-15.5625, dtype=bfloat16), 
# 'logps/rejected': Array(-14.25, dtype=bfloat16), 
# 'rewards/accuracies': Array(0.75, dtype=float32), 
# 'rewards/chosen': Array(0.00108337, dtype=bfloat16), 
# 'rewards/margins': Array(0.00112915, dtype=bfloat16), 
# 'rewards/rejected': Array(-4.33922e-05, dtype=bfloat16)}