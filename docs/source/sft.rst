.. _sft:

Supervised Fine-Tuning + LoRA
=============================

:bdg-primary:`Example` 

In this guide, we're going to transform the powerful Gemma 2B model into your very own customized AI assistant. 

This example runs on both single host and multi-host environments. You can also find this script on `Github <https://github.com/AI-Hypercomputer/kithara/blob/main/examples/singlehost/sft_lora_example.py>`_.  


Let's Get Started! 🎉
--------------------

First things first, log into HuggingFace, set up the Keras backend to use JAX and import necessary modules::

    from huggingface_hub import login
    login(token="your_hf_token", add_to_git_credential=False)

    import os
    os.environ["KERAS_BACKEND"] = "jax"

    from kithara import (
        KerasHubModel,
        Dataloader,
        Trainer,
        SFTDataset,
    )
    import keras
    from transformers import AutoTokenizer

Step 1: Initialize Model
----------------------------------

.. tip::
    This examples uses LoRA for efficient fine-tuning, but you can also fine-tune the entire model by setting ``lora_rank=None``.

Create the model, tokenizer, and optimizer::

    model = KerasHubModel.from_preset(
        "hf://google/gemma-2-2b",
        lora_rank=16  # ✨ LoRA rank for parameter-efficient fine-tuning
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    optimizer = keras.optimizers.AdamW(
        learning_rate=2e-5,
        weight_decay=0.01
    )


Step 2: Prepare Training Data
-----------------------------

.. tip:: 
    To finetune on your custom dataset, check out :doc:`supported data formats <datasets>`.

Set up the dataset formatting and loading::

    from datasets import load_dataset
    import ray

    # Define Alpaca-style prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:"""

    def formatting_prompts_func(examples):
        """Format examples using the Alpaca prompt template"""
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(
                instruction=instruction,
                input=input
            )
            texts.append(text)
        return {
            "prompt": texts,
            "answer": outputs
        }

    # Load and split the dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    datasets = dataset.train_test_split(test_size=200)
    train_source, eval_source = datasets["train"], datasets["test"]

Step 3: Create Training Datasets
-----------------------------

.. tip:: 
    Per-device batch size is set to 1 per device, but you can increase it if you have enough HBM memory.

Initialize the training and evaluation datasets::

    train_dataset = SFTDataset(
        train_source,
        tokenizer=tokenizer,
        max_seq_len=4096,
        custom_formatting_fn=formatting_prompts_func,
    )

    eval_dataset = SFTDataset(
        eval_source,
        tokenizer=tokenizer,
        max_seq_len=4096,
        custom_formatting_fn=formatting_prompts_func,
    )

    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=1,
    )

    eval_dataloader = Dataloader(
        eval_dataset,
        per_device_batch_size=1,
    )

Step 4: Initialize and Run Training
--------------------------------

Set up the trainer and start the training process::

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=100, # You can also use epochs instead of steps
        eval_steps_interval=20,
        max_eval_samples=50,
        log_steps_interval=10,
    )

    # 3...2...1... Go!
    trainer.train()


Step 5: Model Inference
---------------------

Test the fine-tuned model::

    test_prompt = alpaca_prompt.format(
        instruction="Continue the fibonnaci sequence.",
        input="1, 1, 2, 3, 5, 8",
    )

    pred = model.generate(
        test_prompt,
        max_length=500,
        tokenizer=tokenizer,
        return_decoded=True
    )
    print("Generated response:", pred)

Step 6: Save Model
------------------

Save the model in the Hugging Face format::

    model.save_in_hf_format(
        "model_output/", # You can also save the model to Google Cloud Storage, or directly to HuggingFace Hub
        only_save_adapters=True, # You can also save the base model, or merge the base model with the adapters
        save_adapters_separately=True
    )

Example Output
-------------

.. code-block:: text

    Generated response: The next number in the sequence is 13.

    Explanation:
    The fibonacci sequence is a sequence of numbers where each number
    is the sum of the two previous numbers. The sequence starts with
    1 and 1, and the next number is 2. The next number is 3, and
    the next number is 5. The next number is 8, and the next number
    is 13.


Notes
-----

- Give ~10 minutes for this script to complete the first time you run it. Subsequent runs will take shorter as model and compilation would be cached. 
- To run this example on multihost, use this `script <https://github.com/AI-Hypercomputer/kithara/blob/main/examples/multihost/ray/TPU/sft_lora_example.py>`_.
