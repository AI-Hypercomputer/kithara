.. _pretraining:

Continued Pre-training
======================

:bdg-primary:`Example`

In this guide, we'll demonstrate how to continue pretraining the Gemma 2 model using a large Math Pretraining dataset. 

This example runs on both single-host and multi-host environments. You can also find this script on `Github <https://github.com/AI-Hypercomputer/kithara/blob/main/examples/singlehost/continued_pretraining_example.py>`_.  


Let's Get Started! 🚀
--------------------

First things first, log into HuggingFace, set up the Keras backend to use JAX and import necessary modules::

    from huggingface_hub import login
    login(token="your_hf_token", add_to_git_credential=False)

    import os
    os.environ["KERAS_BACKEND"] = "jax"

    import keras
    import ray
    from kithara import (
        MaxTextModel,
        Dataloader,
        TextCompletionDataset,
        SFTDataset,
        Trainer,
        Checkpointer,
    )
    from datasets import load_dataset

Next, create a GCS bucket to store your checkpoints and logs::

    gsutil mb gs://my-bucket


Step 1: Load Data 
-----------------

For illustration purposes, we use the Open Web Math dataset for training 
and the GSM8K dataset for evaluation.

In this example we use HuggingFace datasets. You can also load your own dataset, 
check out :doc:`supported data formats <datasets>`.

.. code-block:: python

    # Load dataset in streaming mode. This avoids downloading ~30GB of data.
    hf_train_dataset = load_dataset(
        "open-web-math/open-web-math", split="train", streaming=True
    )

    hf_eval_dataset = load_dataset("openai/gsm8k", "main", split="test")

Step 2: Create Dataloaders
-----------------------------

.. tip:: 
    Sequence packing helps improve training efficiency by reducing padding. Learn about it :doc:`here <packing>`.

Create Kithara Dataset and Dataloader. Per-device batch size is set to 1 per device, but you can increase it if you have enough HBM memory.

.. code-block:: python

    train_dataset = TextCompletionDataset(
        hf_train_dataset,
        tokenizer_handle="hf://google/gemma-2-2b",
        max_seq_len=4096,
    ).to_packed_dataset()  # Activate sequence packing

    eval_dataset = SFTDataset(
        hf_eval_dataset,
        tokenizer_handle="hf://google/gemma-2-2b",
        max_seq_len=4096,
        column_mapping={"prompt": "question", "answer": "answer"},
    ).to_packed_dataset()

    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=1,
    )

    eval_dataloader = Dataloader(
        eval_dataset,
        per_device_batch_size=1,
    )
    

Step 3: Initialize Model and Optimizer
---------------------------------------

You can use a larger model (e.g. ``hf://google/gemma-2-9b``, ``hf://google/gemma-2-27b``) if you are training on multiple hosts and have enough memory.

.. code-block:: python

    model = MaxTextModel.from_preset(
        "hf://google/gemma-2-2b",
        seq_len=4096,
        per_device_batch_size=1,
        scan_layers=True
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )


Step 4: Setup Checkpointing
---------------------------

Save checkpoints to a cloud storage bucket every 50 steps and keep the last 5 checkpoints::
    
    checkpointer = Checkpointer(
        "gs://my-bucket/checkpoints",
        model=model,
        save_interval_steps=50,
        max_to_keep=5,
    )


Step 5: Start Training
---------------------------------------

Train for 100 steps, evaluate every 10 steps, and log every step::

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=100,
        eval_steps_interval=10,
        log_steps_interval=1,
        max_eval_samples=50,
        checkpointer=checkpointer,
        tensorboard_dir="gs://my-bucket/tensorboard",
    )

    # 3...2...1... Go!
    trainer.train()

Step 6: Model Inference
----------------------

Test the continued pre-trained model. Note that the model output will not make sense since we've only trained it for 100 steps::

    test_prompt = "Harry slept 9 hours last night. His friend James slept only 2/3 of what Harry slept. How many more hours did Harry sleep than James?"

    pred = model.generate(
        test_prompt,
        max_length=1000,
        tokenizer_handle="hf://google/gemma-2-2b",
        skip_special_tokens=True,
        return_decoded=True,
        strip_prompt=True,
    )
    print("Generated response:", pred)


Step 7: Save Model
-----------------

Save the model in the HuggingFace format::

    model.save_in_hf_format("gs://my-bucket/models")
    # Or, if you prefer saving to HuggingFace Hub 
    # model.save_in_hf_format("hf://my-hf-id/repo-name")


Notes
-----

- Give ~10 minutes for this script to complete the first time you run it. Subsequent runs will be shorter as the model and compilation would be cached.
- To run this example on multihost, use this `script <https://github.com/AI-Hypercomputer/kithara/blob/main/examples/multihost/ray/TPU/continued_pretraining_example.py>`_.
- In practice you will train for much longer steps. 
