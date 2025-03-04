"""
Launch Script: HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/dev/shm/temp/hf KERAS_HOME=/dev/shm/temp/keras python perf/kerashub_singlehost.py
"""

def run_benchmark():
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    from examples.example_datasets import example_datasets
    from kithara import KerasHubModel
    from kithara.dataset import Dataloader, TextCompletionDataset
    from kithara.trainer import Trainer
    from kithara.callbacks import Profiler

    # Run parameters
    BASE_OUTPUT_DIR = "gs://wenxindong-vm/mar04/perf/singlehost"  # MODIFY with your GS bucket
    MODEL_HANDLE = "hf://google/gemma-2-2b"
    SEQ_LEN = 4096
    PER_DEVICE_BATCH_SIZE = 1
    
    train_data, eval_data = example_datasets(option="finetune_toy")

    #Successfully installed keras-nightly-3.9.0.dev2025030403
    from keras.src.backend.common.remat import RematScope
    import keras
    
    keras.config.disable_flash_attention()
            
    # with RematScope(mode="full"):

    model = KerasHubModel.from_preset(
        MODEL_HANDLE,
        precision="mixed_bfloat16"
    )

    # Create Keras optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Create Dataset
    train_ds = TextCompletionDataset(
        source = train_data, 
        tokenizer_handle=MODEL_HANDLE,
        max_seq_len=SEQ_LEN,
    )

    # Create Dataloader
    train_dataloader = Dataloader(train_ds, per_device_batch_size=PER_DEVICE_BATCH_SIZE)

    # Create Xprof Profiler
    profiler = Profiler(
        mode="xplane",
        output_path=BASE_OUTPUT_DIR,
        max_profile_steps=5,
        skip_first_n_steps=5,
        optional_postfix="kerashub",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        steps=10,
        log_steps_interval=1,
        tensorboard_dir=BASE_OUTPUT_DIR,
        profiler=profiler,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    run_benchmark()
    

