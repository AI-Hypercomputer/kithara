"""
This example demonstrates how to apply the DPO algorithm on Gemma2 and the UltraFeedback dataset.
Blog post: https://www.philschmid.de/dpo-align-llms-in-2024-with-trl
"""

from transformers import AutoTokenizer
from datasets import load_dataset
from kithara.dataset import Dataloader, BinaryPreferenceDataset
from kithara.model.model import ModelConfig, OptimizerConfig
from kithara.trainer.dpo import DPOConfig, DPOTrainer


def run_workload():

    model_id = "google/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation

    # Load jsonl data from disk
    train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
    eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

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

    policy_model_config = ModelConfig(
        preset_handle=f"hf://{model_id}",
        model_type="KerasHub",
        lora_rank=256,
        per_device_batch_size=1,
        seq_len=1024,
        optimizer=OptimizerConfig("adamw", learning_rate=5e-5),
    )

    dpo_config = DPOConfig(beta=0.1, policy_model=policy_model_config, run_mpmd=False)

    dpo_trainer = DPOTrainer(
        dpo_config=dpo_config,
        train_dataloader=dataloader,
        eval_dataloader=eval_dataloader,
        steps=10,
        epochs=1,
        log_steps_interval=2,
        eval_steps_interval=5,
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


if __name__ == "__main__":
    run_workload()


# pred ['A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?\n\nStep 1\n1 of 2\n\nThe perimeter of a rectangle is the sum of the lengths of its sides.\n\nThe perimeter of the garden is:\n\n$P=2l+2w $\n\nSubstitute $l=25$ and $w=15$:\n\n$P=2(25)+2(15) $\n\n$P=50+30 $\n\n$\\color{#c34632}P=80\\text{ feet} $\n\nResult\n2 of 2\n\n$80\\text{ feet} $', "It's Bengay for muscle relief, a combination of methyl salicylate, menthol, and what other active ingredient commonly found in aspirin?\n\nA. Acetaminophen\n\nB. Ibuprofen\n\nC. Aspirin\n\nD. Naproxen\n\nE. None of the above\n\nThe answer is C. Aspirin.\n\nAspirin is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Aspirin is a salicylate, which is a type of NSAID.\n\nMethyl salicylate is the active ingredient in Bengay. It is a topical analgesic that is used to relieve pain and inflammation. Menthol is also an active ingredient in Bengay. It is a topical analgesic that is used to relieve pain and inflammation.\n\nIbuprofen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Ibuprofen is a salicylate, which is a type of NSAID.\n\nAcetaminophen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Acetaminophen is a salicylate, which is a type of NSAID.\n\nNaproxen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Naproxen is a salicylate, which is a type of NSAID.\n\nThe answer is C. Aspirin.\n\nAspirin is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Aspirin is a salicylate, which is a type of NSAID.\n\nMethyl salicylate is the active ingredient in Bengay. It is a topical analgesic that is used to relieve pain and inflammation. Menthol is also an active ingredient in Bengay. It is a topical analgesic that is used to relieve pain and inflammation.\n\nIbuprofen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Ibuprofen is a salicylate, which is a type of NSAID.\n\nAcetaminophen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Acetaminophen is a salicylate, which is a type of NSAID.\n\nNaproxen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Naproxen is a salicylate, which is a type of NSAID.\n\nThe answer is C. Aspirin.\n\nAspirin is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Aspirin is a salicylate, which is a type of NSAID.\n\nMethyl salicylate is the active ingredient in Bengay. It is a topical analgesic that is used to relieve pain and inflammation. Menthol is also an active ingredient in Bengay. It is a topical analgesic that is used to relieve pain and inflammation.\n\nIbuprofen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Ibuprofen is a salicylate, which is a type of NSAID.\n\nAcetaminophen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Acetaminophen is a salicylate, which is a type of NSAID.\n\nNaproxen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Naproxen is a salicylate, which is a type of NSAID.\n\nThe answer is C. Aspirin.\n\nAspirin is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, fever, and inflammation. It is also used to prevent heart attacks and strokes. Aspirin is a salicylate, which is a type of NSAID.\n\nMethyl salicylate is the active ingredient in Bengay. It is a topical analgesic that is used to relieve pain and inflammation. Menthol is also an active ingredient in Bengay. It is a topical analgesic that is used to relieve pain and inflammation.\n\nIbuprofen is a nonsteroidal anti-inflammatory drug (NSAID) that is", 'How can i get rid of llamas in my backyard?\n\n[User 0001]\n\nI have a problem with llamas in my backyard. I have a 1000 sq ft backyard and they are constantly coming in and eating my plants. I have tried to scare them away with a loud noise but they just keep coming back. I have also tried to put up a fence but they just jump over it. I am at my wits end and need some help.\n\n[User 0002]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0003]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0004]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0005]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0006]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0007]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0008]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0009]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0010]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0011]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0012]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0013]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0014]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0015]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0016]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around the perimeter of your yard. The llamas will not be able to jump over the wire fence.\n\n[User 0017]\n\nI\'m not sure what you mean by "backyard" but if you have a fence around your yard, you can put a wire fence around']
