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

"""This test validates the bidirectional conversion of model weights
between KerasHub and HuggingFace implementations. 

Steps:
    1. Load the HuggingFace model into KerasHub
    2. Save the KerasHub model in HuggingFace format
    3. Load the converted model back in HuggingFace
    4. Comparing weights and outputs with the original HuggingFace model

Metrics: 
    Max absolute difference between model weights
    Max absolute difference between the logits for the first 5 tokens.
    Disagreement among top1 tokens.

Usage: 
    Run script on single host VM: RUN_SKIPPED_TESTS=1 python -m unittest tests/model/kerashub/ckpt_compatibility/test_saving_models.py
"""

import os
os.environ["KERAS_BACKEND"] = "jax"
import unittest
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kithara import KerasHubModel
from kithara.utils.gcs_utils import find_cache_root_dir
from tests.model.test_prompt import TEST_PROMPT
from tests.model.utils import check_arrays_match, check_predicted_tokens_match


class TestSavingModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")
        cls.MAX_TARGET_LENGTH = 512

    def setUp(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)

    def _get_all_modules(self, model):
        """Get all weights names from a HF model."""
        modules = []
        for name, _ in model.named_modules():
            if name and hasattr(model.get_submodule(name), "weight"):
                modules.append(name)
        return modules

    def _check_weights_match(self, model, golden_model, tol):
        """Compare weights between two HF models."""
        modules = self._get_all_modules(golden_model)
        
        for module in modules:
            golden_weights = golden_model.get_submodule(module).state_dict()["weight"]
            model_weight = model.get_submodule(module).state_dict()["weight"]
            check_arrays_match(golden_weights, model_weight, tol)

    def _get_logits(self, model_id, model, golden_model):
        """Get logits from two HF models for comparison."""
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer.encode(TEST_PROMPT, return_tensors="pt")[:, :self.MAX_TARGET_LENGTH]

        logits = model(inputs, output_hidden_states=True).logits
        golden_logits = golden_model(inputs, output_hidden_states=True).logits

        return logits, golden_logits

    def run_conversion_test(self, model_id, weight_tol, logits_tol, top1_token_tol):
        # Create Model
        model = KerasHubModel.from_preset(
            preset_handle=f"hf://{model_id}",
            precision="float32"
        )

        # Save model
        model.save_in_hf_format(self.TMP_DIR)

        # Load converted model
        converted_model = AutoModelForCausalLM.from_pretrained(
            self.TMP_DIR, 
            torch_dtype=torch.float32
        )

        # Load reference model
        golden_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32
        )

        # Run comparison tests
        self._check_weights_match(converted_model, golden_model, weight_tol)
        
        # Get logits from both models
        logits, golden_logits = self._get_logits(model_id, converted_model, golden_model)

        # Compare logits from the first 5 tokens
        check_arrays_match(logits[0, :5, :], golden_logits[0, :5, :], logits_tol)
        
        # Check token predictions
        check_predicted_tokens_match(logits, golden_logits, top1_token_tol)

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2_2b_conversion(self):
        self.run_conversion_test(
            model_id="google/gemma-2-2b",
            weight_tol=0.0001,
            logits_tol=1.5,
            top1_token_tol=0.1
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2_9b_conversion(self):
        self.run_conversion_test(
            model_id="google/gemma-2-9b",
            weight_tol=0.0001,
            logits_tol=1.5,
            top1_token_tol=0.1
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_llama_31_8b_conversion(self):
        self.run_conversion_test(
            model_id="meta-llama/Llama-3.1-8B",
            weight_tol=0.0001,
            logits_tol=0.0001,
            top1_token_tol=0.0001
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_llama_32_1b_conversion(self):
        self.run_conversion_test(
            model_id="meta-llama/Llama-3.2-1B",
            weight_tol=0.0001,
            logits_tol=0.0001,
            top1_token_tol=0.0001
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_llama_32_3b_conversion(self):
        self.run_conversion_test(
            model_id="meta-llama/Llama-3.2-3B",
            weight_tol=0.0001,
            logits_tol=0.0001,
            top1_token_tol=0.0001
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
