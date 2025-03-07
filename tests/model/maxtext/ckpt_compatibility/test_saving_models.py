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
between MaxText and HuggingFace implementations. 

Steps:
    1. Load the HuggingFace model into MaxText
    2. Save the MaxText model in HuggingFace format
    3. Load the saved model into a HuggingFace model
    4. Comparing weights and outputs with the original HuggingFace model

Metrics: 
    Max absolute difference between model weights.
    Max absolute difference between the logits for the first 5 tokens.
    Disagreement among top1 tokens.

Usage: 
    Run script on single host VM: RUN_SKIPPED_TESTS=1 python -m unittest tests/model/maxtext/ckpt_compatibility/test_saving_models.py
"""

import unittest
import os
import shutil
import torch
from kithara import MaxTextModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from kithara.utils.gcs_utils import find_cache_root_dir
from tests.model.utils import (
    check_arrays_match,
    check_predicted_tokens_match,
)
from tests.model.test_prompt import TEST_PROMPT

os.environ["KERAS_BACKEND"] = "jax"

class TestSavingModels(unittest.TestCase):
    """Test suite for validating bidirectional conversion of model weights between MaxText and HuggingFace."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
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

    def _run_conversion_test(self, model_id, weight_tol, logits_tol, top1_token_tol):
        # Create Model
        model = MaxTextModel.from_preset(
            preset_handle=f"hf://{model_id}",
            seq_len=self.MAX_TARGET_LENGTH,
            per_device_batch_size=1,
            scan_layers=True,
            precision="mixed_bfloat16",
        )

        # Save model
        model.save_in_hf_format(self.TMP_DIR)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(self.TMP_DIR, torch_dtype=torch.float32)

        # Load reference model
        golden_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        
        # Check weights match
        self._check_weights_match(model, golden_model, weight_tol)

        # Run forward pass to get logits
        logits, golden_logits = self._get_logits(model_id, model, golden_model)

        # Check logits from the first 5 tokens match
        check_arrays_match(logits[0, :5, :], golden_logits[0, :5, :], logits_tol)
        check_predicted_tokens_match(logits, golden_logits, top1_token_tol)

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2_2b_conversion(self):
        """Test conversion for Gemma 2B model."""
        self._run_conversion_test(
            model_id="google/gemma-2-2b",
            weight_tol=0.0001,
            logits_tol=0.0001,
            top1_token_tol=0.0001
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2_9b_conversion(self):
        """Test conversion for Gemma 9B model."""
        self._run_conversion_test(
            model_id="google/gemma-2-9b",
            weight_tol=0.0001,
            logits_tol=2.0,
            top1_token_tol=0.1
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_llama_31_8b_conversion(self):
        """Test conversion for Llama 3.1 8B model."""
        self._run_conversion_test(
            model_id="meta-llama/Llama-3.1-8B",
            weight_tol=0.0001,
            logits_tol=2.0,
            top1_token_tol=0.1
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
