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

"""Compare model outputs between HuggingFace and KerasHub model. 

This test validates that the KerasHub implementations produces logits that are
numerically comparable to the reference HuggingFace implementation. 

Steps: 
    1. Load HF model into KerasHub model.
    2. Get logits from HF model 
    3. Get logits from KerasHub model.
    4. Compare logits 
    
Usage:
    Run script on single host VM: RUN_SKIPPED_TESTS=1 python -m unittest  tests/model/kerashub/ckpt_compatibility/test_loading_models.py
"""

import unittest
import os
from kithara import KerasHubModel
from tests.model.test_prompt import TEST_PROMPT
import tests.model.utils as utils
from tests.model.utils import check_arrays_match, check_predicted_tokens_match

class TestLoadingModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.max_target_length = 512
        
    def get_kerashub_logits(self, model_id, input_ids):
        """Get logits from KerasHub model.
        
        Args:
            model_id (str): HuggingFace model identifier
            input_ids (tensor): Input token IDs
            
        Returns:
            tensor: Model logits
        """
        model = KerasHubModel.from_preset(
            preset_handle=f"hf://{model_id}",
            precision="float32"
        )

        # Run forward pass
        input = utils.get_kerashub_model_input(input_ids)
        logits, _ = model.stateless_call(
            model.trainable_variables,
            model.non_trainable_variables,
            input,
        )
        return logits

    def _test_model_correctness(self, model_id, logits_tol, top1_token_tol):
        """
        Args:
            model_id (str): HuggingFace model identifier
            logits_tol (float): Tolerance for logits comparison
            top1_token_tol (float): Tolerance for top-1 token prediction match
        """
        # Get logits from both models
        input_ids, logits_hf = utils.get_hf_logits(
            model_id, 
            TEST_PROMPT, 
            self.max_target_length, 
            return_input_ids=True
        )
        logits_kerashub = self.get_kerashub_logits(model_id, input_ids)
        
        # Extract logits for the first input sequence
        logits_hf = logits_hf[0, :, :]
        logits_kerashub = logits_kerashub[0, :, :]

        # Perform checks
        with self.subTest("Testing logits match"):
            check_arrays_match(
                logits_hf[:5, :], 
                logits_kerashub[:5, :], 
                atol=logits_tol
            )
            
        with self.subTest("Testing predicted tokens match"):
            check_predicted_tokens_match(
                logits_hf, 
                logits_kerashub, 
                tolerance=top1_token_tol
            )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2_2b(self):
        self._test_model_correctness(
            model_id="google/gemma-2-2b",
            logits_tol=0.5,
            top1_token_tol=0.2
        )
    
    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2_9b(self):
        self._test_model_correctness(
            model_id="google/gemma-2-9b",
            logits_tol=1.0,
            top1_token_tol=0.2
        )
        
    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_llama_31_8b(self):
        self._test_model_correctness(
            model_id="meta-llama/Llama-3.1-8B",
            logits_tol=2.5,
            top1_token_tol=0.2
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_llama_32_1b(self):
        self._test_model_correctness(
            model_id="meta-llama/Llama-3.2-1B",
            logits_tol=2.0,
            top1_token_tol=0.3
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_llama_32_3b(self):
        self._test_model_correctness(
            model_id="meta-llama/Llama-3.2-3B",
            logits_tol=1.0,
            top1_token_tol=0.2
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
