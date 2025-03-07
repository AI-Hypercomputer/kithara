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

"""Compare model outputs between HuggingFace and MaxText model. 

This test validates that the MaxText implementations produces logits that are
numerically comparable to the reference HuggingFace implementation. 

Steps: 
    1. Load HF model into MaxText model.
    2. Get logits from HF model 
    3. Get logits from MaxText model.
    4. Compare logits 

Usage:
    Run script on single host VM: RUN_SKIPPED_TESTS=1 python -m unittest tests/model/maxtext/ckpt_compatibility/test_loading_models.py
"""

import unittest
from kithara import MaxTextModel
from tests.model.test_prompt import TEST_PROMPT
from tests.model.utils import check_arrays_match, check_predicted_tokens_match
import tests.model.utils as utils
import os 

class TestLoadingModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.MAX_TARGET_LENGTH = 512

    def get_maxtext_logits(self, model_id, input_ids):
        """Get logits from MaxText model.
        
        Args:
            model_id: HuggingFace model identifier
            input_ids: Input token IDs
            
        Returns:
            Model logits
        """
        model = MaxTextModel.from_preset(
            preset_handle=f"hf://{model_id}",
            seq_len=self.MAX_TARGET_LENGTH,
            per_device_batch_size=1,
            scan_layers=True,
            precision="float32",
        )

        # Run forward pass
        input = utils.get_maxtext_model_input(input_ids)
        logits, _ = model.stateless_call(
            model.trainable_variables,
            model.non_trainable_variables,
            input,
        )
        return logits

    def _test_model_compatibility(self, model_id, logits_tol, top1_token_tol):
        """Verify MaxText model and HF model produce the same logits. 
        
        Args:
            model_id: HuggingFace model identifier
            logits_tol: Tolerance for logits comparison
            top1_token_tol: Tolerance for top token prediction comparison
        """
        input_ids, logits_hf = utils.get_hf_logits(
            model_id, 
            TEST_PROMPT, 
            self.MAX_TARGET_LENGTH, 
            return_input_ids=True
        )
        
        # Extract logits from the first input sequence
        logits_hf = logits_hf[0, :, :]
        
        # Get MaxText logits and handle potential vocab padding
        vocab_size = logits_hf.shape[1]
        logits_maxtext = self.get_maxtext_logits(model_id, input_ids)
        logits_maxtext = logits_maxtext[0, :, :vocab_size]

        # Verify logits from the first 5 tokens match
        check_arrays_match(
            logits_hf[:5, :], 
            logits_maxtext[:5, :], 
            atol=logits_tol
        )
        
        # Verify predicted tokens match
        check_predicted_tokens_match(
            logits_hf, 
            logits_maxtext, 
            tolerance=top1_token_tol
        )
    
    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2_2b(self):
        self._test_model_compatibility(
            model_id="google/gemma-2-2b",
            logits_tol=0.5,
            top1_token_tol=0.01
        )
    
    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2_7b(self):
        self._test_model_compatibility(
            model_id="google/gemma-2-9b",
            logits_tol=1.0,
            top1_token_tol=0.02
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_llama_31_8b(self):
        self._test_model_compatibility(
            model_id="meta-llama/Llama-3.1-8B",
            logits_tol=0.1,
            top1_token_tol=0.01
        )



if __name__ == '__main__':
    unittest.main(verbosity=2)
