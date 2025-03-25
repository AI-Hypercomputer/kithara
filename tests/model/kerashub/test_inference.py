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

"""Unit tests for correctness of KerasHubModel.generate() function 

Run test on a TPU VM: python -m unittest tests/model/kerashub/test_inference.py 

Note: This test suite will take around 300s in total to complete. 
"""
import unittest
import numpy as np
from transformers import AutoTokenizer
from kithara import KerasHubModel
import time
import unittest.result
from tests.test_utils import timeout
import os 

@unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
class TestModelGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):        
        cls.model = KerasHubModel.from_preset(
            "hf://google/gemma-2-2b",
            lora_rank=6
        )        
        cls.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        cls.text_prompt = "hello world"
        cls.token_prompt = np.array([5, 6, 7])


    def setUp(self):
        print(f"\nStarting test: {self._testMethodName}")
        self.start_time = time.time()

    def tearDown(self):
        duration = time.time() - self.start_time
        print(f"Completed test: {self._testMethodName} in {duration:.2f} seconds\n")

    @timeout(30)
    def test_generate_without_tokenizer(self):
        with self.assertRaises(AssertionError):
            self.model.generate(self.text_prompt)
        
        with self.assertRaises(AssertionError):
            self.model.generate(self.token_prompt, return_decoded=True, max_length=-1)

    @timeout(200)
    def test_generate_with_string_input_decoded(self):
        pred = self.model.generate(
            self.text_prompt,
            max_length=5,
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=True,
        )
        self.assertIsInstance(pred, list)
        self.assertIsInstance(pred[0], str)
        self.assertLess(len(pred[0].split(" ")), 5)
    
    @timeout(200)
    def test_generate_with_string_input_not_decoded(self):
        pred = self.model.generate(
            self.text_prompt,
            max_length=5,
            stop_token_ids=[],
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=False,
        )
        self.assertIsInstance(pred, list)
        self.assertEqual(len(pred[0]), 5)

    @timeout(200)
    def test_generate_with_string_input_strip_prompt_decoded(self):
        pred = self.model.generate(
            self.text_prompt,
            max_length=5,
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=True,
            strip_prompt=True,
        )
        self.assertIsInstance(pred[0], str)
        self.assertFalse(pred[0].startswith(self.text_prompt))

    @timeout(200)
    def test_generate_with_token_input_strip_prompt(self):
        pred = self.model.generate(
            "hello world",
            max_length=5,
            stop_token_ids=[],
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=False,
            strip_prompt=True,
        )
        self.assertIsInstance(pred, list)
        self.assertEqual(len(pred[0]), 2)


    @timeout(200)
    def test_generate_with_mutiple_string_prompts(self):
        pred = self.model.generate(
            [self.text_prompt]*10,
            max_length=5,
            stop_token_ids=[],
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=False,
            strip_prompt=False,
        )
        self.assertIsInstance(pred, list)
        self.assertEqual(len(pred), 10)
        self.assertTrue(all(len(p) == 5 for p in pred))
    
    @timeout(200)
    def test_generate_with_multiple_token_prompts(self):
        pred = self.model.generate(
            [self.token_prompt]*10,
            max_length=5,
            stop_token_ids=[],
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=False,
            strip_prompt=True,
        )
        self.assertIsInstance(pred, list)
        self.assertEqual(len(pred), 10)
        self.assertTrue(all(len(p) == 2 for p in pred))


    @timeout(200)
    def test_generate_with_tokenizer_object(self):
        pred = self.model.generate(
            self.token_prompt,
            tokenizer=self.tokenizer,
            max_length=5,
            return_decoded=True
        )
        self.assertIsInstance(pred[0], str)
    

if __name__ == '__main__':
    unittest.main(verbosity=2)
