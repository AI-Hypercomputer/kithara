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

"""Unit tests for qwen dense model configs handling.

Run test on a TPU VM: python -m unittest tests/model/hf_compatibility/test_qwen_model_configs.py
"""
from typing import Any, Dict, Iterable

import os
import json
import pathlib

os.environ["KERAS_BACKEND"] = "jax"

import unittest

from kithara.model.hf_compatibility import qwen_model_configs

class TestQwenModelConfig(unittest.TestCase):

    TESTDATA_DIR = pathlib.Path(".") / "tests" / "model" / "hf_compatibility" / "testdata"
    IGNORED_KEYS = {
        "architectures",  # Always "Qwen2ForCausalLM"
        "model_type",  # Will always be "qwen2"
        "torch_dtype",
        "transformers_version",
    }

    def _get_filtered_config(self, config_file_name: str,
                             ignored_keys: Iterable[str]) -> Dict[str, Any]:
        with open(self.TESTDATA_DIR / config_file_name, "r",
                  encoding="utf-8") as fd:
            raw_config = json.load(fd)
        return {k: v for k, v in raw_config.items() if k not in ignored_keys}

    def test_qwen25_d5_config(self):
        expected_config = self._get_filtered_config(
            "qwen2.5-0.5b_config.json", self.IGNORED_KEYS)
        actual_config = {k: getattr(qwen_model_configs.qwen25_d5_config, k) for k in expected_config}
        self.assertTrue(expected_config == actual_config)

    def test_qwen25_1d5_config(self):
        expected_config = self._get_filtered_config(
            "qwen2.5-1.5b_config.json", self.IGNORED_KEYS)
        actual_config = {k: getattr(qwen_model_configs.qwen25_1d5_config, k) for k in expected_config}
        self.assertTrue(expected_config == actual_config)

    def test_qwen25_3b_config(self):
        expected_config = self._get_filtered_config(
            "qwen2.5-3b_config.json", self.IGNORED_KEYS)
        actual_config = {k: getattr(qwen_model_configs.qwen25_3b_config, k) for k in expected_config}
        self.assertTrue(expected_config == actual_config)

    def test_qwen25_7b_config(self):
        expected_config = self._get_filtered_config(
            "qwen2.5-7b_config.json", self.IGNORED_KEYS)
        actual_config = {k: getattr(qwen_model_configs.qwen25_7b_config, k) for k in expected_config}
        self.assertTrue(expected_config == actual_config)

    def test_qwen25_14b_config(self):
        expected_config = self._get_filtered_config(
            "qwen2.5-14b_config.json", self.IGNORED_KEYS)
        actual_config = {k: getattr(qwen_model_configs.qwen25_14b_config, k) for k in expected_config}
        self.assertTrue(expected_config == actual_config)

    def test_qwen25_32b_config(self):
        expected_config = self._get_filtered_config(
            "qwen2.5-32b_config.json", self.IGNORED_KEYS)
        actual_config = {k: getattr(qwen_model_configs.qwen25_32b_config, k) for k in expected_config}
        self.assertTrue(expected_config == actual_config)

    def test_qwen25_72b_config(self):
        expected_config = self._get_filtered_config(
            "qwen2.5-72b_config.json", self.IGNORED_KEYS)
        actual_config = {k: getattr(qwen_model_configs.qwen25_72b_config, k) for k in expected_config}
        # print(expected_config, actual_config)
        self.assertTrue(expected_config == actual_config)
