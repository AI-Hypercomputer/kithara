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

Run test on a TPU VM: python -m unittest tests/model/hf_compatibility/test_shape_mapping.py
"""
from typing import Any, Dict, Iterable

import os
import json
import pathlib

os.environ["KERAS_BACKEND"] = "jax"

import unittest

from kithara.model import supported_models
from kithara.model.hf_compatibility import shape_mapping, qwen_model_configs

class TestModelMapping(unittest.TestCase):

    TESTDATA_DIR = pathlib.Path(".") / "tests" / "model" / "hf_compatibility" / "testdata"

    def test_qwen25_model_mapping(self):
        tests_param = (
            (
                supported_models.QWEN25_D5B,
                qwen_model_configs.qwen25_d5_config,
                self.TESTDATA_DIR / "qwen_qwen2.5-0.5b_model_shape.json"),
            (
                supported_models.QWEN25_1D5B,
                qwen_model_configs.qwen25_1d5_config,
                self.TESTDATA_DIR / "qwen_qwen2.5-1.5b_model_shape.json"),
            (
                supported_models.QWEN25_3B,
                qwen_model_configs.qwen25_3b_config,
                self.TESTDATA_DIR / "qwen_qwen2.5-3b_model_shape.json"),
            (
                supported_models.QWEN25_7B,
                qwen_model_configs.qwen25_7b_config,
                self.TESTDATA_DIR / "qwen_qwen2.5-7b_model_shape.json"))

        for model, config, golden_mapping_path in tests_param:

            with golden_mapping_path.open("r", encoding = "utf-8") as fd:
                model_shape = json.load(fd)

            self.assertDictEqual(
                shape_mapping.SHAPE_MAPPING[model](config.to_dict()),
                model_shape)
