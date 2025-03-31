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

"""Unit tests for model configs handling.

Run test on a TPU VM: python -m unittest tests/model/test_model_configs.py
"""
import os

os.environ["KERAS_BACKEND"] = "jax"

import unittest
from unittest import mock

from kithara.model.hf_compatibility import model_configs
from kithara.model import supported_models

IGNORABLE = None

class TestModelConfig(unittest.TestCase):

    def test_get_model_name_from_preset(self):
        m, h = "model_type", "num_hidden_layers"
        hs = "hidden_size"
        expected_mapping = [
            # gemma series model name mapping test
            ({m: "gemma2", h: 26}, supported_models.GEMMA2_2B),
            ({m: "gemma2", h: 42}, supported_models.GEMMA2_9B),
            ({m: "gemma2", h: 46}, supported_models.GEMMA2_27B),

            # llama series model name mapping test
            ({m: "llama", h: 32}, supported_models.LLAMA31_8B),
            ({m: "llama", h: 80}, supported_models.LLAMA31_70B),
            ({m: "llama", h: 126}, supported_models.LLAMA31_405B),
            ({m: "llama", h: 16}, supported_models.LLAMA32_1B),
            ({m: "llama", h: 28}, supported_models.LLAMA32_3B),

            #Qwen Seris model name mapping test
            ({m: "qwen2", h: 24, hs: IGNORABLE}, supported_models.QWEN25_D5B),
            ({m: "qwen2", h: 28, hs: 1536}, supported_models.QWEN25_1D5B),
            ({m: "qwen2", h: 36, hs: IGNORABLE}, supported_models.QWEN25_3B),
            ({m: "qwen2", h: 28, hs: 3584}, supported_models.QWEN25_7B),
            ({m: "qwen2", h: 48, hs: IGNORABLE}, supported_models.QWEN25_14B),
            ({m: "qwen2", h: 64, hs: IGNORABLE}, supported_models.QWEN25_32B),
            ({m: "qwen2", h: 80, hs: IGNORABLE}, supported_models.QWEN25_72B),
            ({m: "qwen2", h: -1, hs: IGNORABLE}, None),
            ({m: "non_exsit_model", h: -1}, None)
        ]

        for simulated_config, expected_model_name in expected_mapping:
            with mock.patch.object(
                model_configs,
                "load_json",
                return_value=simulated_config,
                autospec=True
            ) as mocked_load_json:
                self.assertEqual(
                    model_configs.get_model_name_from_preset_handle("dummy_handle"),
                    expected_model_name
                )
                mocked_load_json.assert_called_once_with("dummy_handle")
