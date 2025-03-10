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

from kithara.distributed.sharding.models import GEMMA_LAYOUT, LLAMA_LAYOUT
from dataclasses import dataclass
from typing import ClassVar


# Layout configurations for different model architectures
@dataclass
class Layout:
    
    _mesh_types: ClassVar[dict] = {}
    
    @classmethod
    def initialize_mesh_types(cls):
        # Avoid circular import
        from kithara.model import supported_models
        cls._mesh_types = {
            supported_models.GEMMA2_2B: lambda: Layout.gemma(),
            supported_models.GEMMA2_9B: lambda: Layout.gemma(),
            supported_models.GEMMA2_27B: lambda: Layout.gemma(),
            supported_models.LLAMA31_8B: lambda: Layout.llama(),
            supported_models.LLAMA31_70B: lambda: Layout.llama(),
            supported_models.LLAMA31_405B: lambda: Layout.llama(),
            supported_models.LLAMA32_1B: lambda: Layout.llama(),
            supported_models.LLAMA32_3B: lambda: Layout.llama(),
            supported_models.LLAMA2_7B: lambda: Layout.llama(),
            supported_models.LLAMA2_13B: lambda: Layout.llama(),
            supported_models.LLAMA2_70B: lambda: Layout.llama(),
        }

    def __class_getitem__(cls, key: str):
        if not cls._mesh_types:
            cls.initialize_mesh_types()
        if key not in cls._mesh_types:
            raise KeyError(f"Unknown mesh type: {key}")
        return cls._mesh_types[key]()

    @classmethod
    def gemma(cls):
        return GEMMA_LAYOUT
    
    @classmethod
    def llama(cls):
        return LLAMA_LAYOUT
